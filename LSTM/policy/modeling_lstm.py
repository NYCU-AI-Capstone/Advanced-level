#!/usr/bin/env python
"""LSTM (recurrent BC) policy model for ShellBench.

架構（見 implementation_plan.md §2）：
    每幀 observation ─┬─ front camera ─┐
                      └─ wrist camera ─┴─ ResNet18 encoder ─┐
                         joint state ──── MLP ──────────────┤
                                                            ▼
                                              [影像特徵 + 狀態特徵] 串接
                                                            ▼
                          上一幀 hidden ──→  LSTM（攜帶記憶）──→ 更新 hidden
                                                            ▼
                                              MSE action head → 8 維動作

兩種前向路徑：
  - forward(batch)       訓練：batch 是 [B, L, ...] 整段序列，對 L 步做 BPTT，回傳 MSE loss。
  - select_action(batch) 推論：每次餵單一幀，內部維護 hidden state 跨 episode 攜帶記憶。

normalization 由 processor pipeline 在 forward/select_action **之前**完成
（見 processor_lstm.py），所以這裡拿到的 batch 已是正規化後的 float CHW 影像。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_lstm import LSTMConfig


def _build_backbone(name: str, weights_name: str | None) -> tuple[nn.Module, int]:
    """建立 torchvision resnet backbone，移除最後的分類 fc，回傳 (網路, 特徵維度)。"""
    backbone_fn = getattr(torchvision.models, name)
    net = backbone_fn(weights=weights_name)  # torchvision 接受字串如 "IMAGENET1K_V1" 或 None
    feat_dim = net.fc.in_features  # resnet18 → 512
    net.fc = nn.Identity()
    return net, feat_dim


class LSTMPolicy(PreTrainedPolicy):
    config_class = LSTMConfig
    name = "lstm"

    def __init__(self, config: LSTMConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 相機 key（排序以確保 train/eval 順序一致）
        self.image_keys = sorted(config.image_features.keys())
        n_cam = len(self.image_keys)

        # 影像 encoder（所有相機共用同一個 backbone）
        self.backbone, img_feat_dim = _build_backbone(
            config.vision_backbone, config.pretrained_backbone_weights
        )

        # proprioception MLP（可有可無）
        self.has_state = config.robot_state_feature is not None
        state_out = 0
        if self.has_state:
            state_dim = config.robot_state_feature.shape[0]
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, config.state_feature_dim),
                nn.ReLU(),
            )
            state_out = config.state_feature_dim

        # LSTM
        lstm_input_dim = n_cam * img_feat_dim + state_out
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0.0,
        )

        # action head（簡單 MSE 回歸）
        action_dim = config.action_feature.shape[0]
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, action_dim),
        )

        # 推論用的 hidden state（跨 select_action 呼叫攜帶；命名刻意避開 eval
        # 的 _clear_cached_actions() 會掃到的 action_queue 之類欄位）
        self._lstm_state: tuple[Tensor, Tensor] | None = None

        # 推論 stride：每 N 次 select_action 呼叫才真正更新 hidden state、算新動作，
        # 其餘次回傳上一次的動作。讓 eval 端不需要任何修改——它照常每幀呼叫 policy，
        # 但 policy 自己按 obs_stride 節奏處理，跟訓練時的時間動態一致。
        self._stride = config.obs_stride
        self._stride_counter = 0
        self._cached_action: Tensor | None = None

        self.reset()

    # ------------------------------------------------------------------ helpers

    def _encode_frames(self, images: Tensor) -> Tensor:
        """images: [..., C, H, W]（任意前置維度）→ [..., feat_dim]。

        會 resize 到 config.image_size，並對 backbone 做選配的 gradient checkpointing。
        """
        lead_shape = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])  # [N, C, H, W]
        flat = F.interpolate(
            flat, size=(self.config.image_size, self.config.image_size),
            mode="bilinear", align_corners=False,
        )
        if self.training and self.config.use_gradient_checkpointing:
            feat = torch.utils.checkpoint.checkpoint(self.backbone, flat, use_reentrant=False)
        else:
            feat = self.backbone(flat)
        return feat.reshape(*lead_shape, feat.shape[-1])  # [..., feat_dim]

    def _encode_observation(self, batch: dict[str, Tensor]) -> Tensor:
        """把一個 batch 的觀測編碼成 LSTM 的輸入特徵。

        支援兩種形狀：
          - 訓練：影像 [B, L, C, H, W]、狀態 [B, L, state]  → 回傳 [B, L, feat]
          - 推論：影像 [B, C, H, W]、狀態 [B, state]        → 回傳 [B, 1, feat]
        """
        first_img = batch[self.image_keys[0]]
        has_time = first_img.dim() == 5  # [B, L, C, H, W] 才有時間維

        cam_feats = []
        for key in self.image_keys:
            img = batch[key]
            if not has_time:
                img = img.unsqueeze(1)  # [B, C, H, W] → [B, 1, C, H, W]
            cam_feats.append(self._encode_frames(img))  # [B, L, feat]
        feat = torch.cat(cam_feats, dim=-1)  # [B, L, n_cam*feat]

        if self.has_state:
            state = batch[OBS_STATE]
            if not has_time:
                state = state.unsqueeze(1)
            feat = torch.cat([feat, self.state_mlp(state)], dim=-1)
        return feat  # [B, L, lstm_input_dim]

    # ------------------------------------------------------------------ API

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _save_pretrained(self, save_directory: Path) -> None:
        """覆寫預設存檔，修掉 nn.LSTM + safetensors 的共用 storage 問題。

        nn.LSTM 在 cuDNN 下會把 weight_ih_l0 / weight_hh_l0 / bias_* flatten 進同一塊
        storage，safetensors 拒絕儲存共用 storage 的 tensor。存檔前各 clone 一份打斷共用即可。
        """
        self.config._save_pretrained(save_directory)
        state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        save_file(state, str(Path(save_directory) / SAFETENSORS_SINGLE_FILE), metadata={"format": "pt"})

    def reset(self) -> None:
        """每個 episode 開頭呼叫，清空 LSTM 記憶與 stride 狀態。"""
        self._lstm_state = None
        self._stride_counter = 0
        self._cached_action = None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """訓練：對整段序列做 BPTT，回傳 masked MSE loss。"""
        feats = self._encode_observation(batch)          # [B, L, input]
        out, _ = self.lstm(feats)                          # [B, L, hidden]
        pred = self.action_head(out)                       # [B, L, action_dim]
        target = batch[ACTION]                             # [B, L, action_dim]

        per_step = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # [B, L]
        pad_key = f"{ACTION}_is_pad"
        if pad_key in batch:
            valid = (~batch[pad_key]).float()              # [B, L]，padding 幀不算 loss
            loss = (per_step * valid).sum() / valid.sum().clamp(min=1.0)
        else:
            loss = per_step.mean()
        return loss, {"mse_loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """抽象介面要求；recurrent policy 逐步輸出，這裡回傳單一動作（加 chunk 維）。"""
        return self.select_action(batch, **kwargs).unsqueeze(1)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """推論：餵單一幀，依 obs_stride 決定是否更新 hidden state。

        stride=1（預設）：每次都處理。
        stride=N>1：每 N 次呼叫才真正 encode+LSTM forward（更新記憶、算新動作），
        其餘次直接回傳上一次的動作。這讓 eval 端不用改——照常每幀呼叫 policy，
        但 policy 按 stride 節奏處理，跟訓練的 observation_delta_indices 一致。
        """
        should_process = (self._stride_counter % self._stride == 0)
        self._stride_counter += 1

        if should_process or self._cached_action is None:
            feat = self._encode_observation(batch)             # [B, 1, input]
            out, self._lstm_state = self.lstm(feat, self._lstm_state)
            self._cached_action = self.action_head(out[:, -1]) # [B, action_dim]

        return self._cached_action
