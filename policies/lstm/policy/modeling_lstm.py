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

from collections import deque
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

    def __init__(self, config: LSTMConfig, **kwargs):
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

        # action head：長期 hidden state，可選擇再串接最近 N 個 observation features。
        action_dim = config.action_feature.shape[0]
        action_input_dim = config.hidden_size + config.current_obs_frames * lstm_input_dim
        self.action_head = nn.Sequential(
            nn.Linear(action_input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, action_dim * config.action_horizon),
        )

        # cup classification head（解決 MSE mode averaging：強制 LSTM 學會
        # 根據記憶區分不同杯子，提供離散的梯度信號）
        self.cup_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.num_cups),
        )

        # 推論用的 hidden state（跨 select_action 呼叫攜帶；命名刻意避開 eval
        # 的 _clear_cached_actions() 會掃到的 action_queue 之類欄位）
        self._lstm_state: tuple[Tensor, Tensor] | None = None
        self._cup_logits: Tensor | None = None

        # 推論 stride：每 N 次 select_action 呼叫才把 hidden state 往前推進一次。
        # 每次呼叫仍會用當前 observation + 最近的 hidden state 產生 action，避免 act 階段
        # 連續多幀沿用同一個 action，讓抓取控制保有逐幀閉迴路修正。
        self._stride = config.obs_stride
        self._stride_counter = 0
        self._current_features: deque[Tensor] = deque(
            maxlen=max(1, config.current_obs_frames)
        )
        self._action_queue: deque[Tensor] = deque()

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

    def get_optim_params(self) -> list[dict]:
        """Use a smaller LR for the visual backbone while training task heads faster."""
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(param) for param in backbone_params}
        model_params = [param for param in self.parameters() if id(param) not in backbone_ids]
        # Keep the main model first because LeRobot logs optimizer.param_groups[0]["lr"].
        return [
            {"params": model_params, "lr": self.config.optimizer_lr, "name": "model"},
            {"params": backbone_params, "lr": self.config.backbone_lr, "name": "backbone"},
        ]

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
        self._current_features.clear()
        self._action_queue.clear()

    def reset_action_queue(self) -> None:
        """Discard a pending action chunk without clearing recurrent memory."""
        self._action_queue.clear()

    def _action_input(self, memory: Tensor, current: Tensor | None = None) -> Tensor:
        if self.config.current_obs_frames == 0:
            return memory
        if current is None:
            raise ValueError("current observation features are required by this checkpoint")
        return torch.cat([memory, current], dim=-1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """訓練：對整段序列做 BPTT，回傳 masked MSE loss。"""
        feats = self._encode_observation(batch)            # [B, Q, input]
        memory_positions = torch.as_tensor(
            self.config.memory_observation_positions, device=feats.device
        )
        memory_feats = feats.index_select(1, memory_positions)  # [B, L, input]
        out, _ = self.lstm(memory_feats)                    # [B, L, hidden]

        current = None
        if self.config.current_obs_frames > 0:
            positions = torch.as_tensor(
                self.config.current_observation_positions, device=feats.device
            )
            current = feats[:, positions, :].flatten(start_dim=2)  # [B, L, N*input]
        batch_size, sequence_length = out.shape[:2]
        action_dim = self.config.action_feature.shape[0]
        pred = self.action_head(self._action_input(out, current)).reshape(
            batch_size, sequence_length, self.config.action_horizon, action_dim
        )
        target = batch[ACTION].reshape(
            batch_size, sequence_length, self.config.action_horizon, action_dim
        )

        per_action = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # [B, L, H]
        pad_key = f"{ACTION}_is_pad"
        if pad_key in batch:
            valid = (~batch[pad_key]).reshape(
                batch_size, sequence_length, self.config.action_horizon
            ).float()
            loss = (per_action * valid).sum() / valid.sum().clamp(min=1.0)
        else:
            loss = per_action.mean()
        return loss, {"mse_loss": loss.item()}

    def _step_observation(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        """Consume one frame, update recurrent memory on stride, and return action features."""
        should_process = (self._stride_counter % self._stride == 0)
        self._stride_counter += 1

        feat = self._encode_observation(batch)             # [B, 1, input]
        out, next_state = self.lstm(feat, self._lstm_state)
        if should_process or self._lstm_state is None:
            self._lstm_state = next_state

        current = None
        if self.config.current_obs_frames > 0:
            feature = feat[:, -1]
            self._current_features.append(feature)
            while len(self._current_features) < self.config.current_obs_frames:
                self._current_features.appendleft(feature)
            current = torch.cat(list(self._current_features), dim=-1)
        return out[:, -1], current

    def _make_action_chunk(self, memory: Tensor, current: Tensor | None) -> Tensor:
        batch_size = memory.shape[0]
        action_dim = self.config.action_feature.shape[0]
        return self.action_head(self._action_input(memory, current)).reshape(
            batch_size, self.config.action_horizon, action_dim
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Consume the current frame and predict a fresh [B, H, action_dim] chunk."""
        self._action_queue.clear()
        memory, current = self._step_observation(batch)
        return self._make_action_chunk(memory, current)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Consume every observation, but replan actions only when the chunk queue is empty."""
        memory, current = self._step_observation(batch)
        if not self._action_queue:
            chunk = self._make_action_chunk(memory, current)
            self._action_queue.extend(chunk.unbind(dim=1))
        return self._action_queue.popleft()
