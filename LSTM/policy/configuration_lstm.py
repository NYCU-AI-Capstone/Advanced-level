#!/usr/bin/env python
"""LSTM (recurrent BC) policy configuration for ShellBench.

這個 config 讓 LeRobot 認得 `--policy.type=lstm`，並定義：
  1. 超參數（LSTM / encoder / 序列長度）
  2. 訓練時的「序列取樣」—— 透過 observation_delta_indices / action_delta_indices
     告訴 LeRobot dataset 每個 training sample 要取哪幾幀（這就是 BPTT 的時間窗）。

對應 implementation_plan.md §7.1。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("lstm")
@dataclass
class LSTMConfig(PreTrainedConfig):
    """Recurrent (LSTM) behaviour-cloning policy。

    記憶機制：影像/狀態 → encoder → LSTM hidden state（跨整段 episode 攜帶）→ action。
    這是「真的有跨整段 episode memory」的對照組，對比看得有限的 ACT / Diffusion。
    """

    # --- 序列取樣（BPTT 時間窗）------------------------------------------------
    # seq_len (L): 每個 training sample 涵蓋幾個時間步；LSTM 對這 L 步做 BPTT。
    # obs_stride: 時間步之間的跳幀間隔。stride=1 = 每幀都取。
    #   ⚠️ stride > 1 時，評估端也必須以相同 stride 餵幀進 LSTM，否則 train/eval
    #      的時間動態不一致。第一版固定 stride=1（評估端每幀都 step LSTM）。
    seq_len: int = 200
    obs_stride: int = 1

    # --- 影像 encoder ----------------------------------------------------------
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "IMAGENET1K_V1"  # None = 不載 pretrained
    image_size: int = 160  # 影像 resize 邊長（96 太小看不到球；160 在 4090 上可行）

    # --- LSTM ------------------------------------------------------------------
    hidden_size: int = 512
    num_lstm_layers: int = 2
    state_feature_dim: int = 128  # proprioception 經一層 MLP 後的維度
    dropout: float = 0.1

    # --- 記憶體節省 ------------------------------------------------------------
    # 對 backbone 的逐幀前向做 gradient checkpointing（用重算換記憶體），
    # 讓長序列 + 多相機在 4090 24GB 上跑得動。
    use_gradient_checkpointing: bool = True

    # --- normalization（由 processor pipeline 套用，見 processor_lstm.py）------
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- optimizer preset ------------------------------------------------------
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` 必須是 torchvision 的 resnet 變體，得到 {self.vision_backbone}。"
            )
        if self.seq_len < 2:
            raise ValueError(f"`seq_len` 至少要 2 才有時序可言，得到 {self.seq_len}。")
        if self.obs_stride < 1:
            raise ValueError(f"`obs_stride` 必須 >= 1，得到 {self.obs_stride}。")

    # --- PreTrainedConfig 抽象介面 ---------------------------------------------

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("LSTM policy 至少需要一個相機影像輸入（front / wrist）。")
        if self.action_feature is None:
            raise ValueError("LSTM policy 需要 'action' 作為輸出 feature。")

    @property
    def observation_delta_indices(self) -> list:
        # 一段長度 seq_len、以當前幀 (0) 結尾、間隔 obs_stride 的時間窗。
        # 例：seq_len=200, stride=1 → [-199, -198, ..., 0]（長度 200）
        #     seq_len=64,  stride=8 → [-504, -496, ..., 0]（長度 64，涵蓋 ~520 幀）
        return list(range(-(self.seq_len - 1) * self.obs_stride, 1, self.obs_stride))

    @property
    def action_delta_indices(self) -> list:
        # action 與 observation 對齊：每個時間步觀測 o_t、預測動作 a_t。
        return list(range(-(self.seq_len - 1) * self.obs_stride, 1, self.obs_stride))

    @property
    def reward_delta_indices(self) -> None:
        return None  # 純 imitation learning，不用 reward
