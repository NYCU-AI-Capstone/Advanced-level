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
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)


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
    # Action head 額外直接查看最近幾個原始 frame（1=當前，2=上一幀+當前）。
    # 0 僅供載入舊 checkpoint，代表沿用只有 hidden state 的舊 action head。
    current_obs_frames: int = 0
    # 每次規劃輸出幾個連續 action。預設 1 以相容舊 checkpoint。
    action_horizon: int = 1

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
    backbone_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 5_000
    scheduler_decay_steps: int = 200_000
    scheduler_decay_lr: float = 1e-5

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
        if self.current_obs_frames < 0:
            raise ValueError(
                f"`current_obs_frames` 必須 >= 0，得到 {self.current_obs_frames}。"
            )
        if self.action_horizon < 1:
            raise ValueError(f"`action_horizon` 必須 >= 1，得到 {self.action_horizon}。")
        if not 0 < self.backbone_lr <= self.optimizer_lr:
            raise ValueError("`backbone_lr` 必須 > 0 且不大於 `optimizer_lr`。")
        if self.scheduler_warmup_steps < 0:
            raise ValueError("`scheduler_warmup_steps` 必須 >= 0。")
        if self.scheduler_decay_steps <= 0:
            raise ValueError("`scheduler_decay_steps` 必須 > 0。")
        if not 0 < self.scheduler_decay_lr <= self.optimizer_lr:
            raise ValueError(
                "`scheduler_decay_lr` 必須 > 0 且不大於 `optimizer_lr`。"
            )

    # --- PreTrainedConfig 抽象介面 ---------------------------------------------

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("LSTM policy 至少需要一個相機影像輸入（front / wrist）。")
        if self.action_feature is None:
            raise ValueError("LSTM policy 需要 'action' 作為輸出 feature。")

    @property
    def observation_delta_indices(self) -> list:
        # Memory 使用稀疏時間點；action shortcut 另外取每個時間點之前的密集局部幀。
        indices = set(self.memory_delta_indices)
        for memory_index in self.memory_delta_indices:
            for offset in range(self.current_obs_frames):
                indices.add(memory_index - offset)
        return sorted(indices)

    @property
    def memory_delta_indices(self) -> list[int]:
        return list(range(-(self.seq_len - 1) * self.obs_stride, 1, self.obs_stride))

    @property
    def memory_observation_positions(self) -> list[int]:
        lookup = {delta: index for index, delta in enumerate(self.observation_delta_indices)}
        return [lookup[delta] for delta in self.memory_delta_indices]

    @property
    def current_observation_positions(self) -> list[list[int]]:
        if self.current_obs_frames == 0:
            return []
        lookup = {delta: index for index, delta in enumerate(self.observation_delta_indices)}
        return [
            [
                lookup[memory_index - offset]
                for offset in reversed(range(self.current_obs_frames))
            ]
            for memory_index in self.memory_delta_indices
        ]

    @property
    def action_delta_indices(self) -> list:
        # 每個 memory 時間點預測從當下開始的 action chunk：a_t ... a_{t+H-1}。
        return [
            memory_index + action_offset
            for memory_index in self.memory_delta_indices
            for action_offset in range(self.action_horizon)
        ]

    @property
    def reward_delta_indices(self) -> None:
        return None  # 純 imitation learning，不用 reward
