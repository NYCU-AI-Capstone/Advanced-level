#!/usr/bin/env python
"""LSTM plugin 煙霧測試（在容器內跑，不需 dataset / GPU）。

驗證三件事：
  1. 註冊：lerobot 認得 --policy.type=lstm，get_policy_class('lstm') 回傳 LSTMPolicy
  2. 訓練路徑：forward 對 [B, L, ...] 序列算出 loss、BPTT backward 成功
  3. 推論路徑：select_action 回傳單一動作、hidden state 跨呼叫攜帶（記憶機制）

用法（在容器內）：
    cd /workspace/aicapstone && python LSTM/scripts/smoke_test.py
"""

import pathlib
import sys

# 讓 `python LSTM/scripts/smoke_test.py` 也能 import LSTM（把 repo root 加進 sys.path）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

import LSTM.policy.register  # noqa: F401,E402  觸發註冊 + factory patch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class

from LSTM.policy.configuration_lstm import LSTMConfig
from LSTM.policy.modeling_lstm import LSTMPolicy


def test_registration() -> None:
    assert get_policy_class("lstm") is LSTMPolicy
    choices = PreTrainedConfig.get_known_choices()
    assert "lstm" in choices, f"'lstm' 不在 draccus 選項中: {list(choices)}"
    print("[1/4] 註冊 OK：--policy.type=lstm 可被解析，get_policy_class('lstm') -> LSTMPolicy")


def _dummy_config() -> LSTMConfig:
    cfg = LSTMConfig(
        device="cpu", pretrained_backbone_weights=None, image_size=32,
        seq_len=4, hidden_size=64, num_lstm_layers=1, state_feature_dim=16,
        use_gradient_checkpointing=False,
    )
    cfg.input_features = {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return cfg


def test_train_path() -> None:
    policy = LSTMPolicy(_dummy_config())
    policy.train()
    b, length = 2, 4
    batch = {
        "observation.images.front": torch.rand(b, length, 3, 64, 64),
        "observation.images.wrist": torch.rand(b, length, 3, 64, 64),
        "observation.state": torch.rand(b, length, 9),
        "action": torch.rand(b, length, 8),
        "action_is_pad": torch.zeros(b, length, dtype=torch.bool),
    }
    loss, info = policy.forward(batch)
    loss.backward()
    print(f"[2/4] 訓練路徑 OK：forward loss={loss.item():.4f}，BPTT backward 成功")


def test_eval_path() -> None:
    policy = LSTMPolicy(_dummy_config())
    policy.eval()
    policy.reset()
    b = 2
    step = {
        "observation.images.front": torch.rand(b, 3, 64, 64),
        "observation.images.wrist": torch.rand(b, 3, 64, 64),
        "observation.state": torch.rand(b, 9),
    }
    with torch.inference_mode():
        a1 = policy.select_action(step)
        policy.select_action(step)
    assert tuple(a1.shape) == (b, 8), a1.shape
    assert policy._lstm_state is not None, "hidden state 未被攜帶"
    print(f"[3/4] 推論路徑 OK：select_action -> {tuple(a1.shape)}，hidden state 跨呼叫攜帶")


def test_save_load() -> None:
    import tempfile

    policy = LSTMPolicy(_dummy_config())
    with tempfile.TemporaryDirectory() as d:
        policy.save_pretrained(d)  # 會崩在這裡如果 nn.LSTM + safetensors 共用 storage 沒修
        reloaded = LSTMPolicy.from_pretrained(d, config=policy.config)
    assert reloaded is not None
    print("[4/4] 存檔/重載 OK：save_pretrained + from_pretrained 成功（LSTM safetensors 問題已修）")


if __name__ == "__main__":
    test_registration()
    test_train_path()
    test_eval_path()
    test_save_load()
    print("\n✅ ALL SMOKE TESTS PASSED")
