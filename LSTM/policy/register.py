#!/usr/bin/env python
"""把 LSTM policy 註冊進 LeRobot —— import 這個模組即完成所有接線。

背景（見 implementation_plan.md §5）：LeRobot 0.4.2 的
`get_policy_class` / `make_pre_post_processors` 是寫死的 if/elif，沒有 plugin 機制。
所以除了用 `@PreTrainedConfig.register_subclass("lstm")` 讓 `--policy.type=lstm`
能被 draccus 解析外，還要 monkeypatch 這兩個 factory，讓它們認得 lstm。

用法：任何進程（train wrapper / eval wrapper）在呼叫 lerobot 前先
    import LSTM.policy.register
即可。重複 import 是安全的（patch 有 idempotent 保護）。
"""

from __future__ import annotations

import sys

import lerobot.policies.factory as _factory

# import 即觸發 @register_subclass("lstm")，讓 --policy.type=lstm 可被解析
from .configuration_lstm import LSTMConfig
from .modeling_lstm import LSTMPolicy
from .processor_lstm import make_lstm_pre_post_processors

_PATCHED_FLAG = "_shellbench_lstm_patched"


def _install_patches() -> None:
    if getattr(_factory, _PATCHED_FLAG, False):
        return

    _orig_get_policy_class = _factory.get_policy_class
    _orig_make_pp = _factory.make_pre_post_processors

    def get_policy_class(name: str):
        if name == "lstm":
            return LSTMPolicy
        return _orig_get_policy_class(name)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        # 從頭建立（非載入）且是 LSTMConfig 時，走我們的 processor factory。
        if pretrained_path is None and isinstance(policy_cfg, LSTMConfig):
            return make_lstm_pre_post_processors(
                config=policy_cfg, dataset_stats=kwargs.get("dataset_stats")
            )
        # 載入既有 checkpoint 的路徑：lerobot 原本就只靠檔案、不分 policy 型別，沿用即可。
        return _orig_make_pp(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    _factory.get_policy_class = get_policy_class
    _factory.make_pre_post_processors = make_pre_post_processors

    # 消費端（如 lerobot_train、eval_shell_game）常用
    # `from lerobot.policies.factory import make_pre_post_processors` 把名字綁進自己的
    # namespace。若它們已先 import，rebind 它們那份才會生效（順序無關保護）。
    for module in list(sys.modules.values()):
        if module is None or module is _factory:
            continue
        if getattr(module, "get_policy_class", None) is _orig_get_policy_class:
            module.get_policy_class = get_policy_class
        if getattr(module, "make_pre_post_processors", None) is _orig_make_pp:
            module.make_pre_post_processors = make_pre_post_processors

    setattr(_factory, _PATCHED_FLAG, True)


_install_patches()

__all__ = ["LSTMConfig", "LSTMPolicy", "make_lstm_pre_post_processors"]
