#!/usr/bin/env python
"""評估 wrapper —— 註冊 LSTM policy 後跑 ShellBench 既有的 eval_shell_game.py。

零侵入（C2）：不改 scripts/eval_shell_game.py，用 runpy 以 __main__ 執行它，
但在執行前先 import 我們的 register（把 lstm 接進 lerobot 的 factory）。
用法跟 eval_shell_game.py 完全一樣：

    cd /workspace/aicapstone
    python LSTM/scripts/eval_lstm.py \\
        --task HCIS-ShellGame-SingleArm-v0 --device cuda --enable_cameras \\
        --policy_backend lerobot --policy_type lerobot-lstm \\
        --policy_checkpoint_path LSTM/outputs/ns0_v1/checkpoints/last/pretrained_model \\
        --policy_action_horizon 1 --num_episodes 20 --num_cups 3 --num_shuffles 0 \\
        --output_json LSTM/outputs/ns0_v1/metrics.json

備註：eval_shell_game.py 用 `get_policy_class("lstm")` 載入 policy，所以這個 patch 必須
在它 import lerobot factory 之前生效 —— 本 wrapper 先 import register 即滿足。
"""

import pathlib
import runpy
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import LSTM.policy.register  # noqa: F401,E402  必須在跑 eval 前：註冊 + factory patch

_EVAL_SCRIPT = REPO / "scripts" / "eval_shell_game.py"

if __name__ == "__main__":
    # 以 __main__ 執行既有 eval 腳本；它的 argparse 會讀我們這層傳進來的 sys.argv。
    runpy.run_path(str(_EVAL_SCRIPT), run_name="__main__")
