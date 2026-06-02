#!/usr/bin/env python
"""訓練 wrapper —— 註冊 LSTM policy 後直接呼叫 lerobot-train。

這是「不 fork lerobot」整合的訓練端入口（見 implementation_plan.md §4.2）。
用法跟 lerobot-train 完全一樣，只是先 import 我們的 plugin：

    cd /workspace/aicapstone
    python LSTM/scripts/train_lstm.py \\
        --policy.type=lstm \\
        --dataset.repo_id=johnnyli1220/shellbench-num_shuffles-0 \\
        --policy.device=cuda --batch_size=2 --steps=5000 \\
        --output_dir=LSTM/outputs/ns0_v1

所有 lerobot-train 的 flag 都能用（透過 draccus 直接 pass-through）。
"""

import pathlib
import sys

# 讓 `python LSTM/scripts/train_lstm.py` 能 import LSTM（把 repo root 加進 sys.path）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import LSTM.policy.register  # noqa: F401,E402  必須在呼叫 lerobot 前 import：註冊 + factory patch
from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()  # @parser.wrap() 會解析 sys.argv 裡的訓練 flag
