"""把所有 ML/data cache 指向專案目錄下的 .cache/，避免污染 $HOME。

在任何 LSTM 腳本的最開頭 import 即可生效（必須在 import lerobot/torch/hf 之前）：

    import policies.lstm.scripts.setup_cache  # noqa: F401
"""

import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / ".cache"

_VARS = {
    "HF_HOME": CACHE_ROOT / "huggingface",
    "HF_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
    "HUGGINGFACE_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
    "HF_DATASETS_CACHE": CACHE_ROOT / "huggingface" / "datasets",
    "HF_LEROBOT_HOME": CACHE_ROOT / "huggingface" / "lerobot",
    "HF_LEROBOT_CALIBRATION": CACHE_ROOT / "huggingface" / "lerobot" / "calibration",
    "TRANSFORMERS_CACHE": CACHE_ROOT / "huggingface" / "transformers",
    "TORCH_HOME": CACHE_ROOT / "torch",
    "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "WANDB_DIR": CACHE_ROOT / "wandb" / "runs",
    "WANDB_CACHE_DIR": CACHE_ROOT / "wandb" / "cache",
    "WANDB_CONFIG_DIR": CACHE_ROOT / "wandb" / "config",
}

for var, path in _VARS.items():
    os.environ[var] = str(path)
    path.mkdir(parents=True, exist_ok=True)
