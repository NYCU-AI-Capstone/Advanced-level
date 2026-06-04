"""Project-local cache configuration helpers.

The training/datagen stack pulls data through several libraries. Keep their
implicit downloads under the repository instead of the user's home directory.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_project_cache(project_root: str | Path | None = None) -> Path:
    """Point common ML/data caches at ``<project>/.cache``.

    ``AICAPSTONE_CACHE_DIR`` may be set to override the cache root. All other
    cache variables are intentionally overwritten so subprocesses agree on one
    location.
    """

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    cache_root = Path(os.environ.get("AICAPSTONE_CACHE_DIR", root / ".cache")).resolve()

    paths = {
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
        "HF_LEROBOT_HOME": cache_root / "huggingface" / "lerobot",
        "HF_LEROBOT_CALIBRATION": cache_root / "huggingface" / "lerobot" / "calibration",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "WANDB_DIR": cache_root / "wandb" / "runs",
        "WANDB_CACHE_DIR": cache_root / "wandb" / "cache",
        "WANDB_CONFIG_DIR": cache_root / "wandb" / "config",
    }

    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)

    return cache_root


def project_cache_env(project_root: str | Path | None = None) -> dict[str, str]:
    """Return an environment dict after applying project-local cache settings."""

    configure_project_cache(project_root)
    return os.environ.copy()
