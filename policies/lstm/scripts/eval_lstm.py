#!/usr/bin/env python
"""評估 wrapper —— 註冊 LSTM policy 後跑 ShellBench 既有的 eval_shell_game.py。

零侵入（C2）：不改 scripts/eval_shell_game.py，用 runpy 以 __main__ 執行它，
但在執行前先 import 我們的 register（把 lstm 接進 lerobot 的 factory）。

用法 1：直接從 training output 自動推 evaluation 參數：

    cd /workspace/aicapstone
    python policies/lstm/scripts/eval_lstm.py --run_dir outputs/lstm/num_shuffles-3

預設載入 checkpoints/best；若不存在則依序回退到 last、最新數字 checkpoint。

用法 2：跟 eval_shell_game.py 完全一樣：

    cd /workspace/aicapstone
    python policies/lstm/scripts/eval_lstm.py \
        --task HCIS-ShellGame-SingleArm-v0 --device cuda --enable_cameras \
        --policy_backend lerobot --policy_type lerobot-lstm \
        --policy_checkpoint_path outputs/lstm/ns0_v1/checkpoints/last/pretrained_model \
        --policy_action_horizon 1 --num_episodes 20 --num_cups 3 --num_shuffles 0 \
        --output_json outputs/lstm/ns0_v1/metrics.json

備註：eval_shell_game.py 用 `get_policy_class("lstm")` 載入 policy，所以這個 patch 必須
在它 import lerobot factory 之前生效 —— 本 wrapper 先 import register 即滿足。
"""

import argparse
import os
import pathlib
import re
import runpy
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"


def _ensure_uv_python() -> None:
    """Re-exec with the repository uv environment when invoked by plain python."""
    uv_python = REPO / ".venv" / "bin" / "python"
    if not uv_python.is_file():
        return
    if pathlib.Path(sys.executable).resolve() == uv_python.resolve():
        return
    os.execv(
        str(uv_python),
        [str(uv_python), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]],
    )


_ensure_uv_python()


def _add_isaaclab_source_paths() -> None:
    """Allow eval from a uv env when IsaacLab is present as a repo submodule."""
    source_root = REPO / "dependencies" / "IsaacLab" / "source"
    for package_dir in (
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "isaaclab_mimic",
        "isaaclab_rl",
    ):
        path = source_root / package_dir
        if path.is_dir():
            sys.path.insert(0, str(path))


_add_isaaclab_source_paths()


def _check_isaaclab_available() -> None:
    try:
        import isaaclab  # noqa: F401
    except ModuleNotFoundError as exc:
        source_root = REPO / "dependencies" / "IsaacLab" / "source"
        raise ModuleNotFoundError(
            "IsaacLab is not importable. Run `git submodule update --init --recursive` "
            "from the repo root, then `uv sync` again. Expected IsaacLab source at "
            f"{source_root}."
        ) from exc


_check_isaaclab_available()

import policies.lstm.scripts.setup_cache  # noqa: F401,E402  必須在 lerobot/torch 之前
import policies.lstm.policy.register  # noqa: F401,E402  必須在跑 eval 前：註冊 + factory patch

_EVAL_SCRIPT = REPO / "scripts" / "eval_shell_game.py"


def _has_arg(args: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def _read_train_config(run_dir: pathlib.Path, checkpoint: pathlib.Path) -> dict:
    """Read config from the run root or directly from a saved checkpoint."""
    import json
    import yaml

    yaml_path = run_dir / "train_config.yaml"
    if yaml_path.is_file():
        with yaml_path.open() as f:
            return yaml.safe_load(f) or {}

    for json_path in (
        checkpoint / "train_config.json",
        checkpoint / "config.json",
    ):
        if json_path.is_file():
            with json_path.open() as f:
                return json.load(f) or {}

    raise FileNotFoundError(
        "Could not find training config. Checked "
        f"{yaml_path}, {checkpoint / 'train_config.json'}, and "
        f"{checkpoint / 'config.json'}"
    )


def _infer_eval_params(run_dir: pathlib.Path, cfg: dict) -> dict[str, str]:
    """Infer ShellBench eval knobs from train_config.yaml and output naming."""
    text = " ".join(
        str(x)
        for x in (
            run_dir.name,
            cfg.get("output_dir", ""),
            (cfg.get("dataset") or {}).get("repo_id", ""),
            cfg.get("job_name", ""),
        )
    )

    policy_cfg = cfg.get("policy") or cfg
    params = {
        "task": "HCIS-ShellGame-SingleArm-v0",
        "device": "cuda",
        "policy_backend": "lerobot",
        "policy_type": "lerobot-lstm",
        "policy_action_horizon": str(policy_cfg.get("action_horizon", 1)),
        "num_episodes": str((cfg.get("eval") or {}).get("n_episodes", 20)),
        "num_cups": "3",
        "num_shuffles": "2",
        "shuffle_speed": "1.0",
        "reveal_frames": "50",
        "cover_frames": "10",
        "shuffle_per_swap_frames": "30",
        "act_frames": "150",
        "max_act_steps": "600",
        "output_json": str(run_dir / "metrics.json"),
    }

    match = re.search(r"num_shuffles[-_](\d+)", text)
    if match:
        params["num_shuffles"] = match.group(1)

    match = re.search(r"num_cups[-_](\d+)", text)
    if match:
        params["num_cups"] = match.group(1)

    match = re.search(r"shuffle_speed[-_]([0-9.]+)", text)
    if match:
        params["shuffle_speed"] = match.group(1).rstrip(".")

    return params


def _resolve_checkpoint(run_dir: pathlib.Path, checkpoint_step: str | None) -> pathlib.Path:
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints}")

    if checkpoint_step:
        step_dir = checkpoints / checkpoint_step
        if not step_dir.is_dir() and checkpoint_step.isdigit():
            target_step = int(checkpoint_step)
            step_dir = next(
                (
                    path
                    for path in checkpoints.iterdir()
                    if path.is_dir() and path.name.isdigit() and int(path.name) == target_step
                ),
                step_dir,
            )
        checkpoint = step_dir / "pretrained_model"
    else:
        checkpoint = checkpoints / "best" / "pretrained_model"
        if not checkpoint.exists():
            checkpoint = checkpoints / "last" / "pretrained_model"
        if not checkpoint.exists():
            steps = sorted(
                (p for p in checkpoints.iterdir() if p.is_dir() and p.name.isdigit()),
                key=lambda p: int(p.name),
            )
            if not steps:
                raise FileNotFoundError(f"No checkpoints found under {checkpoints}")
            checkpoint = steps[-1] / "pretrained_model"

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _expand_run_dir_args() -> None:
    """Translate our compact --run_dir mode into eval_shell_game.py arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run_dir", type=pathlib.Path)
    parser.add_argument("--checkpoint_step", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    compact, passthrough = parser.parse_known_args(sys.argv[1:])

    if compact.run_dir is None:
        return

    run_dir = compact.run_dir
    checkpoint = _resolve_checkpoint(run_dir, compact.checkpoint_step)
    cfg = _read_train_config(run_dir, checkpoint)
    params = _infer_eval_params(run_dir, cfg)
    params["policy_checkpoint_path"] = str(checkpoint)

    expanded = []
    for name, value in params.items():
        flag = f"--{name}"
        if not _has_arg(passthrough, flag):
            expanded.extend([flag, value])

    for flag in ("--enable_cameras", "--headless"):
        if not _has_arg(passthrough, flag):
            expanded.append(flag)

    sys.argv[1:] = expanded + passthrough
    print("[eval_lstm] compact --run_dir expanded to:")
    print("[eval_lstm] " + " ".join(sys.argv[1:]))
    if compact.dry_run:
        raise SystemExit(0)


if __name__ == "__main__":
    _expand_run_dir_args()
    # 以 __main__ 執行既有 eval 腳本；它的 argparse 會讀我們這層傳進來的 sys.argv。
    runpy.run_path(str(_EVAL_SCRIPT), run_name="__main__")
