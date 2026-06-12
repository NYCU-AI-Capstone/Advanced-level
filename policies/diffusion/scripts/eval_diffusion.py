#!/usr/bin/env python
"""Evaluate a Diffusion checkpoint from a training run directory.

Example:
    python policies/diffusion/scripts/eval_diffusion.py \
        --run_dir outputs/diffusion/shellbench-shuffle_speed-2.0 \
        --num_episodes 100

Compact mode automatically selects a checkpoint, reads its saved config, infers
ShellBench task parameters, and runs Isaac Sim headlessly using the repo uv env.
"""

from __future__ import annotations

import argparse
import json
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
        raise ModuleNotFoundError(
            "IsaacLab is not importable from the uv environment. Run `uv sync` "
            "and ensure the IsaacLab submodule is initialized."
        ) from exc


_check_isaaclab_available()

import policies.lstm.scripts.setup_cache  # noqa: F401,E402

_EVAL_SCRIPT = REPO / "scripts" / "eval_shell_game.py"


def _has_arg(args: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


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
                (path for path in checkpoints.iterdir() if path.is_dir() and path.name.isdigit()),
                key=lambda path: int(path.name),
            )
            if not steps:
                raise FileNotFoundError(f"No checkpoints found under {checkpoints}")
            checkpoint = steps[-1] / "pretrained_model"

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _read_train_config(checkpoint: pathlib.Path) -> dict:
    for config_path in (
        checkpoint / "train_config.json",
        checkpoint / "config.json",
    ):
        if config_path.is_file():
            with config_path.open() as file:
                return json.load(file) or {}
    raise FileNotFoundError(f"No train_config.json or config.json found in {checkpoint}")


def _infer_eval_params(run_dir: pathlib.Path, cfg: dict) -> dict[str, str]:
    policy_cfg = cfg.get("policy") or cfg
    text = " ".join(
        str(value)
        for value in (
            run_dir.name,
            cfg.get("output_dir", ""),
            (cfg.get("dataset") or {}).get("repo_id", ""),
            cfg.get("job_name", ""),
        )
    )

    params = {
        "task": "HCIS-ShellGame-SingleArm-v0",
        "device": "cuda",
        "policy_backend": "lerobot",
        "policy_type": f"lerobot-{policy_cfg.get('type', 'diffusion')}",
        "policy_action_horizon": str(policy_cfg.get("n_action_steps", 1)),
        "num_episodes": str((cfg.get("eval") or {}).get("n_episodes", 50)),
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


def _expand_run_dir_args() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run_dir", type=pathlib.Path)
    parser.add_argument("--checkpoint_step", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    compact, passthrough = parser.parse_known_args(sys.argv[1:])

    if compact.run_dir is None:
        return

    run_dir = compact.run_dir.resolve()
    checkpoint = _resolve_checkpoint(run_dir, compact.checkpoint_step)
    cfg = _read_train_config(checkpoint)
    params = _infer_eval_params(run_dir, cfg)
    params["policy_checkpoint_path"] = str(checkpoint)

    expanded: list[str] = []
    for name, value in params.items():
        flag = f"--{name}"
        if not _has_arg(passthrough, flag):
            expanded.extend([flag, value])

    for flag in ("--enable_cameras", "--headless"):
        if not _has_arg(passthrough, flag):
            expanded.append(flag)

    sys.argv[1:] = expanded + passthrough
    print("[eval_diffusion] compact --run_dir expanded to:")
    print("[eval_diffusion] " + " ".join(sys.argv[1:]))
    if compact.dry_run:
        raise SystemExit(0)


if __name__ == "__main__":
    _expand_run_dir_args()
    runpy.run_path(str(_EVAL_SCRIPT), run_name="__main__")
