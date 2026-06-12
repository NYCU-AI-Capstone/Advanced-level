#!/usr/bin/env python
"""訓練 wrapper —— 註冊 LSTM policy，支援 YAML config，呼叫 lerobot-train。

用法：
    # 用 YAML config
    python policies/lstm/scripts/train_lstm.py --config policies/lstm/configs/default.yaml

    # YAML + CLI override（CLI 優先）
    python policies/lstm/scripts/train_lstm.py --config policies/lstm/configs/default.yaml --steps=50000

    # 純 CLI（跟以前一樣）
    python policies/lstm/scripts/train_lstm.py --policy.type=lstm --dataset.repo_id=... --steps=5000

每次 checkpoint 儲存後預設會跑 10 集 ShellGame evaluation，並以 DSR 選出
checkpoints/best。可用 --checkpoint_eval_episodes=0 關閉，或指定其他集數。
訓練結束後會在 output_dir 寫出 train_config.yaml，記錄實際使用的完整參數。
"""

import gc
import json
import os
import pathlib
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import policies.lstm.scripts.setup_cache  # noqa: F401,E402  必須在 lerobot/torch 之前
import policies.lstm.policy.register  # noqa: F401,E402


def _translate_config_arg() -> None:
    """Translate our --config into draccus's --config_path."""
    new_args = []
    i = 0
    args = sys.argv[1:]
    while i < len(args):
        arg = args[i]
        if arg == "--config" and i + 1 < len(args):
            new_args.append(f"--config_path={args[i + 1]}")
            i += 2
        elif arg.startswith("--config="):
            new_args.append("--config_path=" + arg.split("=", 1)[1])
            i += 1
        else:
            new_args.append(arg)
            i += 1
    sys.argv[1:] = new_args



def _pop_checkpoint_eval_args() -> int:
    """Remove wrapper-only eval args before LeRobot parses the CLI."""
    episodes = 10
    kept = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--checkpoint_eval_episodes="):
            episodes = int(arg.split("=", 1)[1])
            i += 1
        elif arg == "--checkpoint_eval_episodes" and i + 1 < len(args):
            episodes = int(args[i + 1])
            i += 2
        else:
            kept.append(arg)
            i += 1
    if episodes < 0:
        raise ValueError("--checkpoint_eval_episodes must be >= 0")
    sys.argv[1:] = kept
    return episodes


def _install_checkpoint_eval(output_dir: pathlib.Path | None, episodes: int):
    """Evaluate every saved checkpoint and point checkpoints/best at the best DSR."""
    if episodes == 0 or output_dir is None:
        return None

    import torch
    import lerobot.scripts.lerobot_train as train_mod

    original_update_last = train_mod.update_last_checkpoint
    output_dir = output_dir.resolve()
    eval_dir = output_dir / "eval"
    best_summary_path = eval_dir / "best_metrics.json"

    def load_best_dsr() -> float:
        if not best_summary_path.is_file():
            return float("-inf")
        try:
            return float(json.loads(best_summary_path.read_text())["DSR"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return float("-inf")

    def update_best(checkpoint_dir: pathlib.Path, metrics: dict, metrics_path: pathlib.Path) -> None:
        best_link = output_dir / "checkpoints" / "best"
        if best_link.exists() and not best_link.is_symlink():
            print(f"[train_lstm] Cannot update best checkpoint: {best_link} is not a symlink")
            return

        temp_link = best_link.with_name(".best.tmp")
        temp_link.unlink(missing_ok=True)
        temp_link.symlink_to(checkpoint_dir.name)
        os.replace(temp_link, best_link)

        summary = {
            "checkpoint": str(checkpoint_dir),
            "step": int(checkpoint_dir.name),
            "metrics_file": str(metrics_path),
            "DSR": metrics["DSR"],
            "MSR": metrics["MSR"],
            "SR": metrics["SR"],
            "kappa": metrics["kappa"],
        }
        best_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"[train_lstm] New best checkpoint: step {summary['step']} "
            f"DSR={summary['DSR']:.4f} MSR={summary['MSR']:.4f}"
        )

    def patched_update_last(checkpoint_dir):
        original_update_last(checkpoint_dir)
        checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
        step = int(checkpoint_dir.name)
        eval_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = eval_dir / f"checkpoint_{step:06d}.json"

        print(f"[train_lstm] Evaluating checkpoint {step} for {episodes} episodes...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        command = [
            sys.executable,
            str(REPO_ROOT / "policies" / "lstm" / "scripts" / "eval_lstm.py"),
            "--run_dir", str(output_dir),
            "--checkpoint_step", str(step),
            "--num_episodes", str(episodes),
            "--output_json", str(metrics_path),
        ]
        eval_env = os.environ.copy()
        eval_env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        completed = None
        last_error = None
        for attempt in range(1, 4):
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=eval_env,
                )
                metrics = json.loads(metrics_path.read_text())
                break
            except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 3:
                    delay = attempt * 5
                    print(
                        f"[train_lstm] Evaluation attempt {attempt}/3 failed; "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
        else:
            print(
                f"[train_lstm] Checkpoint evaluation failed at step {step} "
                f"after 3 attempts: {last_error}"
            )
            captured_result = completed
            if captured_result is None and isinstance(last_error, subprocess.CalledProcessError):
                captured_result = last_error
            if captured_result is not None:
                captured = "\n".join(
                    part.strip()
                    for part in (captured_result.stdout, captured_result.stderr)
                    if part and part.strip()
                )
                if captured:
                    print("[train_lstm] Last evaluation output:")
                    print("\n".join(captured.splitlines()[-40:]))
            return

        results = metrics.get("per_episode", [])
        correct = sum(bool(result.get("dsr")) for result in results)
        lifted = sum(
            bool(result.get("msr")) and not bool(result.get("dsr"))
            for result in results
        )
        miss = len(results) - correct - lifted
        print(
            f"[train_lstm] Checkpoint {step}: "
            f"CORRECT={correct} LIFTED={lifted} MISS={miss} "
            f"| DSR={metrics['DSR']:.1%} MSR={metrics['MSR']:.1%}"
        )
        if float(metrics["DSR"]) > load_best_dsr():
            update_best(checkpoint_dir, metrics, metrics_path)

    train_mod.update_last_checkpoint = patched_update_last

    def close():
        train_mod.update_last_checkpoint = original_update_last

    return close


def _resolve_steps() -> int | None:
    """Best-effort: read training steps from CLI or the YAML config file."""
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--steps="):
            return int(arg.split("=", 1)[1])
        if arg == "--steps" and i + 2 < len(sys.argv):
            return int(sys.argv[i + 2])

    for i, arg in enumerate(sys.argv[1:]):
        cfg_path = None
        if arg.startswith("--config_path="):
            cfg_path = arg.split("=", 1)[1]
        elif arg == "--config_path" and i + 2 < len(sys.argv):
            cfg_path = sys.argv[i + 2]
        if cfg_path:
            import yaml
            with open(cfg_path) as f:
                d = yaml.safe_load(f)
            if d and "steps" in d:
                return int(d["steps"])
    return None


def _install_tqdm_progress(total_steps: int | None):
    """Attach a tqdm progress bar to LeRobot's per-step policy update.

    LeRobot owns the actual training loop, so this wrapper observes update_policy()
    instead of copying trainer code. Set LSTM_TQDM=0 to disable it.
    """
    if os.environ.get("LSTM_TQDM", "1").lower() in {"0", "false", "no", "off"}:
        return None

    import lerobot.scripts.lerobot_train as train_mod
    from tqdm.auto import tqdm

    original_update_policy = train_mod.update_policy
    state = {"bar": None}

    def patched_update_policy(train_metrics, *args, **kwargs):
        is_main_process = not train_metrics.accelerator or train_metrics.accelerator.is_main_process
        if is_main_process and state["bar"] is None:
            state["bar"] = tqdm(
                total=total_steps,
                initial=train_metrics.steps,
                unit="step",
                dynamic_ncols=True,
                desc="LSTM train",
            )

        train_metrics, output_dict = original_update_policy(train_metrics, *args, **kwargs)

        bar = state["bar"]
        if is_main_process and bar is not None:
            bar.update(1)
            postfix = {
                "loss": f"{train_metrics.loss.val:.4f}",
                "lr": f"{train_metrics.lr.val:.1e}",
                "updt_s": f"{train_metrics.update_s.val:.3f}",
                "data_s": f"{train_metrics.dataloading_s.val:.3f}",
                "epoch": f"{train_metrics.epochs:.2f}",
            }
            bar.set_postfix(postfix)

        return train_metrics, output_dict

    train_mod.update_policy = patched_update_policy

    def close():
        train_mod.update_policy = original_update_policy
        if state["bar"] is not None:
            state["bar"].close()

    return close


def _resolve_output_dir() -> pathlib.Path | None:
    """Best-effort: read output_dir from CLI or the YAML config file."""
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--output_dir="):
            return pathlib.Path(arg.split("=", 1)[1])
        if arg == "--output_dir" and i + 2 < len(sys.argv):
            return pathlib.Path(sys.argv[i + 2])

    for i, arg in enumerate(sys.argv[1:]):
        cfg_path = None
        if arg.startswith("--config_path="):
            cfg_path = arg.split("=", 1)[1]
        elif arg == "--config_path" and i + 2 < len(sys.argv):
            cfg_path = sys.argv[i + 2]
        if cfg_path:
            import yaml
            with open(cfg_path) as f:
                d = yaml.safe_load(f)
            if d and "output_dir" in d:
                return pathlib.Path(d["output_dir"])
    return None


def _dump_resolved_config(output_dir: pathlib.Path) -> None:
    """After training, read the lerobot-generated config.json and also dump a
    human-readable train_config.yaml next to it."""
    import json
    import yaml

    # lerobot saves train_config.json + config.json inside each checkpoint's
    # pretrained_model/ dir. "last" is a symlink to the latest step.
    sources = [
        output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json",
        output_dir / "checkpoints" / "last" / "pretrained_model" / "config.json",
    ]
    for src in sources:
        if src.is_file():
            with open(src) as f:
                cfg_dict = json.load(f)
            dst = output_dir / "train_config.yaml"
            with open(dst, "w") as f:
                yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"[train_lstm] Resolved config saved to {dst}")
            return

    # Fallback: reconstruct from the YAML input + CLI args
    for i, arg in enumerate(sys.argv[1:]):
        cfg_path = None
        if arg.startswith("--config_path="):
            cfg_path = arg.split("=", 1)[1]
        elif arg == "--config_path" and i + 2 < len(sys.argv):
            cfg_path = sys.argv[i + 2]
        if cfg_path and pathlib.Path(cfg_path).is_file():
            import shutil
            dst = output_dir / "train_config.yaml"
            shutil.copy2(cfg_path, dst)
            print(f"[train_lstm] Input config copied to {dst}")
            return


if __name__ == "__main__":
    _translate_config_arg()
    checkpoint_eval_episodes = _pop_checkpoint_eval_args()
    output_dir = _resolve_output_dir()
    total_steps = _resolve_steps()

    from lerobot.scripts.lerobot_train import main

    close_tqdm = _install_tqdm_progress(total_steps)
    close_checkpoint_eval = _install_checkpoint_eval(output_dir, checkpoint_eval_episodes)
    try:
        main()
    finally:
        if close_checkpoint_eval is not None:
            close_checkpoint_eval()
        if close_tqdm is not None:
            close_tqdm()

    if output_dir and output_dir.is_dir():
        _dump_resolved_config(output_dir)
