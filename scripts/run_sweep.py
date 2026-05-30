"""
Experiment sweep runner for ShellBench.

Reads base.yaml + sweep.yaml, expands parameter combinations, and runs
optional data generation, training, and evaluation for each variant.

Training backends:
  - lerobot:    run lerobot-train using the repo's documented training flow
  - command:    run a user-provided command template
  - pretrained: skip training and evaluate an existing checkpoint
  - none:       skip training (useful for oracle evaluation)

Usage:
    python scripts/run_sweep.py \
        --base configs/base.yaml \
        --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
        --output_dir results/exp1_shuffle_scaling
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def set_nested(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_nested(d: dict, dotted_key: str, default: Any = None) -> Any:
    current: Any = d
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out


def safe_name(value: str) -> str:
    value = os.path.expandvars(str(value))
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "default"


def render_template(template: str, context: dict[str, Any]) -> str:
    rendered = os.path.expandvars(template)
    for key, value in sorted(context.items(), key=lambda item: len(item[0]), reverse=True):
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return render_template(value, context)
    return value


def local_dataset_exists(path: str) -> bool:
    dataset_path = Path(os.path.expanduser(path))
    if not dataset_path.exists():
        return False
    # LeRobot datasets usually contain a meta directory; HDF5 fallback is kept
    # for the StreamingRecorder path used by non-LeRobot recorders.
    return (dataset_path / "meta").exists() or dataset_path.is_file() or any(dataset_path.iterdir())


def hf_dataset_exists(repo_id: str) -> bool:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except Exception:
        print("  [datagen] huggingface_hub is unavailable; cannot check HF dataset reuse.")
        return False

    try:
        HfApi().repo_info(repo_id=repo_id, repo_type="dataset")
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as exc:
        print(f"  [datagen] HF dataset check failed for {repo_id}: {exc}")
        return False


def expand_sweep(base_cfg: dict, sweep_cfg: dict) -> list[tuple[str, dict]]:
    sweep_params = sweep_cfg.get("sweep", {})
    if not sweep_params:
        return [("default", copy.deepcopy(base_cfg))]

    keys = list(sweep_params.keys())
    value_lists = [sweep_params[key] for key in keys]

    variants: list[tuple[str, dict]] = []
    for combo in itertools.product(*value_lists):
        merged = copy.deepcopy(base_cfg)
        name_parts = []
        for key, value in zip(keys, combo):
            set_nested(merged, key, value)
            name_parts.append(f"{key.split('.')[-1]}={value}")
        variants.append((safe_name("_".join(name_parts)), merged))
    return variants


def build_context(cfg: dict, variant_name: str, variant_dir: str) -> dict[str, Any]:
    context = flatten_dict(cfg)
    context.update(
        {
            "variant_name": variant_name,
            "variant_dir": variant_dir,
            "dataset_file": os.path.join(variant_dir, "dataset.hdf5"),
            "checkpoint_dir": os.path.join(variant_dir, "checkpoints"),
            "metrics_json": os.path.join(variant_dir, "metrics.json"),
        }
    )

    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data_collection", {})
    policy_cfg = cfg.get("policy", {})

    repo_variant = safe_name(variant_name)
    dataset_repo_id = data_cfg.get("dataset_repo_id")
    if dataset_repo_id is None:
        prefix = training_cfg.get("dataset_repo_prefix", "${HF_USER}/shellbench")
        dataset_repo_id = f"{os.path.expandvars(prefix)}-{repo_variant}"

    policy_repo_id = policy_cfg.get("repo_id")
    if policy_repo_id is None:
        prefix = training_cfg.get("policy_repo_prefix", "${HF_USER}/shellbench-policy")
        policy_repo_id = f"{os.path.expandvars(prefix)}-{repo_variant}"

    local_dataset_dir = data_cfg.get("local_dataset_dir")
    if local_dataset_dir is None:
        local_dataset_dir = ".cache/lerobot/{dataset_repo_id}"

    context["dataset_repo_id"] = dataset_repo_id
    rendered_dataset_dir = os.path.expanduser(
        render_template(str(local_dataset_dir), {**context, "dataset_repo_id": dataset_repo_id})
    )
    if not os.path.isabs(rendered_dataset_dir):
        rendered_dataset_dir = os.path.abspath(rendered_dataset_dir)
    context["local_dataset_dir"] = rendered_dataset_dir
    context["policy_repo_id"] = policy_repo_id
    context["policy_checkpoint_path"] = policy_cfg.get(
        "checkpoint", os.path.join(context["checkpoint_dir"], "pretrained_model")
    )
    return context


def append_key_value_args(cmd: list[str], args: dict[str, Any]) -> None:
    for key, value in flatten_dict(args).items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        cmd.append(f"--{key}={value}")


def run_command(cmd: list[str] | str, *, shell: bool = False) -> None:
    printable = cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"  $ {printable}")
    subprocess.run(cmd, check=True, shell=shell)


def run_datagen(cfg: dict, context: dict[str, Any], extra_args: list[str] | None = None) -> str:
    task_cfg = cfg.get("task", {})
    phase_cfg = cfg.get("phase_duration", {})
    data_cfg = cfg.get("data_collection", {})
    runtime_cfg = cfg.get("runtime", {})

    python_exe = runtime_cfg.get("python", sys.executable)
    dataset_file = context["dataset_file"]
    local_reuse = bool(data_cfg.get("local_reuse_if_exists", True))
    hf_reuse = bool(data_cfg.get("HF_reuse_if_exists", True))

    # Prefer local reuse: this avoids both Isaac datagen and a fresh Hub download.
    # Training will add --dataset.root=<local_dataset_dir> when this path exists.
    if local_reuse and local_dataset_exists(context["local_dataset_dir"]):
        print(f"  [datagen] local dataset exists, reusing: {context['local_dataset_dir']}")
        return dataset_file

    # If the dataset is already on Hugging Face, skip expensive simulation and
    # let lerobot-train resolve/cache the repo through --dataset.repo_id.
    if hf_reuse and hf_dataset_exists(context["dataset_repo_id"]):
        print(f"  [datagen] HF dataset exists, skipping generation: {context['dataset_repo_id']}")
        return dataset_file

    cmd = [
        python_exe,
        "scripts/datagen/generate_shell_game.py",
        "--task",
        "HCIS-ShellGame-SingleArm-v0",
        "--num_envs",
        "1",
        "--device",
        runtime_cfg.get("sim_device", "cuda"),
        "--record",
        "--dataset_file",
        dataset_file,
        "--num_demos",
        str(data_cfg.get("num_demos", 100)),
        "--num_cups",
        str(task_cfg.get("num_cups", 3)),
        "--num_shuffles",
        str(task_cfg.get("num_shuffles", 2)),
        "--shuffle_speed",
        str(task_cfg.get("shuffle_speed", 1.0)),
        "--ball_position",
        str(task_cfg.get("ball_position", "random")),
        "--seed",
        str(cfg.get("evaluation", {}).get("seed", 42)),
        "--reveal_frames",
        str(phase_cfg.get("reveal", 50)),
        "--cover_frames",
        str(phase_cfg.get("cover", 10)),
        "--shuffle_per_swap_frames",
        str(phase_cfg.get("shuffle_per_swap", 30)),
        "--act_frames",
        str(phase_cfg.get("act", 150)),
    ]
    if runtime_cfg.get("enable_cameras", True):
        cmd.append("--enable_cameras")
    if data_cfg.get("use_lerobot_recorder", True):
        cmd.extend([
            "--use_lerobot_recorder",
            "--lerobot_dataset_repo_id",
            context["dataset_repo_id"],
            "--lerobot_dataset_fps",
            str(data_cfg.get("lerobot_dataset_fps", 30)),
        ])
    if extra_args:
        cmd.extend(extra_args)

    print("  [datagen]")
    run_command(cmd)
    return dataset_file


def run_training(cfg: dict, context: dict[str, Any]) -> str | None:
    training_cfg = cfg.get("training", {})
    policy_cfg = cfg.get("policy", {})
    policy_backend = policy_cfg.get("backend", "lerobot")
    backend = training_cfg.get("backend", "lerobot")

    if policy_backend == "oracle":
        return None
    if backend == "none":
        print("  [train] skipped: backend=none")
        return None
    if backend == "pretrained":
        checkpoint = render_value(policy_cfg.get("checkpoint"), context)
        if not checkpoint:
            raise ValueError("policy.checkpoint is required when backend=pretrained")
        print(f"  [train] skipped: using pretrained checkpoint {checkpoint}")
        return checkpoint

    checkpoint_dir = context["checkpoint_dir"]
    output_dir = training_cfg.get("output_dir", checkpoint_dir)
    output_dir = render_template(str(output_dir), context)

    if backend == "command":
        command_template = training_cfg.get("command")
        if not command_template:
            raise ValueError("training.command is required when training.backend=command")
        local_context = dict(context)
        local_context["checkpoint_dir"] = output_dir
        command = render_template(command_template, local_context)
        print("  [train:command]")
        run_command(command, shell=True)
        checkpoint_path = training_cfg.get(
            "checkpoint_path", os.path.join(output_dir, "pretrained_model")
        )
        return render_template(str(checkpoint_path), local_context)

    if backend != "lerobot":
        raise ValueError(f"Unsupported training backend: {backend}")

    cmd = [
        training_cfg.get("executable", "lerobot-train"),
        f"--dataset.repo_id={context['dataset_repo_id']}",
        f"--policy.type={policy_cfg.get('type', 'diffusion')}",
        f"--output_dir={output_dir}",
        f"--job_name={training_cfg.get('job_name_prefix', 'shellbench')}_{context['variant_name']}",
        f"--policy.device={training_cfg.get('device', 'cuda')}",
        f"--wandb.enable={str(training_cfg.get('wandb_enable', True)).lower()}",
        f"--policy.repo_id={context['policy_repo_id']}",
    ]
    # Keep dataset.repo_id for metadata, but point LeRobot at the local cache
    # when it exists so training does not fetch the dataset again.
    if local_dataset_exists(context["local_dataset_dir"]):
        cmd.append(f"--dataset.root={context['local_dataset_dir']}")

    train_args = copy.deepcopy(policy_cfg.get("train_args", {}))
    if "observation_horizon" in policy_cfg and get_nested(train_args, "policy.n_obs_steps") is None:
        set_nested(train_args, "policy.n_obs_steps", policy_cfg["observation_horizon"])
    append_key_value_args(cmd, train_args)
    cmd.extend(str(arg) for arg in training_cfg.get("extra_args", []))

    print("  [train:lerobot]")
    run_command(cmd)
    checkpoint_path = training_cfg.get(
        "checkpoint_path", os.path.join(output_dir, "pretrained_model")
    )
    return render_template(str(checkpoint_path), context)


def run_eval(
    cfg: dict,
    context: dict[str, Any],
    checkpoint_path: str | None,
    extra_args: list[str] | None = None,
) -> str:
    task_cfg = cfg.get("task", {})
    phase_cfg = cfg.get("phase_duration", {})
    eval_cfg = cfg.get("evaluation", {})
    policy_cfg = cfg.get("policy", {})
    runtime_cfg = cfg.get("runtime", {})

    policy_backend = policy_cfg.get("backend", "lerobot")
    output_json = context["metrics_json"]
    python_exe = runtime_cfg.get("python", sys.executable)

    cmd = [
        python_exe,
        "scripts/eval_shell_game.py",
        "--task",
        "HCIS-ShellGame-SingleArm-v0",
        "--device",
        runtime_cfg.get("sim_device", "cuda"),
        "--policy_backend",
        policy_backend,
        "--num_episodes",
        str(eval_cfg.get("num_episodes", 50)),
        "--num_cups",
        str(task_cfg.get("num_cups", 3)),
        "--num_shuffles",
        str(task_cfg.get("num_shuffles", 2)),
        "--shuffle_speed",
        str(task_cfg.get("shuffle_speed", 1.0)),
        "--ball_position",
        str(task_cfg.get("ball_position", "random")),
        "--seed",
        str(eval_cfg.get("seed", 42)),
        "--output_json",
        output_json,
        "--policy_action_horizon",
        str(policy_cfg.get("action_horizon", 16)),
        "--reveal_frames",
        str(phase_cfg.get("reveal", 50)),
        "--cover_frames",
        str(phase_cfg.get("cover", 10)),
        "--shuffle_per_swap_frames",
        str(phase_cfg.get("shuffle_per_swap", 30)),
        "--act_frames",
        str(phase_cfg.get("act", 150)),
    ]
    if runtime_cfg.get("enable_cameras", True):
        cmd.append("--enable_cameras")
    if policy_backend == "lerobot":
        checkpoint = checkpoint_path or policy_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError("A checkpoint path is required for policy.backend=lerobot")
        cmd.extend([
            "--policy_type",
            f"lerobot-{policy_cfg.get('type', 'diffusion')}",
            "--policy_checkpoint_path",
            render_template(str(checkpoint), context),
        ])
    cmd.extend(str(arg) for arg in policy_cfg.get("eval_args", []))
    if extra_args:
        cmd.extend(extra_args)

    print("  [eval]")
    run_command(cmd)
    return output_json


def main() -> None:
    parser = argparse.ArgumentParser(description="ShellBench sweep runner.")
    parser.add_argument("--base", type=str, required=True, help="Path to base.yaml")
    parser.add_argument("--sweep", type=str, required=True, help="Path to sweep.yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results.")
    parser.add_argument("--skip_datagen", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base)
    sweep_cfg = load_yaml(args.sweep)

    exp_name = sweep_cfg.get("experiment_name", "unnamed")
    output_dir = args.output_dir or os.path.join("results", exp_name)

    print(f"Experiment: {exp_name}")
    print(f"Description: {sweep_cfg.get('description', 'N/A')}")

    variants = expand_sweep(base_cfg, sweep_cfg)
    print(f"Expanded {len(variants)} variant(s).\n")

    summary = []
    for index, (variant_name, variant_cfg) in enumerate(variants):
        print(f"--- Variant {index + 1}/{len(variants)}: {variant_name} ---")
        variant_dir = os.path.join(output_dir, variant_name)
        os.makedirs(variant_dir, exist_ok=True)
        context = build_context(variant_cfg, variant_name, variant_dir)

        with open(os.path.join(variant_dir, "config.yaml"), "w") as f:
            yaml.dump(variant_cfg, f, default_flow_style=False)

        checkpoint_path: str | None = variant_cfg.get("policy", {}).get("checkpoint")
        if not args.skip_datagen:
            run_datagen(variant_cfg, context)
        if not args.skip_training:
            checkpoint_path = run_training(variant_cfg, context)
        if not args.skip_eval:
            metrics_file = run_eval(variant_cfg, context, checkpoint_path)
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    metrics = json.load(f)
                summary.append(
                    {"variant": variant_name, **{k: v for k, v in metrics.items() if k != "per_episode"}}
                )

    if summary:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'=' * 60}")
        print(f"Sweep complete. Summary saved to {summary_path}")
        print(f"{'=' * 60}")
        for row in summary:
            print(
                f"  {row['variant']}: DSR={row.get('DSR', 'N/A')}, "
                f"MSR={row.get('MSR', 'N/A')}, SR={row.get('SR', 'N/A')}, "
                f"kappa={row.get('kappa', 'N/A')}"
            )


if __name__ == "__main__":
    main()
