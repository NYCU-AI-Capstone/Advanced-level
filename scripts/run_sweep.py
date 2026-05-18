"""
Experiment sweep runner for ShellBench.

Reads base.yaml + sweep.yaml, expands all parameter combinations, and runs
data generation → training → evaluation for each variant.

Usage:
    python scripts/run_sweep.py \
        --base configs/base.yaml \
        --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
        --output_dir results/exp1_shuffle_scaling \
        [--skip_datagen] [--skip_training] [--skip_eval]
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_nested(d: dict, dotted_key: str, value):
    """Set a nested dict value using dotted key notation (e.g. 'task.num_cups')."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_nested(d: dict, dotted_key: str, default=None):
    keys = dotted_key.split(".")
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def expand_sweep(base_cfg: dict, sweep_cfg: dict) -> list[tuple[str, dict]]:
    """Expand sweep parameters into a list of (variant_name, merged_config) tuples."""
    sweep_params = sweep_cfg.get("sweep", {})
    if not sweep_params:
        return [("default", copy.deepcopy(base_cfg))]

    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    variants = []
    for combo in itertools.product(*value_lists):
        merged = copy.deepcopy(base_cfg)
        name_parts = []
        for key, val in zip(keys, combo):
            set_nested(merged, key, val)
            short_key = key.split(".")[-1]
            name_parts.append(f"{short_key}={val}")
        variant_name = "_".join(name_parts)
        variants.append((variant_name, merged))

    return variants


def run_datagen(cfg: dict, variant_dir: str, extra_args: list[str] | None = None):
    """Run data generation for a single variant."""
    task_cfg = cfg.get("task", {})
    data_cfg = cfg.get("data_collection", {})
    eval_cfg = cfg.get("evaluation", {})

    dataset_file = os.path.join(variant_dir, "dataset.hdf5")

    cmd = [
        sys.executable, "scripts/datagen/generate_shell_game.py",
        "--task", "HCIS-ShellGame-SingleArm-v0",
        "--num_envs", "1",
        "--device", "cuda",
        "--enable_cameras",
        "--record",
        "--dataset_file", dataset_file,
        "--num_demos", str(data_cfg.get("num_demos", 100)),
        "--num_cups", str(task_cfg.get("num_cups", 3)),
        "--num_shuffles", str(task_cfg.get("num_shuffles", 2)),
        "--shuffle_speed", str(task_cfg.get("shuffle_speed", 1.0)),
        "--ball_position", str(task_cfg.get("ball_position", "random")),
        "--seed", str(eval_cfg.get("seed", 42)),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  [datagen] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return dataset_file


def run_training(cfg: dict, dataset_file: str, variant_dir: str):
    """Run policy training. Returns checkpoint directory."""
    policy_cfg = cfg.get("policy", {})
    checkpoint_dir = os.path.join(variant_dir, "checkpoints")

    print(f"  [train] Policy type: {policy_cfg.get('type', 'diffusion')}")
    print(f"  [train] Checkpoint dir: {checkpoint_dir}")
    print(f"  [train] NOTE: Training command depends on your lerobot setup.")
    print(f"  [train] Example: lerobot-train --config <...> --dataset {dataset_file}")

    # Training is highly dependent on the specific lerobot configuration.
    # Users should customize this section based on their training pipeline.
    return checkpoint_dir


def run_eval(cfg: dict, checkpoint_dir: str, variant_dir: str, extra_args: list[str] | None = None):
    """Run evaluation for a single variant."""
    task_cfg = cfg.get("task", {})
    eval_cfg = cfg.get("evaluation", {})
    policy_cfg = cfg.get("policy", {})

    output_json = os.path.join(variant_dir, "metrics.json")

    cmd = [
        sys.executable, "scripts/eval_shell_game.py",
        "--task", "HCIS-ShellGame-SingleArm-v0",
        "--device", "cuda",
        "--enable_cameras",
        "--policy_type", f"lerobot-{policy_cfg.get('type', 'diffusion')}",
        "--policy_checkpoint_path", checkpoint_dir,
        "--num_episodes", str(eval_cfg.get("num_episodes", 50)),
        "--num_cups", str(task_cfg.get("num_cups", 3)),
        "--num_shuffles", str(task_cfg.get("num_shuffles", 2)),
        "--shuffle_speed", str(task_cfg.get("shuffle_speed", 1.0)),
        "--ball_position", str(task_cfg.get("ball_position", "random")),
        "--seed", str(eval_cfg.get("seed", 42)),
        "--output_json", output_json,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  [eval] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_json


def main():
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
    for i, (variant_name, variant_cfg) in enumerate(variants):
        print(f"--- Variant {i+1}/{len(variants)}: {variant_name} ---")
        variant_dir = os.path.join(output_dir, variant_name)
        os.makedirs(variant_dir, exist_ok=True)

        # Save variant config
        with open(os.path.join(variant_dir, "config.yaml"), "w") as f:
            yaml.dump(variant_cfg, f, default_flow_style=False)

        dataset_file = os.path.join(variant_dir, "dataset.hdf5")
        checkpoint_dir = os.path.join(variant_dir, "checkpoints")

        if not args.skip_datagen:
            dataset_file = run_datagen(variant_cfg, variant_dir)

        if not args.skip_training:
            checkpoint_dir = run_training(variant_cfg, dataset_file, variant_dir)

        if not args.skip_eval:
            metrics_file = run_eval(variant_cfg, checkpoint_dir, variant_dir)
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    metrics = json.load(f)
                summary.append({"variant": variant_name, **{k: v for k, v in metrics.items() if k != "per_episode"}})

    # Save summary
    if summary:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Sweep complete. Summary saved to {summary_path}")
        print(f"{'='*60}")
        for s in summary:
            print(f"  {s['variant']}: DSR={s.get('DSR', 'N/A')}, "
                  f"MSR={s.get('MSR', 'N/A')}, SR={s.get('SR', 'N/A')}, "
                  f"kappa={s.get('kappa', 'N/A')}")


if __name__ == "__main__":
    main()
