"""
ShellBench evaluation script.

Runs a trained policy through shell game episodes and computes:
  - DSR (Decision Success Rate): chose the correct cup
  - MSR (Manipulation Success Rate): successfully lifted any cup
  - SR  (Success Rate): chose correctly AND lifted
  - kappa (Cohen's Kappa): chance-corrected DSR

Usage:
    python scripts/eval_shell_game.py \
        --task HCIS-ShellGame-SingleArm-v0 \
        --device cuda --enable_cameras \
        --policy_type lerobot-diffusion \
        --policy_checkpoint_path ./checkpoints/shell_game \
        --num_episodes 50 --num_cups 3 --num_shuffles 2 \
        --output_json ./results/metrics.json
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ShellBench evaluation.")
parser.add_argument("--task", type=str, default="HCIS-ShellGame-SingleArm-v0")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--step_hz", type=int, default=60)
parser.add_argument("--episode_length_s", type=float, default=60.0)

# Policy
parser.add_argument("--policy_type", type=str, default="lerobot-diffusion")
parser.add_argument("--policy_action_horizon", type=int, default=16)
parser.add_argument("--policy_language_instruction", type=str, default=None)
parser.add_argument("--policy_checkpoint_path", type=str, required=True)

# Shell game
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--num_cups", type=int, default=3)
parser.add_argument("--num_shuffles", type=int, default=2)
parser.add_argument("--shuffle_speed", type=float, default=1.0)
parser.add_argument("--ball_position", type=str, default="random")

# Output
parser.add_argument("--output_json", type=str, default="./results/shell_game_metrics.json")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

import leisaac  # noqa: F401
import simulator.tasks  # noqa: F401
from simulator import FRANKA_JOINT_NAMES
from simulator.tasks.shell_game.shell_game_phase_manager import ShellGamePhaseManager
from simulator.datagen.state_machine.shell_game import ShellGameStateMachine


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ShellGameEvaluator:
    """Collects per-episode results and computes aggregate metrics."""

    def __init__(self, num_cups: int):
        self.num_cups = num_cups
        self.results: list[dict] = []

    def record_episode(self, episode_idx: int, phase_manager: ShellGamePhaseManager) -> dict:
        selected = phase_manager.selected_cup_index
        true_cup = phase_manager.ball_true_cup_index
        msr = selected is not None
        dsr = msr and (selected == true_cup)
        sr = dsr and msr

        result = {
            "episode": episode_idx,
            "selected_cup": selected,
            "ball_true": true_cup,
            "dsr": dsr,
            "msr": msr,
            "sr": sr,
        }
        self.results.append(result)
        return result

    def compute_metrics(self) -> dict:
        if not self.results:
            return {"DSR": 0.0, "MSR": 0.0, "SR": 0.0, "kappa": 0.0}

        dsr = sum(r["dsr"] for r in self.results) / len(self.results)
        msr = sum(r["msr"] for r in self.results) / len(self.results)
        sr = sum(r["sr"] for r in self.results) / len(self.results)
        chance = 1.0 / self.num_cups
        kappa = (dsr - chance) / (1.0 - chance) if (1.0 - chance) > 0 else 0.0

        return {
            "DSR": round(dsr, 4),
            "MSR": round(msr, 4),
            "SR": round(sr, 4),
            "kappa": round(kappa, 4),
        }


# ---------------------------------------------------------------------------
# Policy wrapper (same as rollout.py LeRobotSyncPolicy, simplified)
# ---------------------------------------------------------------------------

def _patch_lerobot_config(checkpoint_dir: str) -> None:
    from pathlib import Path
    _INCOMPAT = ("use_peft", "resize_shape", "crop_ratio", "compile_model", "compile_mode")
    cfg_path = Path(checkpoint_dir) / "config.json"
    if not cfg_path.is_file():
        return
    try:
        with cfg_path.open("r") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return
    stripped = [k for k in _INCOMPAT if k in cfg]
    if not stripped:
        return
    for k in stripped:
        cfg.pop(k, None)
    try:
        with cfg_path.open("w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[eval] stripped incompatible config fields: {stripped}")
    except OSError:
        pass


class EvalPolicy:
    """Minimal LeRobot sync policy for evaluation."""

    def __init__(self, policy_type: str, checkpoint: str, camera_infos: dict, device: str):
        self.device = device
        self.state_joint_names = FRANKA_JOINT_NAMES
        self.camera_keys = list(camera_infos.keys())

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_joint_names),),
                "names": [f"{n}.pos" for n in self.state_joint_names],
            }
        }
        for cam_key, shape in camera_infos.items():
            features[f"observation.images.{cam_key}"] = {
                "dtype": "image",
                "shape": (shape[0], shape[1], 3),
                "names": ["height", "width", "channels"],
            }
        self.lerobot_features = features

        _patch_lerobot_config(checkpoint)
        policy_class = get_policy_class(policy_type)
        self.policy = policy_class.from_pretrained(checkpoint, local_files_only=True)
        self.policy.to(device)
        self.policy.eval()

        device_override = {"device": device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

    def reset(self):
        if callable(getattr(self.policy, "reset", None)):
            self.policy.reset()

    def get_action(self, obs_dict: dict) -> torch.Tensor:
        raw = {
            key: obs_dict[key].cpu().numpy().astype(np.uint8)[0]
            for key in self.camera_keys
        }
        raw["task"] = obs_dict["task_description"]
        joint_pos = obs_dict["joint_pos"].cpu().numpy()
        for idx, name in enumerate(self.state_joint_names):
            raw[f"{name}.pos"] = joint_pos[0, idx].item()

        observation = raw_observation_to_observation(
            raw, self.lerobot_features, self.policy.config.image_features,
        )
        observation = self.preprocessor(observation)

        with torch.inference_mode():
            action = self.policy.select_action(observation)
        action = self.postprocessor(action)
        actions = action.to("cpu").numpy()
        return torch.from_numpy(actions[:, None, :])


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, hz):
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)
        self.last_time = time.time()

    def sleep(self, env):
        target = self.last_time + self.sleep_duration
        while time.time() < target:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    task_name = args_cli.task

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device("keyboard")
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.episode_length_s

    env_cfg.num_cups = args_cli.num_cups
    env_cfg.num_shuffles = args_cli.num_shuffles
    env_cfg.shuffle_speed = args_cli.shuffle_speed
    env_cfg.ball_position = args_cli.ball_position
    env_cfg.shell_game_seed = args_cli.seed

    # Keep time_out as safety net, disable any_cup_lifted (evaluator handles it)
    env_cfg.recorders = None

    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # Warm up renderer
    print("[eval] warming up renderer...")
    for _ in range(20):
        simulation_app.update()

    obs_dict, _ = env.reset()

    language = args_cli.policy_language_instruction
    if language is None:
        language = getattr(env_cfg, "task_description", "")

    # Detect cameras
    camera_infos = {}
    for key, sensor in env.scene.sensors.items():
        if isinstance(sensor, Camera) and key in obs_dict.get("policy", {}):
            camera_infos[key] = sensor.image_shape

    # Load policy
    policy_type = args_cli.policy_type.split("lerobot-", 1)[1]
    policy = EvalPolicy(policy_type, args_cli.policy_checkpoint_path, camera_infos, args_cli.device)

    # Setup phase manager and hold-action helper
    phase_manager = ShellGamePhaseManager.from_env_cfg(env_cfg)
    sm_helper = ShellGameStateMachine()
    sm_helper.setup(env)

    evaluator = ShellGameEvaluator(args_cli.num_cups)
    rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"[eval] Starting evaluation: {args_cli.num_episodes} episodes, "
          f"{args_cli.num_cups} cups, {args_cli.num_shuffles} shuffles")

    for ep in range(args_cli.num_episodes):
        obs_dict, _ = env.reset()
        policy.reset()
        sm_helper.reset()
        phase_manager.reset(env)

        episode_done = False
        while simulation_app.is_running() and not episode_done:
            with torch.inference_mode():
                is_act = phase_manager.step(env)

                if is_act:
                    # Policy controls
                    policy_obs = obs_dict["policy"].copy()
                    policy_obs["task_description"] = language
                    actions = policy.get_action(policy_obs).to(env.device)

                    for ai in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                        action = actions[ai, :, :]
                        obs_dict, _, terminated, timed_out, _ = env.step(action)
                        phase_manager.update_selection(env)

                        if terminated[0] or timed_out[0]:
                            episode_done = True
                            break
                        if rate_limiter:
                            rate_limiter.sleep(env)

                    if phase_manager.selected_cup_index is not None:
                        episode_done = True
                else:
                    # Phase 1-3: hold position, feed observations
                    hold_action = sm_helper.get_hold_action(env)
                    obs_dict, _, terminated, timed_out, _ = env.step(hold_action)

                    if terminated[0] or timed_out[0]:
                        episode_done = True

                    if rate_limiter:
                        rate_limiter.sleep(env)

        result = evaluator.record_episode(ep, phase_manager)
        status = "CORRECT" if result["dsr"] else ("LIFTED" if result["msr"] else "MISS")
        print(f"  Episode {ep + 1}/{args_cli.num_episodes}: {status} "
              f"(selected={result['selected_cup']}, truth={result['ball_true']})")

    # Compute and save metrics
    metrics = evaluator.compute_metrics()
    output = {
        "config": {
            "num_cups": args_cli.num_cups,
            "num_shuffles": args_cli.num_shuffles,
            "shuffle_speed": args_cli.shuffle_speed,
            "policy_checkpoint": args_cli.policy_checkpoint_path,
        },
        "num_episodes": args_cli.num_episodes,
        **metrics,
        "per_episode": evaluator.results,
    }

    output_dir = os.path.dirname(args_cli.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args_cli.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results ({args_cli.num_episodes} episodes):")
    print(f"  DSR   = {metrics['DSR']:.4f}  (chance = {1/args_cli.num_cups:.4f})")
    print(f"  MSR   = {metrics['MSR']:.4f}")
    print(f"  SR    = {metrics['SR']:.4f}")
    print(f"  kappa = {metrics['kappa']:.4f}")
    print(f"Saved to {args_cli.output_json}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
