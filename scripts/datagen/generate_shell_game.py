"""
Data generation script for the ShellBench shell game task.

Unlike generate.py which requires --object_poses (UMI pipeline), this script
generates episodes parametrically: ball position and shuffle sequences are
randomised per episode based on --seed.

Usage:
    python scripts/datagen/generate_shell_game.py \
        --task HCIS-ShellGame-SingleArm-v0 \
        --num_envs 1 --device cuda --enable_cameras \
        --record --dataset_file ./datasets/shell_game.hdf5 \
        --num_demos 100 --num_cups 3 --num_shuffles 2 --seed 42
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from project_cache import configure_project_cache


configure_project_cache()
import signal
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ShellBench data generation.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="HCIS-ShellGame-SingleArm-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--record", action="store_true")
parser.add_argument("--step_hz", type=int, default=60)
parser.add_argument("--dataset_file", type=str, default="./datasets/shell_game.hdf5")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--quality", action="store_true")
parser.add_argument("--use_lerobot_recorder", action="store_true")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None)
parser.add_argument("--lerobot_dataset_fps", type=int, default=30)

# Shell game specific
parser.add_argument("--num_demos", type=int, default=100, help="Number of demo episodes to generate.")
parser.add_argument("--num_cups", type=int, default=3)
parser.add_argument("--num_shuffles", type=int, default=2)
parser.add_argument("--shuffle_speed", type=float, default=1.0)
parser.add_argument("--ball_position", type=str, default="random")
parser.add_argument("--reveal_frames", type=int, default=50)
parser.add_argument("--cover_frames", type=int, default=10)
parser.add_argument("--shuffle_per_swap_frames", type=int, default=30)
parser.add_argument("--act_frames", type=int, default=150)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import simulator.tasks  # noqa: F401
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

from simulator.datagen.state_machine.shell_game import ShellGameStateMachine
from simulator.tasks.shell_game.shell_game_phase_manager import ShellGamePhaseManager


class RateLimiter:
    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()


def main():
    task_name = args_cli.task
    if args_cli.num_envs != 1:
        raise ValueError("ShellBench v1 only supports --num_envs 1; PhaseManager state is single-env.")
    if args_cli.num_cups < 3 or args_cli.num_cups > 5:
        raise ValueError("ShellBench supports --num_cups in [3, 5].")

    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configure environment
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device("keyboard")
    env_cfg.seed = args_cli.seed

    # Apply shell game parameters
    env_cfg.num_cups = args_cli.num_cups
    env_cfg.num_shuffles = args_cli.num_shuffles
    env_cfg.shuffle_speed = args_cli.shuffle_speed
    env_cfg.ball_position = args_cli.ball_position
    env_cfg.shell_game_seed = args_cli.seed
    env_cfg.reveal_frames = args_cli.reveal_frames
    env_cfg.cover_frames = args_cli.cover_frames
    env_cfg.shuffle_per_swap_frames = args_cli.shuffle_per_swap_frames
    env_cfg.act_frames = args_cli.act_frames

    # Disable time_out during data gen (FSM controls episode end)
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "any_cup_lifted"):
        env_cfg.terminations.any_cup_lifted = None

    # Configure recorder
    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(
            func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
    else:
        env_cfg.recorders = None

    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # Disable gravity for robot
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics
    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
            from leisaac.enhance.managers.lerobot_recorder_manager import LeRobotRecorderManager
            dataset_cfg = LeRobotDatasetCfg(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
            )
            del env.recorder_manager
            env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
        else:
            del env.recorder_manager
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"

    rate_limiter = RateLimiter(args_cli.step_hz)

    # Setup state machine and phase manager
    sm = ShellGameStateMachine()
    sm.setup(env)

    phase_manager = ShellGamePhaseManager.from_env_cfg(env_cfg)
    sm.set_phase_manager(phase_manager)

    env.reset()
    sm.reset()
    phase_manager.reset(env)

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resuming from {resume_recorded_demo_count} existing demos.")
    current_recorded_demo_count = resume_recorded_demo_count

    episode_idx = 0
    start_record_state = False
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Ctrl+C detected. Cleaning up...")

    original_handler = signal.signal(signal.SIGINT, signal_handler)
    success_ids = []

    try:
        while simulation_app.is_running() and not interrupted and episode_idx < args_cli.num_demos:
            with torch.inference_mode():
                if sm.is_episode_done:
                    # Episode finished — check success and record
                    success = sm.check_success(env)
                    print(f"Episode {episode_idx + 1}/{args_cli.num_demos}: "
                          f"{'SUCCESS' if success else 'FAIL'} "
                          f"(selected={phase_manager.selected_cup_index}, "
                          f"truth={phase_manager.ball_true_cup_index})")

                    if start_record_state:
                        start_record_state = False

                    if args_cli.record and success:
                        auto_terminate(env, True)
                        current_recorded_demo_count += 1
                        success_ids.append(episode_idx + 1)
                    else:
                        auto_terminate(env, False)

                    episode_idx += 1
                    if episode_idx >= args_cli.num_demos:
                        break

                    # Reset for next episode
                    env.reset()
                    sm.reset()
                    phase_manager.reset(env)
                    auto_terminate(env, False)

                else:
                    if not start_record_state:
                        if args_cli.record:
                            print(f"Recording episode {episode_idx + 1}...")
                        start_record_state = True

                    is_act = phase_manager.step(env)

                    if is_act:
                        # Phase 4: let FSM generate action
                        sm.pre_step(env)
                        actions = sm.get_action(env)
                        env.step(actions)
                        phase_manager.update_selection(env)
                        sm.advance()
                    else:
                        # Phase 1-3: hold position, env records observations
                        hold_action = sm.get_hold_action(env)
                        env.step(hold_action)

                if rate_limiter:
                    rate_limiter.sleep(env)

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        env.close()
        simulation_app.close()

    print(f"\nDone. {len(success_ids)}/{args_cli.num_demos} successful demos.")
    print(f"Success IDs: {success_ids}")


if __name__ == "__main__":
    main()
