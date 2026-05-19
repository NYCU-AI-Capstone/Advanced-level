"""Environment configuration for the ShellBench shell game task."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from simulator import ASSETS_ROOT
from simulator.assets.scenes.kitchen import KITCHEN_CFG, KITCHEN_USD_PATH

from simulator.tasks.template import mdp
from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

KITCHEN_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "kitchen" / "objects"

CUP_USD_PATH = str(KITCHEN_OBJECTS_ROOT / "PinkCup" / "PinkCup.usd")

MAX_CUPS = 5

# Cup arrangement: cups placed along x-axis on the table
CUP_BASE_Y = -0.20
CUP_Z = 0.14
CUP_SPACING = 0.12

# Ball geometry
BALL_RADIUS = 0.008
# Ball center inside the cup — cup origin is near the rim, so offset downward.
BALL_Z = 0.05067

# Position to hide unused cups / ball (far below scene)
HIDDEN_POS = (0.0, 0.0, -10.0)


def _cup_x_positions(num_cups: int) -> list[float]:
    """Compute x positions for cups centered around x=0.5."""
    center_x = 0.50
    total_width = (num_cups - 1) * CUP_SPACING
    start_x = center_x - total_width / 2.0
    return [start_x + i * CUP_SPACING for i in range(num_cups)]


def _make_cup_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Scene/cup_{index}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUP_USD_PATH,
            mass_props=MassPropertiesCfg(mass=0.001),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=HIDDEN_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def any_cup_z_above_threshold(
    env,
    cup_cfg_0: SceneEntityCfg,
    cup_cfg_1: SceneEntityCfg,
    cup_cfg_2: SceneEntityCfg,
    cup_cfg_3: SceneEntityCfg,
    cup_cfg_4: SceneEntityCfg,
    lift_threshold: float,
    num_active_cups: int,
) -> torch.Tensor:
    """Termination: any active cup's z exceeds its initial z + lift_threshold."""
    cup_cfgs = [cup_cfg_0, cup_cfg_1, cup_cfg_2, cup_cfg_3, cup_cfg_4]
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for i in range(num_active_cups):
        cup: RigidObject = env.scene[cup_cfgs[i].name]
        cup_pos = cup.data.root_pos_w - env.scene.env_origins
        lifted = cup_pos[:, 2] > (CUP_Z + lift_threshold)
        done = torch.logical_or(done, lifted)

    return done


@configclass
class ShellGameSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene with N identical cups and 1 ball."""

    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    cup_0: RigidObjectCfg = _make_cup_cfg(0)
    cup_1: RigidObjectCfg = _make_cup_cfg(1)
    cup_2: RigidObjectCfg = _make_cup_cfg(2)
    cup_3: RigidObjectCfg = _make_cup_cfg(3)
    cup_4: RigidObjectCfg = _make_cup_cfg(4)

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),
            ),
            mass_props=MassPropertiesCfg(mass=0.0001),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=HIDDEN_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    """Termination: time_out + any_cup_lifted (no success term)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    any_cup_lifted = DoneTerm(
        func=any_cup_z_above_threshold,
        params={
            "cup_cfg_0": SceneEntityCfg("cup_0"),
            "cup_cfg_1": SceneEntityCfg("cup_1"),
            "cup_cfg_2": SceneEntityCfg("cup_2"),
            "cup_cfg_3": SceneEntityCfg("cup_3"),
            "cup_cfg_4": SceneEntityCfg("cup_4"),
            "lift_threshold": 0.05,
            "num_active_cups": 3,
        },
    )


@configclass
class ShellGameEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the ShellBench environment."""

    scene: ShellGameSceneCfg = ShellGameSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "observe the shell game shuffle and lift the cup hiding the ball."

    # Shell game parameters
    num_cups: int = 3
    num_shuffles: int = 2
    shuffle_speed: float = 1.0
    ball_position: str = "random"
    shell_game_seed: int = 42

    # Phase durations (in steps)
    reveal_frames: int = 50
    cover_frames: int = 10
    shuffle_per_swap_frames: int = 30
    act_frames: int = 150

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -math.pi / 4.0,
            "panda_joint3": 0.0,
            "panda_joint4": -3.0 * math.pi / 4.0,
            "panda_joint5": 0.0,
            "panda_joint6": math.pi / 2.0,
            "panda_joint7": math.pi / 4.0,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }

        parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)

        # ShellBench does not use UMI object_poses
        self.object_pose_cfg = None

        # Update termination with actual num_cups
        self.terminations.any_cup_lifted.params["num_active_cups"] = self.num_cups

        # Set episode length to accommodate all phases
        total_steps = (
            self.reveal_frames
            + self.cover_frames
            + self.num_shuffles * self.shuffle_per_swap_frames
            + self.act_frames
        )
        self.episode_length_s = total_steps / 30.0 + 5.0
