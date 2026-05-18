"""State machine for the ShellBench shell game task (data generation).

This FSM knows ground truth (ball_true_cup_index) and always picks the
correct cup. Used to generate expert demonstrations.

Phase 4 (Act) FSM states:
  0: MOVE_ABOVE   - Move end-effector above the correct cup
  1: DESCEND      - Lower to grasp position
  2: GRASP        - Close gripper
  3: LIFT         - Lift cup to expose ball
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from leisaac.datagen.state_machine.base import StateMachineBase

if TYPE_CHECKING:
    from simulator.tasks.shell_game.shell_game_phase_manager import ShellGamePhaseManager

_EE_BODY_NAME = "panda_hand"
_FRANKA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0

_MAX_CARTESIAN_DELTA = 0.018
_MAX_ROT_DELTA = 0.08
_IK_DLS_LAMBDA = 0.01

_HOVER_Z_OFFSET = 0.15
_GRASP_Z_OFFSET = 0.08
_LIFT_Z_OFFSET = 0.20
_GRIPPER_DOWN_ROLL_W = math.pi
_GRIPPER_DOWN_PITCH_W = 0.0

_FRANKA_REST_JOINT_POS = {
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


def _constant_gripper(num_envs: int, device: torch.device, value: float) -> torch.Tensor:
    return torch.full((num_envs, 1), value, device=device)


def _clamp_delta(delta: torch.Tensor, max_norm: float = _MAX_CARTESIAN_DELTA) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return delta * scale


def _shortest_quat(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[:, 0:1] < 0.0, -quat, quat)


def _find_body_index(robot, body_name: str) -> int:
    if hasattr(robot, "find_bodies"):
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) > 0:
            return int(body_ids[0])
    body_names = getattr(robot.data, "body_names", None)
    if body_names is not None and body_name in body_names:
        return body_names.index(body_name)
    return -1


class ShellGameStateMachine(StateMachineBase):
    """Scripted Franka policy for lifting the correct cup in a shell game.

    Requires a ShellGamePhaseManager to be set via set_phase_manager()
    before generating actions. The phase manager provides ground truth
    about which cup hides the ball.
    """

    MAX_STEPS: int = 720

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._ee_body_idx: int = -1
        self._jacobi_body_idx: int = -1
        self._arm_joint_ids: list[int] = []
        self._jacobi_joint_ids: list[int] = []
        self._rest_joint_pos: torch.Tensor | None = None
        self._rest_ee_pos_w: torch.Tensor | None = None
        self._initial_ee_pos_w: torch.Tensor | None = None
        self._gripper_down_yaw_w: torch.Tensor | None = None
        self._phase_manager: ShellGamePhaseManager | None = None
        self._event: int = 0
        self._events_dt = [
            160,  # Phase 0: Move above the correct cup
            80,   # Phase 1: Descend to grasp
            20,   # Phase 2: Close gripper
            100,  # Phase 3: Lift cup
        ]

    def set_phase_manager(self, phase_manager: ShellGamePhaseManager) -> None:
        self._phase_manager = phase_manager

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        robot = env.scene["robot"]
        self._ee_body_idx = _find_body_index(robot, _EE_BODY_NAME)
        joint_names = list(robot.data.joint_names)
        self._arm_joint_ids = [joint_names.index(n) for n in _FRANKA_ARM_JOINT_NAMES]

        if self._ee_body_idx < 0:
            raise ValueError(f"Could not find body '{_EE_BODY_NAME}'.")
        if robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [jid + 6 for jid in self._arm_joint_ids]

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _FRANKA_REST_JOINT_POS:
                self._rest_joint_pos[:, idx] = _FRANKA_REST_JOINT_POS[name]

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._rest_ee_pos_w = self._ee_pos_w(robot).clone()

    def check_success(self, env) -> bool:
        if self._phase_manager is None:
            return False
        selected = self._phase_manager.selected_cup_index
        if selected is None:
            return False
        return selected == self._phase_manager.ball_true_cup_index

    def pre_step(self, env) -> None:
        pass

    def get_action(self, env) -> torch.Tensor:
        """Generate action to lift the correct cup (ground truth)."""
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        if self._phase_manager is None:
            raise RuntimeError("Phase manager not set. Call set_phase_manager() first.")

        # Get the correct cup's world position
        cup_idx = self._phase_manager.ball_true_cup_index
        cup = env.scene[f"cup_{cup_idx}"]
        target_cup_pos_w = cup.data.root_pos_w.clone()

        device = env.device
        num_envs = env.num_envs

        if self._step_count == 0 and self._event == 0:
            self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        target_quat_w = self._gripper_down_quat_w(robot, num_envs, device, robot.data.root_quat_w.dtype)

        if self._event == 0:
            target_pos_w, gripper_cmd = self._phase_move_above(target_cup_pos_w, num_envs, device)
        elif self._event == 1:
            target_pos_w, gripper_cmd = self._phase_descend(target_cup_pos_w, num_envs, device)
        elif self._event == 2:
            target_pos_w, gripper_cmd = self._phase_grasp(target_cup_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_lift(target_cup_pos_w, num_envs, device)

        return self._joint_position_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)

    def get_hold_action(self, env) -> torch.Tensor:
        """Return action that holds the robot in its current joint position."""
        robot = env.scene["robot"]
        current_arm_pos = robot.data.joint_pos[:, self._arm_joint_ids]
        gripper_cmd = _constant_gripper(env.num_envs, env.device, _GRIPPER_OPEN)
        return torch.cat([current_arm_pos, gripper_cmd], dim=-1)

    def advance(self) -> None:
        if self._episode_done:
            return
        self._step_count += 1
        if self._step_count < self._events_dt[self._event]:
            return
        self._event += 1
        self._step_count = 0
        if self._event >= len(self._events_dt):
            self._episode_done = True

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._event = 0
        self._initial_ee_pos_w = None
        self._gripper_down_yaw_w = None

    # ------------------------------------------------------------------
    # Act-phase sub-phases
    # ------------------------------------------------------------------

    def _phase_move_above(self, cup_pos_w, num_envs, device):
        target_pos_w = cup_pos_w.clone()
        target_pos_w[:, 2] += _HOVER_Z_OFFSET
        if self._initial_ee_pos_w is not None:
            denom = max(self._events_dt[0] - 1, 1)
            alpha = min(self._step_count / denom, 1.0)
            target_pos_w = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target_pos_w
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_descend(self, cup_pos_w, num_envs, device):
        target_pos_w = cup_pos_w.clone()
        target_pos_w[:, 2] += _GRASP_Z_OFFSET
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_grasp(self, cup_pos_w, num_envs, device):
        target_pos_w = cup_pos_w.clone()
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lift(self, cup_pos_w, num_envs, device):
        target_pos_w = cup_pos_w.clone()
        target_pos_w[:, 2] += _LIFT_Z_OFFSET
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    # ------------------------------------------------------------------
    # IK helpers (same pattern as CupStackingStateMachine)
    # ------------------------------------------------------------------

    def _ee_pos_w(self, robot) -> torch.Tensor:
        return robot.data.body_pos_w[:, self._ee_body_idx, :]

    def _ee_quat_w(self, robot) -> torch.Tensor:
        return robot.data.body_quat_w[:, self._ee_body_idx, :]

    def _joint_position_franka_action(self, env, target_pos_w, target_quat_w, gripper_cmd):
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        root_quat_inv = quat_inv(root_quat_w)

        target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
        ee_pos_root = quat_apply(root_quat_inv, self._ee_pos_w(robot) - root_pos_w)
        delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)

        delta_quat_w = _shortest_quat(quat_mul(target_quat_w, quat_inv(self._ee_quat_w(robot))))
        delta_rot_w = axis_angle_from_quat(delta_quat_w)
        delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)

        pose_delta_root = torch.cat([delta_pos_root, delta_rot_root], dim=-1)
        joint_pos_target = self._arm_joint_pos(robot) + self._compute_delta_joint_pos(
            pose_delta_root, self._ee_jacobian_root(robot)
        )
        joint_pos_target = self._clamp_arm_joint_pos(robot, joint_pos_target)
        return torch.cat([joint_pos_target, gripper_cmd], dim=-1)

    def _arm_joint_pos(self, robot) -> torch.Tensor:
        return robot.data.joint_pos[:, self._arm_joint_ids]

    def _ee_jacobian_root(self, robot) -> torch.Tensor:
        jacobian = robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ].clone()
        root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(root_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_delta_joint_pos(self, pose_delta, jacobian):
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (_IK_DLS_LAMBDA**2) * torch.eye(
            jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
        )
        delta_joint_pos = (
            jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix) @ pose_delta.unsqueeze(-1)
        )
        return delta_joint_pos.squeeze(-1)

    def _clamp_arm_joint_pos(self, robot, joint_pos):
        joint_pos_limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = getattr(robot.data, "joint_pos_limits", None)
        if joint_pos_limits is None:
            return joint_pos
        arm_limits = joint_pos_limits[:, self._arm_joint_ids, :]
        return torch.clamp(joint_pos, arm_limits[..., 0], arm_limits[..., 1])

    def _gripper_down_quat_w(self, robot, num_envs, device, dtype):
        if self._gripper_down_yaw_w is None or self._gripper_down_yaw_w.shape[0] != num_envs:
            base_yaw = self._current_hand_heading_yaw_w(robot).to(device=device, dtype=dtype)
            self._gripper_down_yaw_w = base_yaw.clone()

        roll = torch.full((num_envs,), _GRIPPER_DOWN_ROLL_W, device=device, dtype=dtype)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device, dtype=dtype)
        yaw = self._gripper_down_yaw_w.to(device=device, dtype=dtype)
        return quat_from_euler_xyz(roll, pitch, yaw)

    def _current_hand_heading_yaw_w(self, robot):
        quat_w = self._ee_quat_w(robot)
        local_x = torch.zeros(quat_w.shape[0], 3, device=quat_w.device, dtype=quat_w.dtype)
        local_y = torch.zeros_like(local_x)
        local_x[:, 0] = 1.0
        local_y[:, 1] = 1.0
        hand_x_w = quat_apply(quat_w, local_x)
        hand_y_w = quat_apply(quat_w, local_y)
        yaw_from_x = torch.atan2(hand_x_w[:, 1], hand_x_w[:, 0])
        yaw_from_y = torch.atan2(hand_y_w[:, 0], -hand_y_w[:, 1])
        x_is_horizontal = torch.linalg.norm(hand_x_w[:, :2], dim=-1) > 1e-4
        return torch.where(x_is_horizontal, yaw_from_x, yaw_from_y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count
