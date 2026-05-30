"""Phase manager for the ShellBench shell game task.

Manages four phases:
  Phase 1 (Reveal):  Ball visible next to a cup.
  Phase 2 (Cover):   Cup covers the ball; ball hidden.
  Phase 3 (Shuffle): Cups swap positions N times along arc trajectories.
  Phase 4 (Act):     Control handed to policy / state machine.
"""

from __future__ import annotations

import math
import random
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .shell_game_env_cfg import (
    BALL_Z,
    CUP_BASE_Y,
    CUP_Z,
    HIDDEN_POS,
    _cup_x_positions,
)


class Phase(IntEnum):
    REVEAL = 0
    COVER = 1
    SHUFFLE = 2
    ACT = 3


class ShellGamePhaseManager:
    """Drives the non-policy phases of the shell game and tracks ground truth."""

    def __init__(
        self,
        num_cups: int = 3,
        num_shuffles: int = 2,
        shuffle_speed: float = 1.0,
        ball_position: str = "random",
        seed: int = 42,
        reveal_frames: int = 50,
        cover_frames: int = 10,
        shuffle_per_swap_frames: int = 30,
        act_frames: int = 150,
    ):
        self._num_cups = num_cups
        self._num_shuffles = num_shuffles
        self._shuffle_speed = shuffle_speed
        self._ball_position_cfg = ball_position
        self._reveal_frames = reveal_frames
        self._cover_frames = cover_frames
        self._shuffle_per_swap_frames = max(1, int(shuffle_per_swap_frames / shuffle_speed))
        self._act_frames = act_frames

        self._rng = random.Random(seed)

        # Per-episode state
        self._phase: Phase = Phase.REVEAL
        self._step_in_phase: int = 0
        self._ball_cup_idx: int = 0
        self._shuffle_sequence: list[tuple[int, int]] = []
        self._current_shuffle_idx: int = 0
        self._cup_positions: list[list[float]] = []
        self._cup_init_positions: list[list[float]] = []
        self._selected_cup_index: int | None = None
        self._ball_hidden: bool = False
        self._ball_revealed: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @classmethod
    def from_env_cfg(cls, cfg) -> ShellGamePhaseManager:
        return cls(
            num_cups=cfg.num_cups,
            num_shuffles=cfg.num_shuffles,
            shuffle_speed=cfg.shuffle_speed,
            ball_position=cfg.ball_position,
            seed=cfg.shell_game_seed,
            reveal_frames=cfg.reveal_frames,
            cover_frames=cfg.cover_frames,
            shuffle_per_swap_frames=cfg.shuffle_per_swap_frames,
            act_frames=cfg.act_frames,
        )

    def reset(self, env: ManagerBasedRLEnv) -> None:
        """Reset for a new episode: pick ball position, generate shuffles, place objects."""
        self._phase = Phase.REVEAL
        self._step_in_phase = 0
        self._current_shuffle_idx = 0
        self._selected_cup_index = None
        self._ball_hidden = False
        self._ball_revealed = False

        # Phase 1-3 are scripted; keep cups kinematic while the manager moves them.
        # Ball is always kinematic (visual marker only, no collision).
        self._set_cups_kinematic(env, True)

        # Decide ball position
        if self._ball_position_cfg == "random":
            self._ball_cup_idx = self._rng.randint(0, self._num_cups - 1)
        else:
            self._ball_cup_idx = int(self._ball_position_cfg) % self._num_cups

        # Generate shuffle sequence
        self._shuffle_sequence = []
        for _ in range(self._num_shuffles):
            i = self._rng.randint(0, self._num_cups - 1)
            j = self._rng.randint(0, self._num_cups - 2)
            if j >= i:
                j += 1
            self._shuffle_sequence.append((i, j))

        # Compute cup positions
        x_positions = _cup_x_positions(self._num_cups)
        self._cup_positions = [[x, CUP_BASE_Y, CUP_Z] for x in x_positions]
        self._cup_init_positions = [pos[:] for pos in self._cup_positions]

        # Place active cups
        self._place_cups(env)

        # Place ball next to the target cup (visible during Reveal)
        ball_x = self._cup_positions[self._ball_cup_idx][0]
        self._set_ball_pose(env, (ball_x, CUP_BASE_Y + 0.06, BALL_Z))

        # Hide unused cups
        for i in range(self._num_cups, 5):
            self._set_cup_pose(env, i, HIDDEN_POS)

    def step(self, env: ManagerBasedRLEnv) -> bool:
        """Advance one step. Returns True when in Act phase (policy should control)."""
        if self._phase == Phase.REVEAL:
            self._step_in_phase += 1
            if self._step_in_phase >= self._reveal_frames:
                self._transition_to_cover(env)

        elif self._phase == Phase.COVER:
            self._step_in_phase += 1
            if self._step_in_phase >= self._cover_frames:
                self._transition_to_shuffle(env)

        elif self._phase == Phase.SHUFFLE:
            self._step_shuffle(env)

        elif self._phase == Phase.ACT:
            self._step_in_phase += 1
            self._update_selection(env)
            if not self._ball_revealed:
                self._track_ball_to_hiding_cup(env)
            return True

        return self._phase == Phase.ACT

    def update_selection(self, env: ManagerBasedRLEnv) -> None:
        """Check if any cup was lifted (for external callers during Act phase)."""
        self._update_selection(env)

    @property
    def ball_true_cup_index(self) -> int:
        return self._ball_cup_idx

    @property
    def cup_positions(self) -> list[list[float]]:
        return [pos[:] for pos in self._cup_positions[:self._num_cups]]

    @property
    def selected_cup_index(self) -> int | None:
        return self._selected_cup_index

    @property
    def current_phase(self) -> Phase:
        return self._phase

    @property
    def num_cups(self) -> int:
        return self._num_cups

    @property
    def is_act_phase(self) -> bool:
        return self._phase == Phase.ACT

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _transition_to_cover(self, env: ManagerBasedRLEnv) -> None:
        """Ball disappears under the cup."""
        self._phase = Phase.COVER
        self._step_in_phase = 0
        self._ball_hidden = True
        self._ball_revealed = False
        # Keep the ball under the physical hiding cup. It should be visually
        # occluded by the cup, and will become visible when the correct cup lifts.
        self._sync_ball_to_hiding_cup(env)

    def _transition_to_shuffle(self, env: ManagerBasedRLEnv) -> None:
        """Begin shuffle phase."""
        self._phase = Phase.SHUFFLE
        self._step_in_phase = 0
        self._current_shuffle_idx = 0
        if self._num_shuffles == 0:
            self._transition_to_act(env)

    def _transition_to_act(self, env: ManagerBasedRLEnv | None = None) -> None:
        """Hand control to policy."""
        self._phase = Phase.ACT
        self._step_in_phase = 0
        if env is not None:
            # Final sync: ensure ball is centered under the hiding cup.
            self._sync_ball_to_hiding_cup(env)
            # Cups become dynamic so the gripper can physically lift them.
            # Ball stays kinematic (visual marker).
            self._set_cups_kinematic(env, False)

    # ------------------------------------------------------------------
    # Shuffle logic
    # ------------------------------------------------------------------

    def _step_shuffle(self, env: ManagerBasedRLEnv) -> None:
        """Animate one step of the current shuffle swap."""
        if self._current_shuffle_idx >= len(self._shuffle_sequence):
            self._transition_to_act(env)
            return

        i, j = self._shuffle_sequence[self._current_shuffle_idx]
        t = self._step_in_phase / max(self._shuffle_per_swap_frames - 1, 1)
        t = min(t, 1.0)

        pos_i_start = self._cup_init_positions[i]
        pos_j_start = self._cup_init_positions[j]

        # Arc interpolation: cups swap along semicircular paths
        arc_height = 0.04
        sin_t = math.sin(math.pi * t)

        # Cup i moves from pos_i_start toward pos_j_start
        new_i = [
            pos_i_start[0] * (1 - t) + pos_j_start[0] * t,
            pos_i_start[1] + sin_t * arc_height,
            pos_i_start[2],
        ]
        # Cup j moves from pos_j_start toward pos_i_start
        new_j = [
            pos_j_start[0] * (1 - t) + pos_i_start[0] * t,
            pos_j_start[1] - sin_t * arc_height,
            pos_j_start[2],
        ]

        self._cup_positions[i] = new_i
        self._cup_positions[j] = new_j
        self._set_cup_pose(env, i, tuple(new_i))
        self._set_cup_pose(env, j, tuple(new_j))
        if self._ball_cup_idx in (i, j):
            self._sync_ball_to_hiding_cup(env)

        self._step_in_phase += 1

        if self._step_in_phase >= self._shuffle_per_swap_frames:
            # Finalize this swap: snap to exact final positions
            self._cup_positions[i] = list(pos_j_start)
            self._cup_positions[j] = list(pos_i_start)
            self._set_cup_pose(env, i, tuple(pos_j_start))
            self._set_cup_pose(env, j, tuple(pos_i_start))

            # Update init positions for next swap
            self._cup_init_positions[i], self._cup_init_positions[j] = (
                list(pos_j_start),
                list(pos_i_start),
            )

            # _ball_cup_idx is the physical cup id hiding the ball. The cup's
            # position changed during the swap, but the hiding cup id does not.
            self._sync_ball_to_hiding_cup(env)

            self._current_shuffle_idx += 1
            self._step_in_phase = 0

            if self._current_shuffle_idx >= len(self._shuffle_sequence):
                self._transition_to_act(env)

    # ------------------------------------------------------------------
    # Selection detection
    # ------------------------------------------------------------------

    def _update_selection(self, env: ManagerBasedRLEnv) -> None:
        """Detect which cup was first lifted above threshold."""
        if self._selected_cup_index is not None:
            return

        lift_threshold = 0.05
        for i in range(self._num_cups):
            cup = env.scene[f"cup_{i}"]
            cup_z = (cup.data.root_pos_w - env.scene.env_origins)[0, 2].item()
            if cup_z > CUP_Z + lift_threshold:
                self._selected_cup_index = i
                self._ball_revealed = True
                return

    # ------------------------------------------------------------------
    # Sim helpers
    # ------------------------------------------------------------------

    def _place_cups(self, env: ManagerBasedRLEnv) -> None:
        """Place all active cups at their computed positions."""
        for i in range(self._num_cups):
            self._set_cup_pose(env, i, tuple(self._cup_positions[i]))

    def _sync_ball_to_hiding_cup(self, env: ManagerBasedRLEnv) -> None:
        """Keep the hidden ball at the table position of the physical hiding cup."""
        pos = self._cup_positions[self._ball_cup_idx]
        self._set_ball_pose(env, (pos[0], pos[1], BALL_Z))

    def _track_ball_to_hiding_cup(self, env: ManagerBasedRLEnv) -> None:
        """During ACT, follow the hiding cup's x,y but stay at table height."""
        cup = env.scene[f"cup_{self._ball_cup_idx}"]
        cup_pos = (cup.data.root_pos_w - env.scene.env_origins)[0]
        self._set_ball_pose(env, (cup_pos[0].item(), cup_pos[1].item(), BALL_Z))

    def _set_cups_kinematic(self, env: ManagerBasedRLEnv, enabled: bool) -> None:
        try:
            import re
            import omni.usd
            from pxr import Usd, UsdPhysics
        except Exception:
            return

        stage = omni.usd.get_context().get_stage()

        for i in range(self._num_cups):
            cup = env.scene[f"cup_{i}"]
            cfg_path = getattr(getattr(cup, "cfg", None), "prim_path", "")
            pattern = re.compile("^" + cfg_path + "$")

            roots = [prim for prim in stage.Traverse() if pattern.match(str(prim.GetPath()))]
            for root in roots:
                for prim in Usd.PrimRange(root):
                    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        continue
                    api = UsdPhysics.RigidBodyAPI.Apply(prim)
                    attr = api.GetKinematicEnabledAttr()
                    if not attr:
                        attr = api.CreateKinematicEnabledAttr()
                    attr.Set(bool(enabled))

    def _set_cup_pose(
        self, env: ManagerBasedRLEnv, cup_index: int, pos: tuple[float, ...]
    ) -> None:
        cup = env.scene[f"cup_{cup_index}"]
        # rot (0, 1, 0, 0) = 180 deg about X so the cup stays inverted (mouth-down).
        # Runs every frame in Phases 1-3, so it must match _make_cup_cfg's init rot;
        # otherwise the cup would be flipped back upright here.
        pose = torch.tensor(
            [[pos[0], pos[1], pos[2], 0.0, 1.0, 0.0, 0.0]],
            device=env.device,
            dtype=torch.float32,
        ).repeat(env.num_envs, 1)
        cup.write_root_pose_to_sim(pose)
        if hasattr(cup, "write_root_velocity_to_sim"):
            cup.write_root_velocity_to_sim(
                torch.zeros(env.num_envs, 6, device=env.device)
            )

    def _set_ball_pose(
        self, env: ManagerBasedRLEnv, pos: tuple[float, ...]
    ) -> None:
        ball = env.scene["ball"]
        pose = torch.tensor(
            [[pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]],
            device=env.device,
            dtype=torch.float32,
        ).repeat(env.num_envs, 1)
        ball.write_root_pose_to_sim(pose)
        if hasattr(ball, "write_root_velocity_to_sim"):
            ball.write_root_velocity_to_sim(
                torch.zeros(env.num_envs, 6, device=env.device)
            )
