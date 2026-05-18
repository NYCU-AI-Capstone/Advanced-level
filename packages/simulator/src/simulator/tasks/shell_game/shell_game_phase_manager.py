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

_CUP_MASS_KG = 0.1


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

        # Phase 1-3 use separate kinematic script cups. Keep the dynamic
        # Act cups hidden so they cannot perturb the scripted shuffle.
        self._apply_cup_mass_properties(env)
        self._set_script_cups_kinematic(env, True)
        self._set_cup_family_visible(env, "script_cup", True)
        self._set_cup_family_visible(env, "cup", False)
        self._hide_dynamic_cups(env)

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

        # Place active scripted cups
        self._place_script_cups(env)

        # Place ball next to the target cup (visible during Reveal)
        ball_x = self._cup_positions[self._ball_cup_idx][0]
        self._set_ball_pose(env, (ball_x, CUP_BASE_Y + 0.06, BALL_Z))

        # Hide unused scripted cups and all dynamic Act cups until handoff.
        for i in range(self._num_cups, 5):
            self._set_script_cup_pose(env, i, HIDDEN_POS)
        self._hide_dynamic_cups(env)

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
            self._handoff_to_dynamic_cups(env)

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
        self._set_script_cup_pose(env, i, tuple(new_i))
        self._set_script_cup_pose(env, j, tuple(new_j))
        if self._ball_cup_idx in (i, j):
            self._sync_ball_to_hiding_cup(env)

        self._step_in_phase += 1

        if self._step_in_phase >= self._shuffle_per_swap_frames:
            # Finalize this swap: snap to exact final positions
            self._cup_positions[i] = list(pos_j_start)
            self._cup_positions[j] = list(pos_i_start)
            self._set_script_cup_pose(env, i, tuple(pos_j_start))
            self._set_script_cup_pose(env, j, tuple(pos_i_start))

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
                if i == self._ball_cup_idx:
                    self._reveal_ball(env)
                return

    # ------------------------------------------------------------------
    # Sim helpers
    # ------------------------------------------------------------------

    def _place_script_cups(self, env: ManagerBasedRLEnv) -> None:
        """Place all active scripted cups at their computed positions."""
        for i in range(self._num_cups):
            self._set_script_cup_pose(env, i, tuple(self._cup_positions[i]))

    def _hide_dynamic_cups(self, env: ManagerBasedRLEnv) -> None:
        """Keep dynamic Act cups out of view until the Act handoff."""
        for i in range(5):
            cup = env.scene[f"cup_{i}"]
            self._set_named_cup_pose(env, f"cup_{i}", HIDDEN_POS)
            self._clear_cup_velocity(env, cup)

    def _handoff_to_dynamic_cups(self, env: ManagerBasedRLEnv) -> None:
        """Swap from kinematic script cups to dynamic Act cups without a visible jump."""
        for i in range(self._num_cups):
            dynamic_cup = env.scene[f"cup_{i}"]
            self._set_named_cup_pose(env, f"cup_{i}", tuple(self._cup_positions[i]))
            self._clear_cup_velocity(env, dynamic_cup)

        # Swap visibility explicitly. Moving scripted cups below the scene is
        # still useful for collision separation, but visibility avoids a one-frame
        # visual overlap that makes cups look larger at handoff.
        self._set_cup_family_visible(env, "script_cup", False)
        self._set_cup_family_visible(env, "cup", True)

        # Hide script cups before the next physics step, so script and dynamic
        # cups are never overlapping collision bodies during simulation.
        for i in range(5):
            self._set_script_cup_pose(env, i, HIDDEN_POS)

        # The FSM reads cup.data immediately after this transition. Refresh the
        # scene buffers so it targets the freshly placed dynamic cups.
        self._refresh_scene_buffers(env)

    def _sync_ball_to_hiding_cup(self, env: ManagerBasedRLEnv) -> None:
        """Keep the hidden ball at the table position of the physical hiding cup."""
        pos = self._cup_positions[self._ball_cup_idx]
        self._set_ball_pose(env, (pos[0], pos[1], BALL_Z))

    def _reveal_ball(self, env: ManagerBasedRLEnv) -> None:
        """Reveal the ball after the correct cup has been lifted."""
        if self._ball_revealed:
            return
        self._sync_ball_to_hiding_cup(env)
        self._ball_revealed = True

    def _set_script_cups_kinematic(self, env: ManagerBasedRLEnv, enabled: bool) -> None:
        """Keep scripted cups kinematic while they are driven by PhaseManager."""
        for i in range(5):
            self._set_asset_kinematic(env, f"script_cup_{i}", enabled)

    def _set_cup_family_visible(self, env: ManagerBasedRLEnv, family: str, visible: bool) -> None:
        for i in range(5):
            self._set_asset_visible(env, f"{family}_{i}", visible)

    def _set_asset_visible(self, env: ManagerBasedRLEnv, asset_name: str, visible: bool) -> None:
        try:
            from pxr import UsdGeom
        except Exception:
            return

        token = "inherited" if visible else "invisible"
        for prim in self._iter_asset_prims(env, asset_name):
            try:
                imageable = UsdGeom.Imageable(prim)
                if not imageable:
                    continue
                imageable.CreateVisibilityAttr().Set(token)
            except Exception:
                continue

    def _apply_cup_mass_properties(self, env: ManagerBasedRLEnv) -> None:
        """Author explicit cup mass on the actual rigid-body prims in the USD asset."""
        for i in range(5):
            self._set_asset_mass(env, f"cup_{i}", _CUP_MASS_KG)
            self._set_asset_mass(env, f"script_cup_{i}", _CUP_MASS_KG)

    def _refresh_scene_buffers(self, env: ManagerBasedRLEnv) -> None:
        update = getattr(env.scene, "update", None)
        if not callable(update):
            return
        try:
            update(dt=getattr(env, "physics_dt", 0.0))
        except Exception:
            return

    def _set_asset_kinematic(self, env: ManagerBasedRLEnv, asset_name: str, enabled: bool) -> None:
        try:
            from pxr import UsdPhysics
        except Exception:
            return

        for prim in self._iter_asset_prims(env, asset_name):
            try:
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                rigid_api.CreateRigidBodyEnabledAttr(True).Set(True)
                rigid_api.CreateKinematicEnabledAttr(bool(enabled)).Set(bool(enabled))
            except Exception:
                continue

    def _set_asset_mass(self, env: ManagerBasedRLEnv, asset_name: str, mass_kg: float) -> None:
        try:
            from pxr import UsdPhysics
        except Exception:
            return

        for prim in self._iter_asset_prims(env, asset_name):
            try:
                if not (prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI)):
                    continue
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateMassAttr(float(mass_kg)).Set(float(mass_kg))
                mass_api.CreateDensityAttr(0.0).Set(0.0)
            except Exception:
                continue

    def _iter_asset_prims(self, env: ManagerBasedRLEnv, asset_name: str):
        asset = env.scene[asset_name]
        stack = list(getattr(asset, "prims", []))
        while stack:
            prim = stack.pop()
            yield prim
            try:
                stack.extend(list(prim.GetAllChildren()))
            except Exception:
                pass

    def _clear_cup_velocity(self, env: ManagerBasedRLEnv, cup) -> None:
        """Clear scripted-phase or hidden-cup residual velocity before handoff."""
        write_velocity = getattr(cup, "write_root_velocity_to_sim", None)
        if not callable(write_velocity):
            return
        try:
            zero_velocity = torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)
            write_velocity(zero_velocity)
        except Exception:
            return

    def _set_script_cup_pose(
        self, env: ManagerBasedRLEnv, cup_index: int, pos: tuple[float, ...]
    ) -> None:
        self._set_named_cup_pose(env, f"script_cup_{cup_index}", pos)

    def _set_named_cup_pose(
        self, env: ManagerBasedRLEnv, cup_name: str, pos: tuple[float, ...]
    ) -> None:
        cup = env.scene[cup_name]
        pose = torch.tensor(
            [[pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]],
            device=env.device,
            dtype=torch.float32,
        ).repeat(env.num_envs, 1)
        cup.write_root_pose_to_sim(pose)

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
