# SPDX-License-Identifier: Apache-2.0
#
# Phase-coded oscillator control for MetaVoxels-style robots in Newton.
#
# The controller is intentionally austere. Every actuator picks one slot from
# a tiny discrete table of (phase, frequency) pairs. At simulation time every
# actuator drives a sinusoidal joint target with its slot's phase. No named
# primitives, no per-timestep decisions, just one integer per actuator.
#
# The design is chosen to stay tractable when the shape itself is being
# searched over by a reinforcement-learning loop. See the companion HTML
# note for the action-space arithmetic.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import warp as wp

import newton


# ----------------------------------------------------------------------------
# Default slot table
# ----------------------------------------------------------------------------
# Three phases spaced by 2pi/3 plus a single frequency is enough for gait-like
# coordination. Extending either table is one line of code. Keep both small
# because the action-space cost for actuator placements is O(cells * K * L).

PHASE_SLOTS_DEFAULT: Tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0)
FREQ_SLOTS_DEFAULT:  Tuple[float, ...] = (1.0,)


# ----------------------------------------------------------------------------
# OscillatorBank
# ----------------------------------------------------------------------------
class OscillatorBank:
    """
    Per-oscillator slot assignments plus a sinusoid evaluator.

    One oscillator per actuator. Each oscillator is bound to a (phase, freq)
    bucket from the slot tables. At time t every oscillator emits

        target_i(t) = amplitude * sin( 2*pi * (f_i * t + phi_i) )

    where phi_i and f_i are looked up from the slot tables using the stored
    bucket indices.

    Buckets are flat integers in [0, K*L). Unpacking is
        phase_bucket = bucket // L
        freq_bucket  = bucket %  L
    where K = len(phase_slots) and L = len(freq_slots).
    """

    def __init__(self,
                 num_oscillators: int,
                 phase_slots: Sequence[float] = PHASE_SLOTS_DEFAULT,
                 freq_slots:  Sequence[float] = FREQ_SLOTS_DEFAULT,
                 base_freq_hz: float = 1.0,
                 amplitude_rad: float = math.radians(45.0)):
        if num_oscillators < 0:
            raise ValueError("num_oscillators must be non-negative")
        if len(phase_slots) == 0 or len(freq_slots) == 0:
            raise ValueError("phase_slots and freq_slots must each have >=1 entry")

        self.num_oscillators = int(num_oscillators)
        self.phase_slots = tuple(float(p) for p in phase_slots)
        self.freq_slots = tuple(float(f) for f in freq_slots)
        self.base_freq_hz = float(base_freq_hz)
        self.amplitude_rad = float(amplitude_rad)

        # Default every oscillator to bucket 0.
        self._phase_bucket = np.zeros(self.num_oscillators, dtype=np.int32)
        self._freq_bucket  = np.zeros(self.num_oscillators, dtype=np.int32)

    # -- sizing ---------------------------------------------------------------

    @property
    def num_phase_slots(self) -> int:
        return len(self.phase_slots)

    @property
    def num_freq_slots(self) -> int:
        return len(self.freq_slots)

    @property
    def bucket_count(self) -> int:
        """Total number of (phase, freq) slot pairs per oscillator. K * L."""
        return self.num_phase_slots * self.num_freq_slots

    # -- slot packing helpers ------------------------------------------------

    def pack(self, phase_bucket: int, freq_bucket: int = 0) -> int:
        self._check_bucket(phase_bucket, freq_bucket)
        return int(phase_bucket) * self.num_freq_slots + int(freq_bucket)

    def unpack(self, bucket_id: int) -> Tuple[int, int]:
        if not (0 <= bucket_id < self.bucket_count):
            raise ValueError(f"bucket_id {bucket_id} out of range")
        return int(bucket_id // self.num_freq_slots), int(bucket_id % self.num_freq_slots)

    def _check_bucket(self, phase_bucket: int, freq_bucket: int) -> None:
        if not (0 <= phase_bucket < self.num_phase_slots):
            raise ValueError(f"phase_bucket {phase_bucket} out of range")
        if not (0 <= freq_bucket < self.num_freq_slots):
            raise ValueError(f"freq_bucket {freq_bucket} out of range")

    # -- assignment -----------------------------------------------------------

    def assign(self, osc_index: int,
               phase_bucket: int,
               freq_bucket: int = 0) -> None:
        """Bind one oscillator to a (phase, freq) bucket."""
        if not (0 <= osc_index < self.num_oscillators):
            raise IndexError(f"osc_index {osc_index} out of range")
        self._check_bucket(phase_bucket, freq_bucket)
        self._phase_bucket[osc_index] = int(phase_bucket)
        self._freq_bucket[osc_index]  = int(freq_bucket)

    def assign_from_bucket_ids(self, bucket_ids: Iterable[int]) -> None:
        """Set all oscillators from a flat list of packed bucket ids."""
        ids = list(bucket_ids)
        if len(ids) != self.num_oscillators:
            raise ValueError(
                f"expected {self.num_oscillators} ids, got {len(ids)}"
            )
        for i, b in enumerate(ids):
            p, f = self.unpack(int(b))
            self.assign(i, p, f)

    def randomize(self, rng: Optional[np.random.Generator] = None) -> None:
        rng = rng or np.random.default_rng()
        self._phase_bucket = rng.integers(0, self.num_phase_slots,
                                          size=self.num_oscillators,
                                          dtype=np.int32)
        self._freq_bucket  = rng.integers(0, self.num_freq_slots,
                                          size=self.num_oscillators,
                                          dtype=np.int32)

    # -- introspection -------------------------------------------------------

    def fingerprint(self) -> str:
        """Compact string summary. Useful for logs and plot titles."""
        parts = []
        for i in range(self.num_oscillators):
            p = self._phase_bucket[i]
            f = self._freq_bucket[i]
            parts.append(f"{p}.{f}")
        return "|".join(parts)

    def bucket_table(self) -> List[Tuple[int, int, float, float]]:
        """(osc, bucket_id, phase_fraction, freq_multiplier) for every oscillator."""
        rows = []
        for i in range(self.num_oscillators):
            pb = int(self._phase_bucket[i])
            fb = int(self._freq_bucket[i])
            rows.append((i,
                         self.pack(pb, fb),
                         self.phase_slots[pb],
                         self.freq_slots[fb]))
        return rows

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, t_seconds: float) -> np.ndarray:
        """Sinusoidal joint targets at time t. Shape (num_oscillators,)."""
        phi = np.asarray([self.phase_slots[b] for b in self._phase_bucket],
                         dtype=np.float32)
        fmul = np.asarray([self.freq_slots[b] for b in self._freq_bucket],
                          dtype=np.float32)
        arg = 2.0 * math.pi * (fmul * self.base_freq_hz * float(t_seconds) + phi)
        return (self.amplitude_rad * np.sin(arg)).astype(np.float32)


# ----------------------------------------------------------------------------
# SinusoidalJointController
# ----------------------------------------------------------------------------
class SinusoidalJointController:
    """
    Binds an OscillatorBank to a Newton model. Each tick it evaluates the bank
    at the current time and writes the result into control.joint_target_pos.
    The DOF indices are resolved from the model so the binding survives
    replication or parallel worlds (pass the matching slice of dof indices).
    """

    def __init__(self,
                 model: "newton.Model",
                 control: "newton.Control",
                 bank: OscillatorBank,
                 dof_indices: Sequence[int]):
        if len(dof_indices) != bank.num_oscillators:
            raise ValueError(
                f"dof_indices has {len(dof_indices)} entries, "
                f"bank expects {bank.num_oscillators}"
            )
        self.model = model
        self.control = control
        self.bank = bank
        self.dof_indices = np.asarray(list(dof_indices), dtype=np.int32)
        self._host_targets = np.zeros(int(model.joint_dof_count), dtype=np.float32)

    def push(self, t_seconds: float) -> None:
        """Evaluate oscillators at t and copy the result to the GPU."""
        vals = self.bank.evaluate(t_seconds)
        for local_i, dof_i in enumerate(self.dof_indices):
            self._host_targets[int(dof_i)] = float(vals[local_i])
        wp_targets = wp.array(self._host_targets, dtype=wp.float32,
                              device=self.model.device)
        wp.copy(self.control.joint_target_pos, wp_targets)


# ----------------------------------------------------------------------------
# Joint / gain helpers (same utility as before but stripped down)
# ----------------------------------------------------------------------------
def find_revolute_dofs(model: "newton.Model") -> List[int]:
    """Indices in joint_target_pos for every revolute DOF in the model."""
    joint_type = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    rev = int(newton.JointType.REVOLUTE)
    return [int(joint_qd_start[j]) for j, jt in enumerate(joint_type) if int(jt) == rev]


def set_pd_gains(builder: "newton.ModelBuilder",
                 kp: float = 60.0,
                 kd: float = 5.0) -> int:
    """Configure every revolute DOF for TARGET_POSITION with the given PD."""
    rev = int(newton.JointType.REVOLUTE)
    count = 0
    joint_dof_count = getattr(builder, "joint_dof_count", 1)
    dof_count_is_scalar = np.isscalar(joint_dof_count)
    qd_starts = list(getattr(builder, "joint_qd_start", []))
    target_len = len(builder.joint_target_ke)
    target_mode = None
    mode_array = None
    if hasattr(newton, "JointTargetMode") and hasattr(builder, "joint_target_mode"):
        target_mode = int(newton.JointTargetMode.POSITION)
        mode_array = builder.joint_target_mode
    elif hasattr(newton, "JointMode") and hasattr(builder, "joint_dof_mode"):
        target_mode = int(newton.JointMode.TARGET_POSITION)
        mode_array = builder.joint_dof_mode

    # Some imported assets may leave mode arrays shorter than target gain arrays.
    # Extend in-place so revolute DOFs can be explicitly actuated.
    if mode_array is not None and isinstance(mode_array, list) and len(mode_array) < target_len:
        fill = 0
        if hasattr(newton, "JointTargetMode"):
            fill = int(newton.JointTargetMode.NONE)
        mode_array.extend([fill] * (target_len - len(mode_array)))

    for j, jt in enumerate(builder.joint_type):
        if int(jt) != rev:
            continue
        start = builder.joint_qd_start[j]
        if not dof_count_is_scalar:
            dofs = int(joint_dof_count[j])
        elif qd_starts and j + 1 < len(qd_starts):
            dofs = int(qd_starts[j + 1] - qd_starts[j])
        else:
            dofs = int(target_len - start)
        if dofs <= 0:
            continue
        for d in range(dofs):
            idx = start + d
            if idx < 0 or idx >= target_len:
                continue
            if mode_array is not None:
                mode_array[idx] = target_mode
            builder.joint_target_ke[idx] = kp
            builder.joint_target_kd[idx] = kd
            count += 1
    return count


def drop_robot_self_contacts(robot, builder: "newton.ModelBuilder") -> int:
    """Add every intra-robot shape pair to the broadphase filter list."""
    shapes = (list(getattr(robot, "control_voxel_shapes", []))
              + list(getattr(robot, "rigid_voxel_shapes", []))
              + list(getattr(robot, "joint_shapes", [])))
    n = 0
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            builder.shape_collision_filter_pairs.append((shapes[i], shapes[j]))
            n += 1
    return n


# ----------------------------------------------------------------------------
# Discrete action-space helpers for co-design RL
# ----------------------------------------------------------------------------
# Shape of the flat action index when some voxel types are "actuator" types
# that carry an extra (phase, freq) bucket at placement time.
#
#   flat_size = [include_stop]
#             + sum_{t in nonactuator_types} cells
#             + sum_{t in actuator_types}    cells * num_buckets
#
# The encode / decode pair below is careful to keep actuator entries grouped
# so that masking kernels can disable a whole type with a contiguous slice.

@dataclass(frozen=True)
class CoDesignActionLayout:
    """Describes how the flat action integer maps to (type, cell, bucket)."""
    grid_side: int
    num_types: int
    actuator_types: Tuple[int, ...]
    num_buckets: int
    include_stop: bool = True

    @property
    def cells(self) -> int:
        return self.grid_side ** 3

    @property
    def stop_offset(self) -> int:
        return 1 if self.include_stop else 0

    def type_slot_size(self, type_id: int) -> int:
        """How many flat action ids one voxel type occupies."""
        if type_id in self.actuator_types:
            return self.cells * self.num_buckets
        return self.cells

    def type_start(self, type_id: int) -> int:
        """First flat action id belonging to the given type."""
        offset = self.stop_offset
        for t in range(type_id):
            offset += self.type_slot_size(t)
        return offset

    def total_size(self) -> int:
        return self.stop_offset + sum(self.type_slot_size(t) for t in range(self.num_types))


def encode_codesign_action(layout: CoDesignActionLayout,
                           *,
                           is_stop: bool,
                           type_id: int = 0,
                           x: int = 0, y: int = 0, z: int = 0,
                           bucket_id: int = 0) -> int:
    """Pack a placement (or stop) into the flat discrete action index."""
    if is_stop:
        if not layout.include_stop:
            raise ValueError("layout has no stop action")
        return 0
    if not (0 <= type_id < layout.num_types):
        raise ValueError(f"type_id {type_id} out of range")
    n = layout.grid_side
    cell_idx = x * n * n + y * n + z
    base = layout.type_start(type_id)
    if type_id in layout.actuator_types:
        if not (0 <= bucket_id < layout.num_buckets):
            raise ValueError(f"bucket_id {bucket_id} out of range")
        return base + bucket_id * layout.cells + cell_idx
    return base + cell_idx


def decode_codesign_action(layout: CoDesignActionLayout,
                           action: int
                           ) -> Tuple[bool, int, int, int, int, int]:
    """
    Return (is_stop, type_id, x, y, z, bucket_id). bucket_id is 0 for
    non-actuator types regardless of the flat value.
    """
    if layout.include_stop and action == 0:
        return True, -1, -1, -1, -1, 0

    offset = layout.stop_offset
    for t in range(layout.num_types):
        size = layout.type_slot_size(t)
        if action < offset + size:
            local = action - offset
            n = layout.grid_side
            cells = layout.cells
            if t in layout.actuator_types:
                bucket_id = local // cells
                cell_idx = local % cells
            else:
                bucket_id = 0
                cell_idx = local
            x = cell_idx // (n * n)
            y = (cell_idx // n) % n
            z = cell_idx % n
            return False, t, x, y, z, bucket_id
        offset += size
    raise ValueError(f"action {action} out of range for layout of size {layout.total_size()}")


# ----------------------------------------------------------------------------
# Rollout stats + helper
# ----------------------------------------------------------------------------
@dataclass
class RolloutStats:
    frames: int = 0
    duration_s: float = 0.0
    start_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end_xyz:   Tuple[float, float, float] = (0.0, 0.0, 0.0)
    displacement_xy: float = 0.0
    forward_x: float = 0.0
    sideways_y: float = 0.0
    height_delta_z: float = 0.0
    max_forward_speed: float = 0.0
    upright_cosine_mean: float = 0.0
    fell_over: bool = False
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    bank_fingerprint: str = ""


def _first_body_index(model: "newton.Model", articulation_key: str) -> int:
    keys = list(model.articulation_key)
    if articulation_key not in keys:
        return 0
    art_idx = keys.index(articulation_key)
    art_start = model.articulation_start.numpy()
    joint_child = model.joint_child.numpy()
    j_lo = int(art_start[art_idx])
    j_hi = int(art_start[art_idx + 1]) if art_idx + 1 < len(art_start) else len(joint_child)
    bodies = sorted({int(joint_child[j]) for j in range(j_lo, j_hi)
                     if joint_child[j] >= 0})
    return bodies[0] if bodies else 0


def run_oscillatory_rollout(example,
                            controller: SinusoidalJointController,
                            duration_s: float,
                            robot_name: str = "robot",
                            sample_every: int = 1,
                            render: bool = False) -> RolloutStats:
    """
    Simulate the example for ``duration_s`` while pushing sinusoidal targets
    from the controller every frame. Returns motion statistics.
    """
    body_idx = _first_body_index(example.model, robot_name)

    def read_xyz() -> Tuple[float, float, float]:
        bq = example.state_0.body_q.numpy()
        p = bq[body_idx]
        return float(p[0]), float(p[1]), float(p[2])

    def read_up_cos() -> float:
        bq = example.state_0.body_q.numpy()
        qx, qy, qz, qw = (float(bq[body_idx, i]) for i in (3, 4, 5, 6))
        return 1.0 - 2.0 * (qx * qx + qy * qy)

    start = read_xyz()
    traj: List[Tuple[float, float, float]] = [start]
    up_samples: List[float] = [read_up_cos()]

    # Drive the sim.
    t = 0.0
    frames = 0
    prev = start
    max_v = 0.0
    total_frames = int(round(duration_s / example.frame_dt))

    for _ in range(total_frames):
        controller.push(t)
        example.step()
        if render and hasattr(example, "render"):
            example.render()
        t += example.frame_dt
        frames += 1

        if frames % sample_every == 0:
            cur = read_xyz()
            v = (cur[0] - prev[0]) / example.frame_dt if example.frame_dt > 0 else 0.0
            if v > max_v:
                max_v = v
            traj.append(cur)
            up_samples.append(read_up_cos())
            prev = cur

    end = read_xyz()
    dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
    up_mean = float(np.mean(up_samples)) if up_samples else 0.0

    return RolloutStats(
        frames=frames,
        duration_s=frames * example.frame_dt,
        start_xyz=start,
        end_xyz=end,
        displacement_xy=float(math.hypot(dx, dy)),
        forward_x=float(dx),
        sideways_y=float(dy),
        height_delta_z=float(dz),
        max_forward_speed=float(max_v),
        upright_cosine_mean=up_mean,
        fell_over=(up_mean < 0.2),
        trajectory=traj,
        bank_fingerprint=controller.bank.fingerprint(),
    )


def pretty_stats(s: RolloutStats) -> str:
    return "\n".join([
        f"bank fingerprint    : {s.bank_fingerprint}",
        f"frames              : {s.frames}   ({s.duration_s:.2f} s)",
        f"start               : ({s.start_xyz[0]:+.3f}, {s.start_xyz[1]:+.3f}, {s.start_xyz[2]:+.3f})",
        f"end                 : ({s.end_xyz[0]:+.3f}, {s.end_xyz[1]:+.3f}, {s.end_xyz[2]:+.3f})",
        f"displacement (xy)   : {s.displacement_xy:+.3f} m",
        f"  forward  (x)      : {s.forward_x:+.3f} m",
        f"  sideways (y)      : {s.sideways_y:+.3f} m",
        f"height delta (z)    : {s.height_delta_z:+.3f} m",
        f"max forward speed   : {s.max_forward_speed:+.3f} m/s",
        f"mean upright cosine : {s.upright_cosine_mean:+.3f}",
        f"fell over           : {s.fell_over}",
    ])
