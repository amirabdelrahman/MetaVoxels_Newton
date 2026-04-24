# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0
#
# Newton port of the MetaVoxels ant, driven by a phase-coded oscillator bank.
# Each revolute joint is assigned a slot from a tiny discrete table of
# (phase, frequency) pairs. The action space for learning this assignment
# scales as cells * num_types * K * L, which stays tractable.

import math
import os
import sys

import numpy as np
import warp as wp

import newton
import newton.examples

try:
    from .metavoxels_library import RobotZoo
    from .metavoxels_control import (
        CoDesignActionLayout,
        OscillatorBank,
        SinusoidalJointController,
        decode_codesign_action,
        drop_robot_self_contacts,
        encode_codesign_action,
        find_revolute_dofs,
        pretty_stats,
        run_oscillatory_rollout,
        set_pd_gains,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from metavoxels_library import RobotZoo
    from metavoxels_control import (
        CoDesignActionLayout,
        OscillatorBank,
        SinusoidalJointController,
        decode_codesign_action,
        drop_robot_self_contacts,
        encode_codesign_action,
        find_revolute_dofs,
        pretty_stats,
        run_oscillatory_rollout,
        set_pd_gains,
    )


FILE_DIR = os.path.dirname(os.path.realpath(__file__))


# ----------------------------------------------------------------------------
# Runtime flags
# ----------------------------------------------------------------------------
ENABLE_SELF_COLLISION = False   # keep contact count near the foot count
USE_OSCILLATOR_CONTROL = True   # install PD gains and a phase-coded bank
# ----------------------------------------------------------------------------


class Example:
    """Newton MetaVoxels ant with a phase-coded oscillator controller."""

    def __init__(self,
                 viewer,
                 args,
                 enable_self_collision: bool = ENABLE_SELF_COLLISION,
                 bank_assignment=None,
                 base_freq_hz: float = 1.2,
                 amplitude_deg: float = 55.0):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.robot_name = "robot"

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        usd_path = os.path.realpath(
            os.path.join(FILE_DIR, "./data/USD/MetaVoxelsNew/")
        )
        zoo = RobotZoo(builder=builder, size=1.0, usd_path=usd_path)
        self.robot = zoo.create_ant1(
            name=self.robot_name,
            # Spawn close to ground so actuation is visible right away.
            pos=wp.vec3(0.0, 0.0, 0.8),
        )

        if not enable_self_collision:
            removed = drop_robot_self_contacts(self.robot, builder)
            print(f"[metavoxels] dropped {removed} robot self-collision pairs")

        if USE_OSCILLATOR_CONTROL:
            # Use stiffer gains, consistent with working robot examples.
            configured = set_pd_gains(builder, kp=400.0, kd=30.0)
            print(f"[metavoxels] PD gains on {configured} revolute DOFs")

        self.model = builder.finalize()

        # Keep MuJoCo constraint buffers comfortably above observed peaks.
        # Some poses briefly require >76 equality/friction constraints.
        try:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, nconmax=512, njmax=256
            )
        except ImportError:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=20)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd,
                       self.state_0)
        self.contacts = self.model.contacts()

        self.viewer = viewer
        self.viewer.set_model(self.model)

        if USE_OSCILLATOR_CONTROL:
            dofs = find_revolute_dofs(self.model)
            self.bank = OscillatorBank(
                num_oscillators=len(dofs),
                base_freq_hz=base_freq_hz,
                amplitude_rad=math.radians(amplitude_deg),
            )
            if bank_assignment is not None:
                self.bank.assign_from_bucket_ids(bank_assignment)
            else:
                # Default: alternating phase 0 / 1 down the list, which gives
                # a diagonal-ish gait for the ant body layout.
                for i in range(len(dofs)):
                    self.bank.assign(i, phase_bucket=(i % self.bank.num_phase_slots))

            self.controller = SinusoidalJointController(
                model=self.model,
                control=self.control,
                bank=self.bank,
                dof_indices=dofs,
            )
            self.controller.push(0.0)  # seed targets before graph capture
        else:
            self.bank = None
            self.controller = None

        self._capture_graph()

    # ------------------------------------------------------------------ sim --
    def _capture_graph(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.controller is not None:
            self.controller.push(self.sim_time)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


# ----------------------------------------------------------------------------
# Headless rollout (no viewer): handy for fitness evaluation and RL scoring.
# ----------------------------------------------------------------------------
def evaluate_bank(bucket_ids,
                  duration_s: float = 8.0,
                  base_freq_hz: float = 1.2,
                  amplitude_deg: float = 40.0):
    """
    Build the ant with the given oscillator assignment, simulate, and return
    RolloutStats. bucket_ids is a list of one packed bucket id per revolute
    joint (8 entries for create_ant1).
    """
    class NullViewer:
        def set_model(self, *a, **k): pass
        def apply_forces(self, *a, **k): pass
        def set_world_offsets(self, *a, **k): pass
        def begin_frame(self, *a, **k): pass
        def log_state(self, *a, **k): pass
        def log_contacts(self, *a, **k): pass
        def end_frame(self, *a, **k): pass

    ex = Example(NullViewer(),
                 args=None,
                 bank_assignment=bucket_ids,
                 base_freq_hz=base_freq_hz,
                 amplitude_deg=amplitude_deg)
    return run_oscillatory_rollout(ex, ex.controller, duration_s=duration_s,
                                   robot_name=ex.robot_name)


# ----------------------------------------------------------------------------
# CLI / viewer entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    viewer, args = newton.examples.init()
    # Example assignment for the 8-DOF ant: alternating phases plus one outlier
    # to break symmetry and encourage forward motion.
    default_plan = [0, 1, 0, 1, 1, 0, 1, 0]
    example = Example(viewer, args, bank_assignment=default_plan)
    newton.examples.run(example, args)
