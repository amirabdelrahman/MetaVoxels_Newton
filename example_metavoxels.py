# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Newton port of the MetaVoxels Omniverse sample. Build a voxelized robot
# from modular USD pieces and simulate it in Newton.

import os
import sys

import warp as wp

import newton
import newton.examples

# Support both "python example_metavoxels.py" and package-style imports
try:
    from .metavoxels_library import RobotZoo
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from metavoxels_library import RobotZoo


FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class Example:
    """Newton port of the MetaVoxels sample."""

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        usd_path = os.path.realpath(
            os.path.join(FILE_DIR, "./data/USD/MetaVoxelsNew/")
        )
        zoo = RobotZoo(builder=builder, size=1.0, usd_path=usd_path)

        # spawn an ant at z = 2 so it clearly drops onto the ground
        self.robot = zoo.create_ant1(
            name="robot",
            pos=wp.vec3(0.0, 0.0, 5.0),
        )

        # finalize model
        self.model = builder.finalize()

        # MuJoCo solver gives the smoothest behavior on articulations with
        # fixed joints and revolute chains. Fall back to XPBD if the MuJoCo
        # backend isn't installed.
        try:
            self.solver = newton.solvers.SolverMuJoCo(self.model)
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

        self._capture_graph()

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


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
