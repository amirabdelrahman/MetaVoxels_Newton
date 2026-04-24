"""Smoke test for the rewritten metavoxels library.

Builds the zoo's robots without USD files present (so every piece falls
back to a size-fitting box), verifies the articulation graph compiles,
and steps the simulation for a handful of frames to make sure nothing
NaNs or blows up.
"""
import sys
import os
sys.path.insert(0, '/home/claude')

import warp as wp
import newton

# make the library importable as a top-level module
import metavoxels_library as mv


def build_and_step(builder_factory, label):
    print(f"\n=== {label} ===")
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    # usd_path does not exist, so we exercise the box fallback
    zoo = mv.RobotZoo(builder=builder, size=1.0,
                      usd_path="/nonexistent/MetaVoxels/")

    robot = builder_factory(zoo)

    # report graph stats before finalize
    print(f"  robot joints queued  : {len(robot._joints)}")
    print(f"  robot root body idx  : {robot._root_body}")

    model = builder.finalize()
    print(f"  model.body_count     : {model.body_count}")
    print(f"  model.shape_count    : {model.shape_count}")
    print(f"  model.joint_count    : {model.joint_count}")
    print(f"  model.articulation_count: {model.articulation_count}")

    # inspect joint types to verify revolute chains exist
    jtypes = model.joint_type.numpy()
    from collections import Counter
    # joint type enum: 0=prismatic, 1=revolute, 2=ball, 3=fixed, 4=free ...
    # (mapping matches newton.JointType, but the numbers are all we need here)
    print(f"  joint_type counts    : {dict(Counter(jtypes.tolist()))}")

    # check initial body positions make sense
    body_q = model.body_q.numpy()
    print(f"  body_q (first 4 z)   : {body_q[:4, 2]}")

    # Step the sim a few frames
    try:
        solver = newton.solvers.SolverXPBD(model, iterations=20)
        s0 = model.state()
        s1 = model.state()
        ctrl = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

        for i in range(10):
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, 1.0 / 600.0)
            s0, s1 = s1, s0

        import numpy as np
        final_q = s0.body_q.numpy()
        print(f"  after 10 steps, z range: [{final_q[:,2].min():.3f}, "
              f"{final_q[:,2].max():.3f}]")
        if np.isnan(final_q).any():
            print("  !! NaN detected in final state")
            return False
    except Exception as e:
        print(f"  !! simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  OK")
    return True


def main():
    ok = True
    ok &= build_and_step(
        lambda z: z.create_robot_trial(name="trial",
                                       pos=wp.vec3(0, 0, 2.0)),
        "trial (minimum viable)",
    )
    ok &= build_and_step(
        lambda z: z.create_ant1(name="ant", pos=wp.vec3(0, 0, 2.0)),
        "ant1 (4 legs, 8 revolute)",
    )
    ok &= build_and_step(
        lambda z: z.create_ant(name="ant2", pos=wp.vec3(0, 0, 2.0)),
        "ant (4 legs, signed limits)",
    )
    ok &= build_and_step(
        lambda z: z.create_snake(name="snake", pos=wp.vec3(0, 0, 2.0),
                                 n_segments=5),
        "snake (5 segments)",
    )
    ok &= build_and_step(
        lambda z: z.create_billE(name="bill", pos=wp.vec3(0, 0, 2.0)),
        "billE (fixed to world)",
    )

    print("\n" + ("all green" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
