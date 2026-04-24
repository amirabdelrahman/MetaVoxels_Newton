# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# MetaVoxels library rewritten for Newton Physics 1.1.0.
#
# Each modular piece (rigid/control voxel, elbow/wrist joint, end cap) owns
# one rigid body; joints are a pair of co-located bodies connected by a
# revolute joint. The graph topology is preserved from the original Isaac Sim
# authoring pipeline so the zoo of robot recipes (createAnt1, createRover,
# createBillE, etc.) can be ported directly.
#
# The tricky parts that tripped up the previous port are handled here as
# follows:
#   * Mesh visuals: the shape `xform` parameter (new in Newton 1.1) carries
#     the Isaac rotate/shift inside the body local frame, and the Isaac
#     axis-swap that used to be a 4x4 on the outer prim is a fixed rotation
#     baked into the shape transform. No vertex math, no recentering, no
#     post-hoc nudging.
#   * Articulation structure: joints always form parent-to-child chains in a
#     proper tree, with the correct side of the hinge (link1 vs link2)
#     chosen based on the placement direction so the Isaac fixJoint i%2 rule
#     is respected.
#   * Anchor points: fixed joints anchor at the parent body center and at
#     -size * direction in the child, matching Isaac localPos0/localPos1.
#     The two bodies of a revolute joint share a world position and the
#     hinge anchor sits at (0, 0, 0.5 * size) in each.

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import warp as wp

import newton
import newton.usd

try:
    from pxr import Usd, UsdGeom
    _USD_AVAILABLE = True
except Exception:
    _USD_AVAILABLE = False


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

DENSITY = 35.0              # kg / m^3, matches Isaac
WRIST_SIZE = 0.1            # matches Isaac MetaVoxelsLibrary wristSize

# optional drive to match the Isaac stiffness=1000 damping=0 angular drive.
# set to zero by default so the robot just falls under gravity like in the
# original example.
DEFAULT_TARGET_KE = 0.0
DEFAULT_TARGET_KD = 0.0

# color palette used when displaying each piece type
COLOR_CONTROL = wp.vec3(0x1c / 255.0, 0x5c / 255.0, 0x61 / 255.0)   # teal
COLOR_RIGID   = wp.vec3(0x02 / 255.0, 0x02 / 255.0, 0x27 / 255.0)   # dark blue
COLOR_JOINT_A = wp.vec3(0xfa / 255.0, 0x6e / 255.0, 0x70 / 255.0)   # peach
COLOR_JOINT_B = wp.vec3(0x38 / 255.0, 0x01 / 255.0, 0x52 / 255.0)   # purple
COLOR_END     = wp.vec3(0x38 / 255.0, 0x01 / 255.0, 0x52 / 255.0)   # purple

# six unit direction vectors (index 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z)
NEIGHBORS: Tuple[wp.vec3, ...] = (
    wp.vec3( 1.0,  0.0,  0.0),
    wp.vec3(-1.0,  0.0,  0.0),
    wp.vec3( 0.0,  1.0,  0.0),
    wp.vec3( 0.0, -1.0,  0.0),
    wp.vec3( 0.0,  0.0,  1.0),
    wp.vec3( 0.0,  0.0, -1.0),
)

# The Isaac pipeline sets the outer prim transform to a 4x4 that cyclically
# permutes axes, (x, y, z) -> (y, z, x) in USD row-vector convention. Applied
# as a column-vector rotation on points, that sends +x to +z, +y to +x,
# +z to +y. It's a 120 degree rotation around (-1, -1, -1) / sqrt(3),
# quaternion (-0.5, -0.5, -0.5, 0.5). (The sign-flipped variant
# (0.5, 0.5, 0.5, 0.5) is the INVERSE rotation and would swap the cycle
# direction; that was a bug in the first pass.)
_Q_COORD = wp.quat(-0.5, -0.5, -0.5, 0.5)


# --------------------------------------------------------------------------- #
# small math helpers
# --------------------------------------------------------------------------- #

def _euler_xyz_quat(rx_deg: float, ry_deg: float, rz_deg: float) -> wp.quat:
    """Quaternion equivalent of USD rotateXYZ(rx, ry, rz) in degrees.

    USD applies X first, then Y, then Z to a point. As a rotation acting on
    a column vector, that is R_z @ R_y @ R_x, i.e. the quaternion product
    q_z * q_y * q_x.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), rx)
    qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ry)
    qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rz)
    return qz * qy * qx


def _vec3_add(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec3_scale(a: wp.vec3, s: float) -> wp.vec3:
    return wp.vec3(a[0] * s, a[1] * s, a[2] * s)


def _vec3_neg(a: wp.vec3) -> wp.vec3:
    return wp.vec3(-a[0], -a[1], -a[2])


# --------------------------------------------------------------------------- #
# Isaac rot/shift tables for joint pieces
# --------------------------------------------------------------------------- #

def _joint_rot_shift(direction: int, orient: int, jtype: str,
                     size: float) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
    """Return (rot_xyz_deg, shift_xyz) for a joint piece.

    Ported verbatim from MetaVoxelsLibrary_isaac.Joint.__init__. The values
    live in the raw mesh frame (before the Isaac coordinate swap is
    applied); they get baked into the shape xform later.
    """
    rot = (0.0, 0.0, 0.0)
    shift = (0.0, 0.0, 0.0)

    # if orient == 0:
    #     rot = (0.0, 0.0, 90.0)

    # if direction in (2, 3):
    #     rot = (-90.0, 0.0, 0.0)
    #     if orient == 1:
    #         rot = (-90.0, 90.0, 0.0)
    #     shift = (0.0, -size * 0.5, size * 0.5)

    # if direction in (0, 1):
    #     rot = (0.0, 90.0, 0.0)
    #     if orient == 0:
    #         rot = (90.0, 0.0, 90.0)
    #     shift = (-size * 0.5, 0.0, size * 0.5)


    # if orient == 0:
    #     rot = (0.0, 0.0, 0.0)

    if direction == 0:
        if orient == 0:
            rot = (-90.0, 90.0, 0.0)
        elif orient == 1:
            rot = (-90.0, 0.0, 0.0)
        shift = (0.0, -size * 0.5, size * 0.5)

    if direction == 1:
        if orient == 0:
            rot = (-90.0, 90.0, 0.0)
        elif orient == 1:
            rot = (-90.0, 0.0, 0.0)
        shift = (0.0, -size * 0.5, size * 0.5)
    ############################################

    if direction == 2:
        if orient == 0:
            rot = (0.0, 0.0, 90.0)
        elif orient == 1:
            rot = (0.0, 0.0, 0.0)
        shift = (0.0, 0.0, 0.0)

    if direction == 3:
        if orient == 0:
            rot = (0.0, 0.0, 90.0)
        elif orient == 1:
            rot = (0.0, 0.0, 0.0)
        shift = (0.0, 0.0, 0)

    ############################################

    if direction == 4:
        if orient == 0:
            rot = (0.0, 90.0, 0.0)
        elif orient == 1:
            rot = (90.0, 0.0, 90.0)
        shift = (-size*0.5, 0.0, size*0.5)

    if direction == 5:
        if orient == 0:
            rot = (0.0, 90.0, 0.0)
        elif orient == 1:
            rot = (90.0, 0.0, 90.0)
        shift = (-size*0.5, 0.0, size*0.5)

    # if direction == 5:
    #     if orient == 0:
    #         rot = (0.0, -90.0, 0.0)
    #     elif orient == 1:
    #         rot = (0.0, -90.0, 0.0)
    #     shift = (size*0.5, 0.0, size*0.5)

    # if jtype == "wrist":
    #     if direction == 3:
    #         shift = (0.0, size * 0.5 - WRIST_SIZE, size * 0.5)
    #     elif direction == 1:
    #         shift = (size * 0.5 - WRIST_SIZE, 0.0, size * 0.5)
    #     elif direction == 5:
    #         shift = (0.0, 0.0, size - WRIST_SIZE)

    return rot, shift


def _end_rot_shift(direction: int,
                   size: float) -> Tuple[Tuple[float, float, float],
                                          Tuple[float, float, float]]:
    """Same for end caps, ported from End.__init__."""
    rot = (0.0, 0.0, 0.0)
    shift = (0.0, 0.0, 0.0)


    if direction == 0:
        rot = (-90.0, 0.0, 0.0)
        shift = (0.0, -size * 0.5, size * 0.5)
    elif direction == 1:
        rot = (90.0, 0.0, 0.0)
        shift = (0, size * 0.5, size * 0.5) 
    elif direction == 2:
        rot = (0.0, 0.0, 0.0)
        shift = (0.0, 0.0, 0.0)
    elif direction == 3:
        rot = (180.0, 0.0, 0.0)
        shift = (0.0, 0.0, size)
    elif direction == 4:
        rot = (0.0, 90.0, 0.0)
        shift = (-size*0.5, 0.0, size*0.5)
    elif direction == 5:
        rot = (0.0, -90.0, 0.0)
        shift = (size*0.5, 0.0, size*0.5)

    return rot, shift


def _shape_xform(rot_xyz_deg: Tuple[float, float, float],
                 shift: Tuple[float, float, float],
                 size: float) -> wp.transform:
    """Build the shape-local xform that places a piece's mesh correctly
    inside its body.

    The Isaac pipeline does T(pos) * S(size) * coord * T(shift) * R(rot)
    applied to a raw mesh vertex. With the body at T(pos) and the mesh
    scaled via the shape's `scale` parameter, the remaining shape-local
    transform is coord * T(size * shift) * R(rot). That simplifies to a
    rigid transform with:
        rotation    = q_coord * q_rot_xyz
        translation = q_coord rotate (size * shift)
    """
    q_rot = _euler_xyz_quat(*rot_xyz_deg)
    q_final = _Q_COORD * q_rot

    shift_scaled = wp.vec3(shift[0] * size, shift[1] * size, shift[2] * size)
    p_final = wp.quat_rotate(_Q_COORD, shift_scaled)

    return wp.transform(p=p_final, q=q_final)


# --------------------------------------------------------------------------- #
# USD mesh loading
# --------------------------------------------------------------------------- #

_MESH_CACHE: dict = {}


def _load_mesh(usd_path: str) -> Optional[newton.Mesh]:
    """Load every UsdGeom.Mesh inside a USD file, bake each prim's own
    xformable transform into its vertices so relative positions are kept,
    and return a single combined newton.Mesh. Cached by path.

    The loader does not perform an up-axis correction. If a USD in the set
    is authored as Y-up (the USD default) and you need Z-up visuals, run
    the `convert_usd_to_zup.py` helper once against the asset folder.
    """
    if not usd_path:
        return None
    cached = _MESH_CACHE.get(usd_path)
    if cached is not None:
        return cached
    if not _USD_AVAILABLE or not os.path.isfile(usd_path):
        _MESH_CACHE[usd_path] = None
        return None

    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception:
        _MESH_CACHE[usd_path] = None
        return None
    if stage is None:
        _MESH_CACHE[usd_path] = None
        return None

    all_v: List[np.ndarray] = []
    all_i: List[np.ndarray] = []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        try:
            m = newton.usd.get_mesh(prim)
        except Exception:
            continue

        v = np.asarray(m.vertices, dtype=np.float64)
        i = np.asarray(m.indices, dtype=np.int32).reshape(-1)

        xformable = UsdGeom.Xformable(prim)
        if xformable is not None:
            M = np.array(
                xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=np.float64,
            )
            if not np.allclose(M, np.eye(4)):
                # USD stores row-vector transforms, so v_world = v_local @ M
                ones = np.ones((len(v), 1))
                v = (np.hstack([v, ones]) @ M)[:, :3]

        all_v.append(v.astype(np.float32))
        all_i.append(i + offset)
        offset += len(v)

    if not all_v:
        _MESH_CACHE[usd_path] = None
        return None

    verts = np.vstack(all_v)
    idx = np.concatenate(all_i)

    mesh = newton.Mesh(
        verts, idx, compute_inertia=True, is_solid=True,
    )
    _MESH_CACHE[usd_path] = mesh
    return mesh


# --------------------------------------------------------------------------- #
# core classes
# --------------------------------------------------------------------------- #

class Robot:
    """One articulated robot composed of voxels, joints, and end caps.

    Maintains its own builder handle, the world-space origin, and the list
    of newton joint ids that make up the articulation.
    """

    def __init__(self, name: str, builder: newton.ModelBuilder,
                 pos: wp.vec3, size: float, cad: str,
                 floating: bool = True,
                 target_ke: float = DEFAULT_TARGET_KE,
                 target_kd: float = DEFAULT_TARGET_KD):
        self.name = name
        self.origin = pos
        self.size = size
        self.cad = cad
        self.builder = builder
        self.floating = floating
        self.target_ke = target_ke
        self.target_kd = target_kd

        self._name_counts: dict = {}
        self._joints: List[int] = []
        self._root_body: Optional[int] = None
        self._root_pos: Optional[wp.vec3] = None
        self._children: dict = {}
        self._finalized = False

    # ---- graph api (same shape as Isaac's authoring api) ---- #

    def addVoxel(self, link, vType: str) -> "Voxel":
        return Voxel(self, link, vType)

    def addJoint(self, link, jType: str, orient: int,
                 limits: Tuple[float, float]) -> "Joint":
        return Joint(self, link, jType, orient, limits)

    def addEnd(self, link) -> "End":
        return End(self, link)

    def fix(self, _piece) -> None:
        """Anchor the robot to the world. Must be called before finalize.

        Pieces argument is ignored; kept for backward compatibility with the
        Isaac authoring api where it took the control voxel. Newton does
        this via the root joint so we just flip the floating flag.
        """
        self.floating = False

    # ---- construction helpers used by Voxel/Joint/End ---- #

    def _unique_name(self, stem: str) -> str:
        n = self._name_counts.get(stem, 0)
        self._name_counts[stem] = n + 1
        return stem if n == 0 else f"{stem}{n}"

    def _make_body_and_shape(
        self,
        stem: str,
        world_pos: wp.vec3,
        mesh: Optional[newton.Mesh],
        rot_xyz_deg: Tuple[float, float, float],
        shift: Tuple[float, float, float],
        color: wp.vec3,
    ) -> int:
        """Create a rigid body at `world_pos` and attach one shape holding
        the mesh (or a size-fitting box if the mesh is missing).
        """
        label = self._unique_name(stem)

        body_id = self.builder.add_link(
            xform=wp.transform(p=world_pos, q=wp.quat_identity()),
            mass=0.0,           # computed from shape density
            label=label,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=DENSITY, is_solid=True,
        )

        if mesh is not None:
            shape_xf = _shape_xform(rot_xyz_deg, shift, self.size)
            self.builder.add_shape_mesh(
                body=body_id, mesh=mesh,
                xform=shape_xf,
                scale=wp.vec3(self.size, self.size, self.size),
                cfg=shape_cfg, color=color,
                label=label + "_shape",
            )
        else:
            # fallback: a voxel-sized box at the body center
            half = self.size * 0.5
            self.builder.add_shape_box(
                body=body_id,
                hx=half, hy=half, hz=half,
                cfg=shape_cfg, color=color,
                label=label + "_box",
            )

        return body_id

    def _add_fixed_to_world(self, child_body: int, at: wp.vec3) -> int:
        jid = self.builder.add_joint_fixed(
            parent=-1, child=child_body,
            parent_xform=wp.transform(p=at, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0),
                                     q=wp.quat_identity()),
            label=f"fixed_root_{child_body}",
        )
        self._joints.append(jid)
        return jid

    def _add_free_to_world(self, child_body: int) -> int:
        # For a free joint, Newton uses the child body's initial xform (set
        # by add_link) as the starting joint_q. Passing a non-identity
        # parent_xform here would stack on top of that and teleport the body
        # at the first eval_fk. Leave both anchors as identity.
        jid = self.builder.add_joint_free(
            parent=-1, child=child_body,
            parent_xform=wp.transform(),
            child_xform=wp.transform(),
            label=f"free_root_{child_body}",
        )
        self._joints.append(jid)
        return jid

    def _add_fixed_along(self, parent_body: int, child_body: int,
                         direction: int, label: str) -> int:
        """Anchor two bodies that sit `size` apart along `direction`.

        Anchor in the parent at its own center, in the child at the
        corresponding face. Matches Isaac voxel-side fixJoint exactly.
        """
        neigh = NEIGHBORS[direction]
        child_anchor = wp.vec3(
            -self.size * neigh[0],
            -self.size * neigh[1],
            -self.size * neigh[2],
        )
        jid = self.builder.add_joint_fixed(
            parent=parent_body, child=child_body,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0),
                                      q=wp.quat_identity()),
            child_xform=wp.transform(p=child_anchor,
                                     q=wp.quat_identity()),
            label=label,
        )
        self._joints.append(jid)
        return jid

    def _add_revolute_between(self, parent_body: int, child_body: int,
                              axis: wp.vec3,
                              limits_deg: Tuple[float, float],
                              label: str,
                              direction: int) -> int:
        # Both sub-bodies of a joint piece share the same world position and
        # the same (identity) world rotation, so the hinge passes through
        # whatever point in the body frame we pick as the anchor. Using the
        # body origin keeps the pivot at the mesh centroid regardless of
        # what the Isaac rot/shift table does with the visual mesh.
        if direction == 0:
            anchor = wp.vec3(0.0, 0.5*self.size, 0.0)
        elif direction == 1:
            anchor = wp.vec3(0.0, 0.5*self.size, 0.0)
        elif direction == 2:
            anchor = wp.vec3(0.0, 0.5*self.size, 0.0)
        elif direction == 3:
            anchor = wp.vec3(0.0, 0.5*self.size, 0.0)
        elif direction == 4:
            anchor = wp.vec3(0.0, 0.5*self.size, 0.0)
        elif direction == 5:
            anchor = wp.vec3(0.0, 0.5*self.size, 0.0)

        # Isaac's axis table ("X" / "Y" / "Z") is expressed in the body
        # frame AFTER the outer coord transform. Our Newton bodies don't
        # carry a coord rotation (we put that on the shape xform instead),
        # so the equivalent axis in Newton body frame is coord * axis.
        axis_newton = wp.quat_rotate(_Q_COORD, axis)

        jid = self.builder.add_joint_revolute(
            parent=parent_body, child=child_body,
            axis=axis_newton,
            parent_xform=wp.transform(p=anchor, q=wp.quat_identity()),
            child_xform=wp.transform(p=anchor, q=wp.quat_identity()),
            limit_lower=math.radians(float(limits_deg[0])),
            limit_upper=math.radians(float(limits_deg[1])),
            target_ke=self.target_ke,
            target_kd=self.target_kd,
            label=label,
        )
        self._joints.append(jid)
        return jid

    def _register_root(self, body_id: int, pos: wp.vec3) -> None:
        """Called by the first voxel added. The actual world-anchor joint is
        created later in finalize() so that r.fix() (which flips the
        floating flag after the root voxel has been added) still takes
        effect.
        """
        self._root_body = body_id
        self._root_pos = pos

    # ---- finalization ---- #

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        if self._root_body is None:
            # nothing to do
            return

        if self.floating:
            self._add_free_to_world(self._root_body)
        else:
            self._add_fixed_to_world(self._root_body, self._root_pos)

        self.builder.add_articulation(sorted(self._joints), label=self.name)

    def apply_colors(self, viewer) -> None:
        """Backward-compat no-op: colors are now set per-shape at build
        time via add_shape_*(color=...). Kept so existing examples that
        call robot.apply_colors(viewer) continue to work.
        """
        return


# --------------------------------------------------------------------------- #
# Voxel
# --------------------------------------------------------------------------- #

class Voxel:
    """A single cube-shaped body (control or rigid)."""

    def __init__(self, robot: Robot, link, vtype: str):
        self.robot = robot
        self.name = vtype
        self.vtype = vtype

        if link[0] is None:
            # root voxel
            self.pos = wp.vec3(0.0, 0.0, 0.0)
            self.loc = wp.vec3(0.0, 0.0, 0.0)
            direction = None
            parent_elem = None
        else:
            parent_elem = link[0]
            direction = link[1]
            self.pos = parent_elem.neighbor_pos(direction)
            self.loc = parent_elem.neighbor_loc(direction)

        self.size = robot.size
        self.direction_from_parent = direction
        self.parent_elem = parent_elem

        # build the body and the shape. voxels have no Isaac rot/shift.
        # control.usd in the MIT CBA asset set is very heavy; when it isn't
        # present, fall back to rigid.usd so the visual still renders.
        cad_path = os.path.join(robot.cad, vtype + ".usd")
        mesh = _load_mesh(cad_path)
        if mesh is None and vtype == "control":
            mesh = _load_mesh(os.path.join(robot.cad, "rigid.usd"))

        color = COLOR_CONTROL if vtype == "control" else COLOR_RIGID
        world_pos = _vec3_add(robot.origin, self.pos)
        self.body_id = robot._make_body_and_shape(
            stem=vtype, world_pos=world_pos, mesh=mesh,
            rot_xyz_deg=(0.0, 0.0, 0.0),
            shift=(0.0, 0.0, 0.0),
            color=color,
        )

        # anchor to world or to parent
        if parent_elem is None:
            robot._register_root(self.body_id, world_pos)
        else:
            parent_link = parent_elem.child_attach_link(direction)
            robot._add_fixed_along(
                parent_body=parent_link,
                child_body=self.body_id,
                direction=direction,
                label=f"fix_{self.name}_{self.body_id}",
            )

        robot._children[_loc_key(self.loc)] = ("v", self)

    # the neighbor_pos / neighbor_loc / child_attach_link helpers make Voxel,
    # Joint, and End interchangeable as graph nodes

    def neighbor_pos(self, direction: int) -> wp.vec3:
        n = NEIGHBORS[direction]
        return wp.vec3(
            self.pos[0] + self.size * n[0],
            self.pos[1] + self.size * n[1],
            self.pos[2] + self.size * n[2],
        )

    def neighbor_loc(self, direction: int) -> wp.vec3:
        n = NEIGHBORS[direction]
        return wp.vec3(
            self.loc[0] + n[0],
            self.loc[1] + n[1],
            self.loc[2] + n[2],
        )

    def child_attach_link(self, direction_to_child: int) -> int:
        """For voxels, every child attaches to the single body."""
        return self.body_id

    # old attribute name kept for compatibility with any external callers
    @property
    def link(self) -> int:
        return self.body_id


# --------------------------------------------------------------------------- #
# Joint (two bodies with a revolute hinge)
# --------------------------------------------------------------------------- #

class Joint:
    """A hinge piece with two sub-bodies connected by a revolute joint.

    Isaac represents the joint as two USD references at the same world
    position, plus one revolute constraint. That layout is preserved here
    so that the drive, the limits, and the Isaac fixJoint side-selection
    (i % 2 picks link1 vs link2) all translate directly.
    """

    def __init__(self, robot: Robot, link, jtype: str, orient: int,
                 limits: Tuple[float, float]):
        self.robot = robot
        self.name = jtype
        self.jtype = jtype
        self.orient = orient
        self.limits = limits

        parent_elem = link[0]
        direction = link[1]
        self.direction_from_parent = direction
        self.parent_elem = parent_elem

        self.pos = parent_elem.neighbor_pos(direction)
        self.loc = parent_elem.neighbor_loc(direction)
        self.size = robot.size

        # ---- decide which side of the hinge faces the parent ---- #
        # Isaac's fixJoint uses i % 2: odd i -> link1, even i -> link2,
        # where i is the direction from this joint to its neighbor. The
        # parent sits at direction (direction ^ 1) from this joint, so:
        #   direction even (placed at +x/+y/+z) -> parent attaches to link1
        #   direction odd  (placed at -x/-y/-z) -> parent attaches to link2
        parent_side_is_link1 = (direction % 2 == 0)

        # ---- build the two bodies with the Isaac rot/shift ---- #
        rot_xyz, shift = _joint_rot_shift(direction, orient, jtype, self.size)

        world_pos = _vec3_add(robot.origin, self.pos)
        cad1 = os.path.join(robot.cad, jtype + "_1.usd")
        cad2 = os.path.join(robot.cad, jtype + "_2.usd")
        mesh1 = _load_mesh(cad1)
        mesh2 = _load_mesh(cad2)

        self.link1 = robot._make_body_and_shape(
            stem=jtype, world_pos=world_pos, mesh=mesh1,
            rot_xyz_deg=rot_xyz, shift=shift,
            color=COLOR_JOINT_A,
        )
        self.link2 = robot._make_body_and_shape(
            stem=jtype + "_1", world_pos=world_pos, mesh=mesh2,
            rot_xyz_deg=rot_xyz, shift=shift,
            color=COLOR_JOINT_B,
        )

        # ---- wire in the articulation tree ---- #
        if parent_side_is_link1:
            self._parent_side = self.link1
            self._child_side = self.link2
        else:
            self._parent_side = self.link2
            self._child_side = self.link1

        parent_link = parent_elem.child_attach_link(direction)
        robot._add_fixed_along(
            parent_body=parent_link,
            child_body=self._parent_side,
            direction=direction,
            label=f"fix_{jtype}_in_{self.body_handle()}",
        )

        # revolute between the two sub-bodies. Axis follows Isaac's rule.
        axis = _joint_axis(jtype, direction, orient)
        robot._add_revolute_between(
            parent_body=self._parent_side,
            child_body=self._child_side,
            axis=axis, limits_deg=limits,
            label=f"rev_{jtype}_{self.body_handle()}",
            direction=direction,
        )

        robot._children[_loc_key(self.loc)] = ("j", self)

    def body_handle(self) -> int:
        return self.link1

    def neighbor_pos(self, direction: int) -> wp.vec3:
        n = NEIGHBORS[direction]
        return wp.vec3(
            self.pos[0] + self.size * n[0],
            self.pos[1] + self.size * n[1],
            self.pos[2] + self.size * n[2],
        )

    def neighbor_loc(self, direction: int) -> wp.vec3:
        n = NEIGHBORS[direction]
        return wp.vec3(
            self.loc[0] + n[0],
            self.loc[1] + n[1],
            self.loc[2] + n[2],
        )

    def child_attach_link(self, direction_to_child: int) -> int:
        """Isaac's rule: the side used for a given direction depends on
        parity (odd direction -> link1, even -> link2). Inside our strict
        articulation tree, however, every subsequent child must hang off
        the distal side of the hinge (the one NOT used for the parent
        attach), otherwise we'd create a cycle. Those two rules happen to
        agree for straight-line chains (ant arms, rover legs), which is
        everything the zoo currently builds.
        """
        return self._child_side

    @property
    def link(self) -> int:
        return self._child_side


def _joint_axis(jtype: str, direction: int, orient: int) -> wp.vec3:
    """Axis selection table, matching Isaac MetaVoxelsLibrary_isaac."""
    if jtype == "wrist":
        if direction in (0, 1):
            return wp.vec3(1.0, 0.0, 0.0)
        if direction in (2, 3):
            return wp.vec3(0.0, 1.0, 0.0)
        return wp.vec3(0.0, 0.0, 1.0)

    # elbow / compliant
    if direction in (0, 1):
        print(f"direction: {direction}, orient: {orient}")
        if orient == 0:
            return wp.vec3(1.0, 0.0, 0.0)  
        else:
            return wp.vec3(0.0, 0.0, 1.0)



    if direction in (2, 3):
        if orient == 0 :
            return wp.vec3(1.0, 0.0, 0.0) 
        else:
            return wp.vec3(0.0, 1.0, 0.0)

    if direction in (4, 5):
        if orient == 0:
            return wp.vec3(0.0, 1.0, 0.0)
        else:
            return wp.vec3(0.0, 0.0, 1.0)

            
    


# --------------------------------------------------------------------------- #
# End caps
# --------------------------------------------------------------------------- #

class End:
    """Foot or tip of a limb; same body model as a voxel but with the
    Isaac rot/shift for a capped mesh.
    """

    def __init__(self, robot: Robot, link):
        self.robot = robot
        self.name = "end"

        parent_elem = link[0]
        direction = link[1]
        self.direction_from_parent = direction
        self.parent_elem = parent_elem

        self.pos = parent_elem.neighbor_pos(direction)
        self.loc = parent_elem.neighbor_loc(direction)
        self.size = robot.size

        rot_xyz, shift = _end_rot_shift(direction, self.size)

        cad_path = os.path.join(robot.cad, "end.usd")
        mesh = _load_mesh(cad_path)

        world_pos = _vec3_add(robot.origin, self.pos)
        self.body_id = robot._make_body_and_shape(
            stem="end", world_pos=world_pos, mesh=mesh,
            rot_xyz_deg=rot_xyz, shift=shift,
            color=COLOR_END,
        )

        parent_link = parent_elem.child_attach_link(direction)
        robot._add_fixed_along(
            parent_body=parent_link,
            child_body=self.body_id,
            direction=direction,
            label=f"fix_end_{self.body_id}",
        )

        robot._children[_loc_key(self.loc)] = ("e", self)

    def neighbor_pos(self, direction: int) -> wp.vec3:
        n = NEIGHBORS[direction]
        return wp.vec3(
            self.pos[0] + self.size * n[0],
            self.pos[1] + self.size * n[1],
            self.pos[2] + self.size * n[2],
        )

    def neighbor_loc(self, direction: int) -> wp.vec3:
        n = NEIGHBORS[direction]
        return wp.vec3(
            self.loc[0] + n[0],
            self.loc[1] + n[1],
            self.loc[2] + n[2],
        )

    def child_attach_link(self, direction_to_child: int) -> int:
        return self.body_id

    @property
    def link(self) -> int:
        return self.body_id


def _loc_key(loc: wp.vec3) -> str:
    return f"[{int(loc[0])},{int(loc[1])},{int(loc[2])}]"


# --------------------------------------------------------------------------- #
# RobotZoo: same recipes the Isaac version shipped with
# --------------------------------------------------------------------------- #

class RobotZoo:
    def __init__(self, builder: newton.ModelBuilder, size: float,
                 usd_path: str, name: str = "metaVoxels"):
        self.builder = builder
        self.size = size
        self.usd_path = usd_path
        self.name = name

    # backward-compat aliases used by the old example_metavoxels.py
    @property
    def usdPath(self) -> str:
        return self.usd_path

    # ------------------------------------------------------------------ #
    def create_robot_trial(self, name: str, pos: wp.vec3) -> Robot:
        """Minimal sanity-check robot: control voxel + 1 rigid + 1 elbow."""
        r = Robot(name, self.builder, pos, self.size, self.usd_path)
        v = r.addVoxel(link=[None, 0], vType="control")
        v4 = r.addVoxel(link=[v, 4], vType="rigid")
        r.addJoint(link=[v4, 4], jType="elbow", orient=0, limits=(-40.0, 40.0))
        r.finalize()
        return r

    # ------------------------------------------------------------------ #
    def create_ant(self, name: str, pos: wp.vec3) -> Robot:
        """Four-legged ant. Port of createAnt from Isaac (limits -50..50
        on the hip, 30..100 on the knee, with signs flipped for -y/-x).
        """
        r = Robot(name, self.builder, pos, self.size, self.usd_path)
        v = r.addVoxel(link=[None, 0], vType="control")

        j2 = r.addJoint(link=[v, 0], jType="elbow", orient=0, limits=(-50, 50))
        j3 = r.addJoint(link=[v, 1], jType="elbow", orient=0, limits=(-50, 50))
        j4 = r.addJoint(link=[v, 2], jType="elbow", orient=0, limits=(-50, 50))
        j5 = r.addJoint(link=[v, 3], jType="elbow", orient=0, limits=(-50, 50))

        j6 = r.addJoint(link=[j2, 0], jType="elbow", orient=1, limits=(30, 100))
        j7 = r.addJoint(link=[j3, 1], jType="elbow", orient=1, limits=(30, 100))
        j8 = r.addJoint(link=[j4, 2], jType="elbow", orient=1, limits=(-100, -30))
        j9 = r.addJoint(link=[j5, 3], jType="elbow", orient=1, limits=(-100, -30))

        v4 = r.addVoxel(link=[j6, 0], vType="rigid")
        v5 = r.addVoxel(link=[j7, 1], vType="rigid")
        v6 = r.addVoxel(link=[j8, 2], vType="rigid")
        v7 = r.addVoxel(link=[j9, 3], vType="rigid")

        r.addEnd(link=[v4, 0])
        r.addEnd(link=[v5, 1])
        r.addEnd(link=[v6, 2])
        r.addEnd(link=[v7, 3])

        r.finalize()
        return r

    # ------------------------------------------------------------------ #
    def create_ant1(self, name: str, pos: wp.vec3) -> Robot:
        """Port of createAnt1 with wide-open limits."""
        r = Robot(name, self.builder, pos, self.size, self.usd_path)
        v = r.addVoxel(link=[None, 0], vType="control")

        j2 = r.addJoint(link=[v, 0], jType="elbow", orient=0, limits=(-80, 80))
        j3 = r.addJoint(link=[v, 1], jType="elbow", orient=0, limits=(-80, 80))
        j4 = r.addJoint(link=[v, 2], jType="elbow", orient=0, limits=(-80, 80))
        j5 = r.addJoint(link=[v, 3], jType="elbow", orient=0, limits=(-80, 80))
        # j6 = r.addJoint(link=[v, 4], jType="elbow", orient=0, limits=(-180, 180))
        # j7 = r.addJoint(link=[v, 5], jType="elbow", orient=0, limits=(-180, 180))

        j8 = r.addJoint(link=[j2, 0], jType="elbow", orient=1, limits=(-80, 80))
        j9 = r.addJoint(link=[j3, 1], jType="elbow", orient=1, limits=(-80, 80))
        j10 = r.addJoint(link=[j4, 2], jType="elbow", orient=1, limits=(-80, 80))
        j11 = r.addJoint(link=[j5, 3], jType="elbow", orient=1, limits=(-80, 80))
        # j12 = r.addJoint(link=[j6, 4], jType="elbow", orient=1, limits=(-180, 180))
        # j13 = r.addJoint(link=[j7, 5], jType="elbow", orient=1, limits=(-180, 180))


        v4 = r.addVoxel(link=[j8, 0], vType="rigid")
        v5 = r.addVoxel(link=[j9, 1], vType="rigid")
        v6 = r.addVoxel(link=[j10, 2], vType="rigid")
        v7 = r.addVoxel(link=[j11, 3], vType="rigid")
        # v8 = r.addVoxel(link=[j12, 4], vType="rigid")
        # v9 = r.addVoxel(link=[j13, 5], vType="rigid")

        r.addEnd(link=[v4, 0])
        r.addEnd(link=[v5, 1])
        r.addEnd(link=[v6, 2])
        r.addEnd(link=[v7, 3])
        # r.addEnd(link=[v8, 4])
        # r.addEnd(link=[v9, 5])

        # r.fix(v)

        r.finalize()

        return r

    # ------------------------------------------------------------------ #
    def create_billE(self, name: str, pos: wp.vec3) -> Robot:
        """Port of createBillE."""
        r = Robot(name, self.builder, pos, self.size, self.usd_path)
        v = r.addVoxel(link=[None, 0], vType="control")
        j1 = r.addJoint(link=[v, 4], jType="elbow", orient=0, limits=(-180, 180))
        v1 = r.addVoxel(link=[j1, 4], vType="rigid")
        v2 = r.addVoxel(link=[v1, 4], vType="rigid")
        j2 = r.addJoint(link=[v2, 0], jType="wrist", orient=0, limits=(-180, 180))
        v3 = r.addVoxel(link=[j2, 0], vType="rigid")
        v4 = r.addVoxel(link=[v3, 4], vType="rigid")
        j3 = r.addJoint(link=[v4, 4], jType="elbow", orient=0, limits=(-180, 180))
        r.addVoxel(link=[j3, 4], vType="rigid")
        r.fix(v)
        r.finalize()
        return r

    # ------------------------------------------------------------------ #
    def create_snake(self, name: str, pos: wp.vec3, n_segments: int = 6) -> Robot:
        """Simple chain of elbow joints alternating orient 0/1."""
        r = Robot(name, self.builder, pos, self.size, self.usd_path)
        cur = r.addVoxel(link=[None, 0], vType="control")
        for i in range(n_segments):
            j = r.addJoint(link=[cur, 0], jType="elbow",
                           orient=i % 2, limits=(-60, 60))
            cur = r.addVoxel(link=[j, 0], vType="rigid")
        r.finalize()
        return r

    # ------------------------------------------------------------------ #
    # Isaac-style snake-case aliases so existing scripts still work
    createRobotTrial = create_robot_trial
    createAnt = create_ant
    createAnt1 = create_ant1
    createBillE = create_billE
    createSnake = create_snake
