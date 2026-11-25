from __future__ import annotations

from typing import Any, Dict

import mujoco
import numpy as np


def support_surface(model: mujoco.MjModel, data: mujoco.MjData,
                    geom_id: int) -> Dict[str, Any]:
    """Return a common local footprint/top representation for support geoms."""
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    geom_type = int(model.geom_type[geom_id])
    confidence, warning = "high", None
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        half_xy = np.asarray([np.inf if size[0] <= 0 else size[0],
                              np.inf if size[1] <= 0 else size[1]])
        top, shape = 0.0, "plane"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        half_xy, top, shape = size[:2], float(size[2]), "box"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half_xy, top, shape = np.repeat(float(size[0]), 2), float(size[1]), "cylinder"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half_xy, top, shape = np.repeat(float(size[0]), 2), float(size[0]), "sphere"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        half_xy, top, shape = size[:2], float(size[2]), "ellipsoid"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_xy = np.repeat(float(size[0]), 2)
        top, shape = float(size[0] + size[1]), "capsule"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        return {"shape": "mesh", "half_extents_xy_m": None, "top_height_m": None,
                "confidence": "high", "warning": None}
    else:
        radius = float(model.geom_rbound[geom_id])
        half_xy, top, shape = np.repeat(radius, 2), radius, "conservative_bound"
        confidence = "low"
        warning = "Unsupported support geom uses its conservative bounding sphere."
    return {"shape": shape, "half_extents_xy_m": half_xy.tolist(),
            "top_height_m": top, "confidence": confidence, "warning": warning}


def point_inside_geom(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int,
                      point_world: np.ndarray, margin_m: float = 0.0) -> Dict[str, Any]:
    """Evaluate primitive membership in the geom's local frame."""
    rotation_world_to_local = data.geom_xmat[geom_id].reshape(3, 3).T
    local = rotation_world_to_local.dot(
        np.asarray(point_world, dtype=np.float64) - data.geom_xpos[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    margin = max(0.0, float(margin_m))
    geom_type = int(model.geom_type[geom_id])
    confidence = "high"
    warning = None
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        dimensions = {"half_extents_m": size[:3].tolist()}
        inside = bool(np.all(np.abs(local) <= size[:3] + margin))
        shape = "box"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(size[0])
        dimensions = {"radius_m": radius}
        inside = bool(np.linalg.norm(local) <= radius + margin)
        shape = "sphere"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius, half_height = float(size[0]), float(size[1])
        dimensions = {"radius_m": radius, "half_height_m": half_height}
        inside = bool(np.linalg.norm(local[:2]) <= radius + margin and
                      abs(float(local[2])) <= half_height + margin)
        shape = "cylinder"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        radii = size[:3]
        dimensions = {"radii_m": radii.tolist()}
        inside = bool(np.sum(np.square(local / (radii + margin))) <= 1.0)
        shape = "ellipsoid"
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half_length = float(size[0]), float(size[1])
        closest_z = float(np.clip(local[2], -half_length, half_length))
        distance = np.linalg.norm(local - np.asarray([0.0, 0.0, closest_z]))
        dimensions = {"radius_m": radius, "segment_half_length_m": half_length}
        inside = bool(distance <= radius + margin)
        shape = "capsule"
    else:
        radius = float(model.geom_rbound[geom_id])
        dimensions = {"conservative_radius_m": radius}
        inside = bool(np.linalg.norm(local) <= radius + margin)
        shape = "conservative_bound"
        confidence = "low"
        warning = "Unsupported geom type uses a conservative bounding sphere."
    return {"inside": inside, "local_position_m": local.tolist(), "shape": shape,
            "dimensions": dimensions, "margin_m": margin,
            "confidence": confidence, "warning": warning}


def mesh_support_gap(model, data, support_id, members, horizontal_margin_m):
    """Measure vertical clearance on the named collision mesh in the gravity frame."""
    gravity = np.asarray(model.opt.gravity)
    up = -gravity / np.linalg.norm(gravity) if np.linalg.norm(gravity) else np.array([0., 0., 1.])
    samples = []
    for member in members:
        if not (model.geom_contype[member] or model.geom_conaffinity[member]):
            continue
        witnesses = []
        for contact in data.contact[:data.ncon]:
            if {int(contact.geom1), int(contact.geom2)} == {support_id, member}:
                witnesses.append((contact.pos, contact.pos, 'final_contact'))
        segment = np.empty(6)
        # The bounding radii only bound the ray's origin / search distance.
        # They are never used as positions of physical surfaces.
        distance = np.linalg.norm(data.geom_xpos[member]-data.geom_xpos[support_id])
        reach = distance + model.geom_rbound[support_id] + model.geom_rbound[member] + 1.
        measured = mujoco.mj_geomDistance(model, data, support_id, member, reach, segment)
        if 0 <= measured < reach:
            delta = segment[3:] - segment[:3]
            if np.linalg.norm(delta-up*(delta@up)) <= horizontal_margin_m + 1e-6:
                witnesses.append((segment[:3], segment[3:], 'nearest_surface_pair'))
        center = data.geom_xpos[member]
        witnesses.append((center, center, 'body_geom_center'))
        for on_support, on_body, source in witnesses:
            # Cast from above the entire table and from below the entire object.
            # A side contact or a body beneath the tabletop cannot pass as support.
            start = on_support + up * reach
            normal = np.empty(3)
            top = mujoco.mj_rayMesh(model, data, support_id, start, -up, normal)
            if top < 0 or normal@up <= 1e-5:
                continue
            below = on_body - up * reach
            kind = int(model.geom_type[member])
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                bottom = mujoco.mj_rayMesh(model, data, member, below, up)
            else:
                bottom = mujoco.mju_rayGeom(data.geom_xpos[member], data.geom_xmat[member],
                                            model.geom_size[member], below, up, kind)
            if bottom < 0:
                continue
            surface = start-up*top
            gap = float((below+up*bottom-surface)@up)
            samples.append((gap, source, surface.tolist(), model.geom(member).name))
    # Minimum signed clearance catches gross penetration even if another part rests correctly.
    if not samples:
        return False, None, {'support_probe': 'no_common_gravity_ray'}
    gap, source, surface, member_name = min(samples, key=lambda row: row[0])
    return True, gap, {'support_probe': source, 'support_point_world_m': surface,
                       'supported_collision_geom': member_name, 'surface_ray_pairs': len(samples)}
