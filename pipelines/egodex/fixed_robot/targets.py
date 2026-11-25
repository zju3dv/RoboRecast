"""Explicit target frames resolved against the committed initial simulation state.

No body target follows a moving object: compilation freezes every target before IK.
A later object location must be supplied as an explicit world target by the caller.
"""
import numpy as np
import mujoco
from .robot_runtime import runtime_contract
from scipy.spatial.transform import Rotation


def vector(value, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain three finite values")
    return result


def rotation(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError("quaternion_wxyz must contain four finite values with nonzero norm")
    return Rotation.from_quat(q[[1, 2, 3, 0]] / np.linalg.norm(q)).as_matrix()


def named_id(model, kind, name):
    if not isinstance(name, str) or not name:
        raise ValueError("reference frame/object requires a nonempty name")
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"unknown reference frame/object: {name}")
    return index


def resolve_target(model, data, value):
    if not isinstance(value, dict):
        raise ValueError("arm target must be an object")
    kind = value.get("type", "world_pose")
    origin, basis = np.zeros(3), np.eye(3)
    if kind == "world_pose":
        point = vector(value.get("position_m", value.get("position")), "world position")
    elif kind == "body_relative":
        index = named_id(model, mujoco.mjtObj.mjOBJ_BODY, value.get("body"))
        origin, basis = data.xpos[index], data.xmat[index].reshape(3, 3)
        point = vector(value.get("offset_m", value.get("position")), "body offset_m")
    elif kind == "contact_frame":
        refs = [key for key in ("body", "geom", "site") if key in value]
        if len(refs) != 1:
            raise ValueError("contact_frame requires exactly one named body, geom, or site")
        ref = refs[0]
        enum = {"body": mujoco.mjtObj.mjOBJ_BODY, "geom": mujoco.mjtObj.mjOBJ_GEOM, "site": mujoco.mjtObj.mjOBJ_SITE}[ref]
        index = named_id(model, enum, value[ref])
        positions = {"body": data.xpos, "geom": data.geom_xpos, "site": data.site_xpos}
        matrices = {"body": data.xmat, "geom": data.geom_xmat, "site": data.site_xmat}
        origin, basis = positions[ref][index].copy(), matrices[ref][index].reshape(3, 3).copy()
        origin += basis @ vector(value.get("frame_position_m", [0, 0, 0]), "frame_position_m")
        basis = basis @ rotation(value.get("frame_quaternion_wxyz", [1, 0, 0, 0]))
        point = vector(value.get("point_m", value.get("offset_m", [0, 0, 0])), "contact point_m").copy()
        if "approach_offset_m" in value:
            normal = vector(value.get("normal_m", [0, 0, 1]), "normal_m")
            distance = float(value["approach_offset_m"])
            if np.linalg.norm(normal) < 1e-10 or not np.isfinite(distance):
                raise ValueError("contact normal must be nonzero and approach offset finite")
            point += normal / np.linalg.norm(normal) * distance
    else:
        raise ValueError("target.type must be world_pose, body_relative, or contact_frame")
    result = {"position": (origin + basis @ point).tolist()}
    if "quaternion_wxyz" in value:
        quat = Rotation.from_matrix(basis @ rotation(value["quaternion_wxyz"])).as_quat()
        result["quaternion_wxyz"] = quat[[3, 0, 1, 2]].tolist()
    return result


def validate_hand_objects(model, phase):
    bindings = phase.get("hand_objects")
    if not isinstance(bindings, dict) or set(bindings) != {"left", "right"}:
        raise ValueError("each phase requires hand_objects for both left and right; roles text alone is insufficient")
    for side, item in bindings.items():
        if not isinstance(item, dict) or item.get("mode") not in ("idle", "grasp", "hold", "contact", "release"):
            raise ValueError(f"hand_objects.{side}.mode must be idle/grasp/hold/contact/release")
        if item["mode"] == "idle":
            if item.get("object") is not None:
                raise ValueError("idle hand must have object null or omitted")
        else:
            index = named_id(model, mujoco.mjtObj.mjOBJ_BODY, item.get("object"))
            robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime_contract(model)["base"]["body"])
            ancestor = index
            while ancestor > 0 and ancestor != robot_root:
                ancestor = int(model.body_parentid[ancestor])
            if index == 0 or (robot_root >= 0 and ancestor == robot_root):
                raise ValueError("hand binding must name a task body outside the robot subtree")
        for name in ("max_contact_gap_sec", "max_slip_m", "max_rotation_slip_deg", "max_tilt_deg", "carry_height_m"):
            if name in item and (not np.isfinite(float(item[name])) or float(item[name]) < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        policy=item.get("hold_policy","legacy_endpoint")
        if policy not in ("legacy_endpoint","gap_slip"):
            raise ValueError("hold_policy must be legacy_endpoint or gap_slip")
        if "hold_policy" in item and item["mode"]!="hold":
            raise ValueError("hold_policy applies only to hold bindings")
        if policy=="gap_slip" and not all(k in item for k in ("max_contact_gap_sec","max_slip_m","max_rotation_slip_deg")):
            raise ValueError("gap_slip requires explicit max_contact_gap_sec, max_slip_m, max_rotation_slip_deg")
        if "min_contact_fraction" in item:
            fraction = float(item["min_contact_fraction"])
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("min_contact_fraction must be finite and in [0,1]")
    return bindings
