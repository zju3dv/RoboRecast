from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import mujoco
import numpy as np

from mobile_robot.control import planar_pose_step


ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
COMPILER_VERSION = "mobile-semantic-trajectory-v2.2.1"


@dataclass(frozen=True)
class CompiledMobileTrajectory:
    timestamp_sec: np.ndarray
    arm_targets: np.ndarray
    base_targets: np.ndarray
    base_nominal_xyyaw: np.ndarray
    phase_index: np.ndarray
    phase_scales: tuple[float, ...]
    max_ik_position_error_m: float
    max_ik_rotation_error_rad: float
    expansion_summary: tuple[dict, ...]


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"model name not found: {name}")
    return value


def _arm(model: mujoco.MjModel, side: str) -> tuple[list[int], list[int], int]:
    legacy = [f"{side}/{name}" for name in ARM_JOINTS]
    generic = [f"{side}/joint{index}" for index in range(1, 7)]
    names = legacy if mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, legacy[0]) >= 0 else generic
    joints = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    addresses = [int(model.jnt_qposadr[joint]) for joint in joints]
    dofs = [int(model.jnt_dofadr[joint]) for joint in joints]
    site = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/pinch")
    return addresses, dofs, site


def _normalize_quaternion(value: Sequence[float]) -> np.ndarray:
    quat = np.asarray(value, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError("quaternion_wxyz must contain four finite values")
    return quat / np.linalg.norm(quat)


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, quat_wxyz)
    return matrix.reshape(3, 3)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=float).reshape(9))
    return quat


def _slerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first, second = _normalize_quaternion(first), _normalize_quaternion(second)
    dot = float(np.dot(first, second))
    if dot < 0:
        second, dot = -second, -dot
    dot = float(np.clip(dot, -1, 1))
    if dot > .9995:
        return _normalize_quaternion(first + alpha * (second - first))
    angle = math.acos(dot)
    return (math.sin((1 - alpha) * angle) * first + math.sin(alpha * angle) * second) / math.sin(angle)


def _smooth(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0, 1))
    return alpha ** 3 * (10 - 15 * alpha + 6 * alpha ** 2)


def _site_quaternion(data: mujoco.MjData, site: int) -> np.ndarray:
    return _matrix_to_quat(data.site_xmat[site].reshape(3, 3))


def _resolve_target(model: mujoco.MjModel, data: mujoco.MjData,
                    value: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray | None]:
    kind = value.get("type")
    if kind == "world_pose":
        position = np.asarray(value.get("position_m"), dtype=float)
        rotation = np.eye(3)
    elif kind in ("body_relative", "contact_frame"):
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, str(value.get("body")))
        rotation = data.xmat[body].reshape(3, 3)
        offset = value.get("offset_m", value.get("point_m"))
        position = data.xpos[body] + rotation @ np.asarray(offset, dtype=float)
        if kind == "contact_frame" and "approach_offset_m" in value:
            normal = np.asarray(value.get("normal_m", [0, 0, 1]), dtype=float)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                raise ValueError("contact_frame normal_m must be nonzero")
            position += rotation @ (normal / norm * float(value["approach_offset_m"]))
    else:
        raise ValueError("waypoint target.type must be world_pose, body_relative, or contact_frame")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("waypoint target position must contain three finite values")
    quaternion = value.get("quaternion_wxyz")
    if quaternion is None:
        return position, None
    local = _quat_to_matrix(_normalize_quaternion(quaternion))
    return position, _matrix_to_quat(rotation @ local if kind != "world_pose" else local)


def _geom_frame(model: mujoco.MjModel, data: mujoco.MjData,
                name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a named geom's world center, rotation, and conservative half extents."""
    geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    center = data.geom_xpos[geom].copy()
    rotation = data.geom_xmat[geom].reshape(3, 3).copy()
    size = model.geom_size[geom].copy()
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        half = size
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.repeat(size[0], 3)
    elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = np.asarray([size[0], size[0], size[1]])
    else:
        half = np.repeat(float(model.geom_rbound[geom]), 3)
    world_half = np.abs(rotation) @ half
    return center, rotation, world_half


def _body_bounds(model: mujoco.MjModel, data: mujoco.MjData,
                 body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for geom in range(model.ngeom):
        ancestor = int(model.geom_bodyid[geom])
        while ancestor > 0 and ancestor != body:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != body:
            continue
        center = data.geom_xpos[geom]
        radius = float(model.geom_rbound[geom])
        lower = np.minimum(lower, center - radius)
        upper = np.maximum(upper, center + radius)
    if not np.isfinite(lower).all():
        raise ValueError(f"body has no collision geometry: {body_name}")
    return lower, upper


def _waypoint(position: np.ndarray, quaternion: object | None, at: float,
              opening: float, event: str) -> dict:
    target = {"type": "world_pose", "position_m": np.asarray(position).tolist()}
    if quaternion is not None:
        target["quaternion_wxyz"] = _normalize_quaternion(quaternion).tolist()
    return {"at": float(at), "event": event, "target": target,
            "gripper_opening": float(opening)}


def _expand_grasp(model: mujoco.MjModel, data: mujoco.MjData,
                  phase: dict, motion: Mapping[str, object]) -> tuple[dict, list[dict]]:
    default_geom_name = str(motion.get("grasp_geom", ""))
    arms = motion.get("arms", {})
    if not isinstance(arms, Mapping) or not arms:
        raise ValueError(f"phase={phase['name']} grasp motion needs arms")
    result = {key: value for key, value in phase.items() if key != "motion"}
    generated = []
    for side, default_lateral in (("left", .04), ("right", -.04)):
        # Explicit Cartesian waypoints always win over macro expansion.
        if phase.get(side, {}).get("mode") == "waypoints":
            continue
        spec = arms.get(side)
        if spec is None:
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"phase={phase['name']} motion.arms.{side} must be an object")
        # Bilateral objects often expose one named handle per hand. Resolve each
        # arm from its own geom while keeping the concise shared form available.
        geom_name = str(spec.get("grasp_geom", default_geom_name))
        if not geom_name:
            raise ValueError(
                f"phase={phase['name']} {side} grasp needs grasp_geom on the arm or motion")
        center, rotation, _ = _geom_frame(model, data, geom_name)
        local_point = np.asarray(spec.get("contact_point_m", [0, default_lateral, 0]), dtype=float)
        approach = np.asarray(spec.get("approach_axis", motion.get("approach_axis", [1, 0, 0])), dtype=float)
        if local_point.shape != (3,) or approach.shape != (3,) or np.linalg.norm(approach) < 1e-8:
            raise ValueError(f"phase={phase['name']} {side} grasp vectors must be finite 3-vectors")
        geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            half = np.asarray(model.geom_size[geom_id], dtype=float)
            # A small allowance lets the pinch site sit just outside a surface,
            # while rejecting points that target empty space beyond the named
            # handle and only happen to create incidental link contact.
            if np.any(np.abs(local_point) > half + .025):
                raise ValueError(
                    f"phase={phase['name']} {side} contact_point_m lies outside "
                    f"box grasp_geom={geom_name}; half_size_m={half.tolist()}")
        approach_world = rotation @ (approach / np.linalg.norm(approach))
        contact = center + rotation @ local_point
        distance = float(spec.get("pregrasp_distance_m", motion.get("pregrasp_distance_m", .08)))
        quaternion = spec.get("quaternion_wxyz", motion.get("quaternion_wxyz"))
        opening = float(spec.get("open_gripper", motion.get("open_gripper", 1.0)))
        closed = float(spec.get("closed_gripper", motion.get("closed_gripper", .0)))
        waypoints = [
            _waypoint(contact - approach_world * distance, quaternion, .55, opening, "pregrasp"),
            _waypoint(contact, quaternion, .82, opening, "contact"),
            _waypoint(contact, quaternion, 1.0, closed, "close"),
        ]
        result[side] = {"mode": "waypoints", "waypoints": waypoints}
        generated.append({"side": side, "events": ["pregrasp", "contact", "close"],
                          "grasp_geom": geom_name})
    if not generated and not any(phase.get(side, {}).get("mode") == "waypoints"
                                 for side in ("left", "right")):
        raise ValueError(f"phase={phase['name']} grasp motion selected no arm")
    return result, generated


def _expand_carry(model: mujoco.MjModel, data: mujoco.MjData,
                  phase: dict, motion: Mapping[str, object]) -> tuple[dict, list[dict]]:
    object_name = str(motion.get("object", phase.get("object", "")))
    destination = motion.get("destination", {})
    if not object_name or not isinstance(destination, Mapping) or not destination.get("support_geom"):
        raise ValueError(f"phase={phase['name']} carry needs object and destination.support_geom")
    object_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    object_center = data.xpos[object_body].copy()
    object_rotation = data.xmat[object_body].reshape(3, 3).copy()
    object_lower, _ = _body_bounds(model, data, object_name)
    support_center, _, support_half = _geom_frame(
        model, data, str(destination["support_geom"]))
    target_center = object_center.copy()
    target_center[:2] = support_center[:2] + np.asarray(
        destination.get("offset_xy_m", [0, 0]), dtype=float)
    bottom_from_center = float(object_center[2] - object_lower[2])
    target_center[2] = (support_center[2] + support_half[2] + bottom_from_center +
                        float(destination.get("clearance_m", .005)))
    route = motion.get("route", {})
    if not isinstance(route, Mapping):
        raise ValueError(f"phase={phase['name']} carry route must be an object")
    lift = max(float(route.get("lift_clearance_m", .12)), .01)
    above = max(float(route.get("destination_clearance_m", lift)), .01)
    object_targets: list[tuple[str, np.ndarray]] = [
        ("vertical_lift", object_center + [0, 0, lift])]
    for index, via in enumerate(route.get("via", [])):
        position, _ = _resolve_target(model, data, via)
        object_targets.append((f"via_{index + 1}", position))
    object_targets.extend([
        ("above_support", target_center + [0, 0, above]),
        ("lower", target_center),
    ])
    result = {key: value for key, value in phase.items() if key != "motion"}
    result["object"] = object_name
    arms = motion.get("arms", {})
    generated = []
    selected = list(phase.get("grasp_sides", [])) or list(arms)
    if not selected:
        raise ValueError(f"phase={phase['name']} carry needs grasp_sides or motion.arms")
    for side in selected:
        if side not in ("left", "right"):
            raise ValueError(f"phase={phase['name']} carry arm must be left or right")
        if phase.get(side, {}).get("mode") == "waypoints":
            continue
        spec = arms.get(side, {})
        if not isinstance(spec, Mapping):
            raise ValueError(f"phase={phase['name']} motion.arms.{side} must be an object")
        _, _, site = _arm(model, side)
        local_hand = np.asarray(spec.get(
            "object_offset_m", object_rotation.T @ (data.site_xpos[site] - object_center)), dtype=float)
        orientation_mode = str(spec.get(
            "orientation_mode", motion.get("orientation_mode", "preserve")))
        if orientation_mode not in ("preserve", "preserve_lift", "free"):
            raise ValueError(
                f"phase={phase['name']} {side} carry orientation_mode must be "
                "preserve, preserve_lift, or free")
        quaternion = spec.get("quaternion_wxyz", _site_quaternion(data, site))
        opening = float(spec.get("gripper_opening", 0.0))
        count = len(object_targets)
        waypoints = [
            _waypoint(center + object_rotation @ local_hand,
                      quaternion if orientation_mode == "preserve" or (
                          orientation_mode == "preserve_lift" and event == "vertical_lift")
                      else None,
                      (index + 1) / count, opening, event)
            for index, (event, center) in enumerate(object_targets)
        ]
        result[side] = {"mode": "waypoints", "waypoints": waypoints,
                        "orientation_mode": orientation_mode}
        generated.append({"side": side, "events": [event for event, _ in object_targets],
                          "support_geom": destination["support_geom"]})
    return result, generated


def _expand_motion(model: mujoco.MjModel, data: mujoco.MjData,
                   phase: dict) -> tuple[dict, list[dict]]:
    motion = phase.get("motion")
    if motion is None:
        return phase, []
    if not isinstance(motion, Mapping):
        raise ValueError(f"phase={phase['name']} motion must be an object")
    kind = motion.get("type")
    if kind == "grasp":
        return _expand_grasp(model, data, phase, motion)
    if kind == "carry":
        return _expand_carry(model, data, phase, motion)
    raise ValueError(f"phase={phase['name']} motion.type must be grasp or carry")


def _orientation_error(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    # World-frame shortest rotation vector. The former sin(theta) error
    # vanished at pi and incorrectly accepted opposite gripper orientations.
    current = np.asarray(current_wxyz, dtype=float)
    target = np.asarray(target_wxyz, dtype=float)
    current = current / np.linalg.norm(current)
    target = target / np.linalg.norm(target)
    inverse = current * np.array([1., -1., -1., -1.])
    relative = np.empty(4)
    mujoco.mju_mulQuat(relative, target, inverse)
    if relative[0] < 0:
        relative = -relative
    sine = float(np.linalg.norm(relative[1:]))
    if sine < 1e-12:
        return 2 * relative[1:]
    angle = 2 * math.atan2(sine, max(0.0, float(relative[0])))
    return relative[1:] * (angle / sine)


def _solve(model: mujoco.MjModel, data: mujoco.MjData, side: str,
           target_position: np.ndarray, target_quaternion: np.ndarray | None,
           position_tolerance: float = .02, rotation_tolerance: float = .15,
           max_step_from_reference: float | None = .10,
           posture_reference: np.ndarray | None = None
           ) -> tuple[np.ndarray, float, float]:
    addresses, dofs, site = _arm(model, side)
    # Resolve limits from the actual DOFs selected by `_arm`. v2 bundles use
    # embodiment-specific names such as joint1..joint6, while Mobile ALOHA
    # keeps the legacy VX300S names.
    joints = [int(model.dof_jntid[dof]) for dof in dofs]
    # Keep each dense solve on the branch selected by the preceding row. Near a
    # wrist singularity, pose-only DLS admits equivalent forearm/wrist-roll
    # solutions and used to jump by 0.2-0.5 rad in one 20 ms sample.
    reference = data.qpos[addresses].copy()
    preferred = reference if posture_reference is None else posture_reference
    rotation_error = 0.0
    for _ in range(180):
        mujoco.mj_forward(model, data)
        position_error = target_position - data.site_xpos[site]
        if target_quaternion is None:
            error = position_error
            rotation_vector = np.zeros(3)
        else:
            rotation_vector = _orientation_error(_site_quaternion(data, site), target_quaternion)
            error = np.concatenate((position_error, .35 * rotation_vector))
        if (np.linalg.norm(position_error) <= .0015 and
                (target_quaternion is None or np.linalg.norm(rotation_vector) <= .012)):
            break
        jac_position = np.zeros((3, model.nv), dtype=float)
        jac_rotation = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jac_position, jac_rotation, site)
        jacobian = jac_position[:, dofs]
        if target_quaternion is not None:
            jacobian = np.vstack((jacobian, .35 * jac_rotation[:, dofs]))
        # A light posture term selects the preceding-row branch without
        # preventing deliberate free-space wrist rotation.
        regularization = .0001
        current = data.qpos[addresses].copy()
        delta = np.linalg.solve(
            jacobian.T @ jacobian + (.001 + regularization) * np.eye(len(addresses)),
            jacobian.T @ error + regularization * (preferred - current))
        for column, (address, joint) in enumerate(zip(addresses, joints)):
            low, high = model.jnt_range[joint]
            candidate = data.qpos[address] + np.clip(delta[column], -.08, .08)
            if max_step_from_reference is not None:
                candidate = np.clip(candidate,
                                    reference[column] - max_step_from_reference,
                                    reference[column] + max_step_from_reference)
            data.qpos[address] = np.clip(candidate, low, high)
    mujoco.mj_forward(model, data)
    position_error = float(np.linalg.norm(target_position - data.site_xpos[site]))
    if target_quaternion is not None:
        rotation_error = float(np.linalg.norm(
            _orientation_error(_site_quaternion(data, site), target_quaternion)))
    # Report inaccurate solutions at the first affected segment. Physical
    # probes still decide collision, grasp and placement quality.
    if position_error > position_tolerance or rotation_error > rotation_tolerance:
        raise ValueError(
            f"{side} waypoint unreachable: position_error_m={position_error:.4f}, "
            f"rotation_error_rad={rotation_error:.4f}")
    return data.qpos[addresses].copy(), position_error, rotation_error


def _base_commands(source_base: np.ndarray, phases: list[dict],
                   initial_xyyaw: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    commands = np.zeros_like(source_base, dtype=float)
    for phase in phases:
        start, end = int(phase["start_row"]), int(phase["end_row"])
        base = phase.get("base", {"mode": "source"})
        mode = base.get("mode", "source")
        if mode == "source":
            scale = np.asarray(base.get("scale", [1, 1, 1]), dtype=float)
            bias = np.asarray(base.get("bias", [0, 0, 0]), dtype=float)
            if scale.shape == (2,):
                scale = np.asarray([scale[0], 1, scale[1]])
            if bias.shape == (2,):
                bias = np.asarray([bias[0], 0, bias[1]])
            commands[start:end] = source_base[start:end] * scale + bias
        elif mode == "constant":
            commands[start:end] = [float(base.get("vx_mps", base.get("v_mps", 0))),
                                   float(base.get("vy_mps", 0)),
                                   float(base.get("yaw_rate_radps", base.get("omega_radps", 0)))]
        elif mode == "stop":
            commands[start:end] = 0
        else:
            raise ValueError(f"unknown base mode: {mode}")
        if phase.get("action") == "operate_arm":
            commands[max(0, start - 1):end] = 0
    if np.any(np.abs(commands[:, 1]) > 1e-9):
        raise ValueError("Tracer accepts forward speed and yaw rate only")
    pose = np.zeros((len(commands), 3), dtype=float)
    pose[0] = initial_xyyaw
    for row in range(1, len(commands)):
        vx, vy, yaw_rate = commands[row - 1]
        pose[row] = planar_pose_step(pose[row - 1], vx, yaw_rate,
                                     float(timestamps[row] - timestamps[row - 1]), vy)
    return commands, pose


def _validate_phases(phases: list[dict], rows: int) -> None:
    if not phases or phases[0].get("start_row") != 0 or phases[-1].get("end_row") != rows:
        raise ValueError("phases must cover the complete source row range")
    previous = 0
    names = set()
    for phase in phases:
        start, end = int(phase["start_row"]), int(phase["end_row"])
        if start != previous or end <= start or end > rows:
            raise ValueError("phases must be contiguous, increasing, and in bounds")
        previous = end
        name = phase.get("name")
        if not name or name in names:
            raise ValueError("phase names must be present and unique")
        names.add(name)
        for side in ("left", "right"):
            arm = phase.get(side, {"mode": "hold_current"})
            mode = arm.get("mode", "hold_current")
            if mode in ("joint", "source_endpoints") or "target" in arm and mode == "joint":
                raise ValueError(f"{side} arm joint targets are not accepted; use Cartesian waypoints")
            if mode not in ("hold", "hold_current", "waypoints"):
                raise ValueError(f"{side} mode must be hold_current or waypoints")
            if mode == "waypoints" and not arm.get("waypoints"):
                raise ValueError(f"{side} waypoints mode requires waypoints")


def compile_semantic_plan(model: mujoco.MjModel, settled: Mapping[str, np.ndarray],
                          source: Mapping[str, np.ndarray], phases: list[dict],
                          runtime: Mapping[str, object],
                          phase_start_overrides: Mapping[str, Mapping[str, np.ndarray]] | None = None
                          ) -> CompiledMobileTrajectory:
    source_qpos = np.asarray(source["qpos"], dtype=float)
    source_base = np.asarray(source["base_action"], dtype=float)
    if source_base.shape == (len(source_qpos), 2):
        source_base = np.column_stack((source_base[:, 0], np.zeros(len(source_base)),
                                       source_base[:, 1]))
    if source_base.shape != (len(source_qpos), 3):
        raise ValueError("source base command must be [T,3] body twist")
    source_timestamps = np.asarray(source.get(
        "timestamp_sec", np.arange(len(source_qpos), dtype=float) * .02), dtype=float)
    rows = len(source_qpos)
    _validate_phases(phases, rows)
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(settled["qpos"], dtype=float)
    data.qvel[:] = np.asarray(settled["qvel"], dtype=float)
    data.ctrl[:] = np.asarray(settled["ctrl"], dtype=float)
    mujoco.mj_forward(model, data)
    base_addresses = [int(model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                      for name in runtime.get("base_pose_joints",
                                              [model.joint(i).name for i in range(3)])]
    base_targets, base_pose = _base_commands(
        source_base, phases, data.qpos[base_addresses].copy(), source_timestamps)
    targets = np.zeros((rows, 14), dtype=float)
    phase_index = np.zeros(rows, dtype=np.int32)
    timestamps = source_timestamps.copy() - source_timestamps[0]
    max_position_error = max_rotation_error = 0.0
    phase_scales: list[float] = []
    expansion_summary: list[dict] = []
    current_gripper = {side: float(np.clip(source_qpos[0, offset + 6], 0, 1))
                       for side, offset in (("left", 0), ("right", 7))}
    phase_start_overrides = phase_start_overrides or {}

    for index, source_phase in enumerate(phases):
        phase = source_phase
        start, end = int(phase["start_row"]), int(phase["end_row"])
        phase_index[start:end] = index
        override = phase_start_overrides.get(phase["name"])
        if override is not None:
            data.qpos[:] = np.asarray(override["qpos"], dtype=float)
            data.qvel[:] = np.asarray(override["qvel"], dtype=float)
            data.ctrl[:] = np.asarray(override["ctrl"], dtype=float)
            base_pose[start] = data.qpos[base_addresses]
            for base_row in range(start + 1, len(base_pose)):
                vx, vy, yaw_rate = base_targets[base_row - 1]
                base_pose[base_row] = planar_pose_step(
                    base_pose[base_row - 1], vx, yaw_rate,
                    float(source_timestamps[base_row] - source_timestamps[base_row - 1]), vy)
        else:
            data.qpos[base_addresses] = base_pose[start]
        mujoco.mj_forward(model, data)
        phase, generated = _expand_motion(model, data, phase)
        if generated:
            expansion_summary.append({"phase": phase["name"],
                                      "motion_type": source_phase["motion"]["type"],
                                      "generated": generated})
        phase_start_qpos = data.qpos.copy()
        for side, offset in (("left", 0), ("right", 7)):
            # Compile both arms from the same phase-start state. The first arm
            # must not leave the second arm at the phase-end base pose.
            data.qpos[:] = phase_start_qpos
            mujoco.mj_forward(model, data)
            addresses, _, site = _arm(model, side)
            initial_joint = data.qpos[addresses].copy()
            arm = phase.get(side, {"mode": "hold_current"})
            if arm.get("mode", "hold_current") in ("hold", "hold_current"):
                if override is not None or start == 0:
                    names = runtime.get("arms", {}).get(side, {}).get("actuator_names")
                    if names is None:
                        names = [model.joint(int(np.flatnonzero(model.jnt_qposadr == address)[0])).name
                                 for address in addresses]
                    held = data.ctrl[[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                      for name in names]].copy()
                else:
                    held = targets[start - 1, offset:offset + 6].copy()
                # Preserve servo effort at a loaded checkpoint. Re-targeting
                # measured (deflected) qpos would cause a second gravity sag.
                targets[start:end, offset:offset + 6] = held
                targets[start:end, offset + 6] = current_gripper[side]
                continue
            sparse = [{"at": 0.0, "position": data.site_xpos[site].copy(),
                       "quaternion": _site_quaternion(data, site),
                       "gripper": current_gripper[side]}]
            carry_orientation = (sparse[0]["quaternion"].copy()
                                 if phase.get("intent") == "supported_carry" and
                                 side in phase.get("grasp_sides", []) and
                                 arm.get("orientation_mode", "preserve") == "preserve"
                                 else None)
            path_mode = arm.get("path_mode", "joint" if phase.get("intent") in
                                (None, "release", "free_motion", "withdraw") and
                                phase.get("action") == "operate_arm" else "cartesian")
            if path_mode not in ("joint", "cartesian"):
                raise ValueError("path_mode must be joint or cartesian")
            preserve_orientation = arm.get("orientation_mode", "preserve") != "free"
            last_at = 0.0
            for waypoint_index, waypoint in enumerate(arm["waypoints"]):
                at = float(waypoint.get("at", (waypoint_index + 1) / len(arm["waypoints"])))
                if not last_at < at <= 1:
                    raise ValueError(f"{side} waypoint at values must increase in (0,1]")
                target_spec = waypoint["target"]
                source_posture = None
                if target_spec.get("type") == "source_pose":
                    source_row = int(target_spec.get("row", round(
                        start + at * max(0, end - start - 1))))
                    if not 0 <= source_row < rows:
                        raise ValueError("source_pose row is outside the source trajectory")
                    sample_name = str(target_spec.get("sample", "action"))
                    if sample_name not in ("action", "qpos"):
                        raise ValueError("source_pose sample must be action or qpos")
                    saved_qpos = data.qpos.copy()
                    data.qpos[base_addresses] = base_pose[source_row]
                    sample = np.asarray(source[sample_name][source_row], dtype=float)
                    for sample_side, sample_offset in (("left", 0), ("right", 7)):
                        sample_addresses, _, _ = _arm(model, sample_side)
                        data.qpos[sample_addresses] = sample[sample_offset:sample_offset + 6]
                    mujoco.mj_forward(model, data)
                    position = data.site_xpos[site].copy()
                    quaternion = _site_quaternion(data, site)
                    source_posture = data.qpos[addresses].copy()
                    data.qpos[:] = saved_qpos
                    mujoco.mj_forward(model, data)
                else:
                    position, quaternion = _resolve_target(model, data, target_spec)
                    if quaternion is None and preserve_orientation:
                        quaternion = (carry_orientation.copy() if carry_orientation is not None
                                      else sparse[-1]["quaternion"])
                sparse.append({"at": at, "position": position,
                               "quaternion": quaternion, "posture": source_posture,
                               "gripper": float(waypoint.get(
                                   "gripper_opening", current_gripper[side]))})
                last_at = at
            if last_at < 1:
                sparse.append({**sparse[-1], "at": 1.0})
            endpoint_joints = [initial_joint.copy()]
            if path_mode == "joint":
                for waypoint_number, endpoint in enumerate(sparse[1:], 1):
                    target_row = start + round(endpoint["at"] * (end - start - 1))
                    data.qpos[base_addresses] = base_pose[target_row]
                    try:
                        joints, pe, re = _solve(
                            model, data, side, endpoint["position"], endpoint["quaternion"],
                            max_step_from_reference=None,
                            posture_reference=endpoint.get("posture"))
                    except ValueError as error:
                        raise ValueError(f"phase={phase['name']} side={side} waypoint={waypoint_number}: {error}") from error
                    endpoint_joints.append(joints)
                    max_position_error = max(max_position_error, pe)
                    max_rotation_error = max(max_rotation_error, re)
                data.qpos[addresses] = initial_joint
            for row in range(start, end):
                data.qpos[base_addresses] = base_pose[row]
                fraction = (row - start) / max(1, end - start - 1)
                segment = min(len(sparse) - 2, max(0, next(
                    (i for i in range(len(sparse) - 1) if fraction <= sparse[i + 1]["at"]),
                    len(sparse) - 2)))
                first, second = sparse[segment], sparse[segment + 1]
                local = _smooth((fraction - first["at"]) / max(1e-9, second["at"] - first["at"]))
                position = first["position"] + local * (second["position"] - first["position"])
                if second["quaternion"] is None:
                    quaternion = None
                else:
                    first_quaternion = (first["quaternion"] if first["quaternion"] is not None
                                        else _site_quaternion(data, site))
                    quaternion = _slerp(first_quaternion, second["quaternion"], local)
                try:
                    if path_mode == "joint":
                        joints = ((1 - local) * endpoint_joints[segment] +
                                  local * endpoint_joints[segment + 1])
                        data.qpos[addresses] = joints
                        position_error = rotation_error = 0.0
                    else:
                        joints, position_error, rotation_error = _solve(
                            model, data, side, position, quaternion,
                            posture_reference=second.get("posture"))
                except ValueError as error:
                    raise ValueError(
                        f"phase={phase['name']} side={side} row={row} "
                        f"waypoint_segment={segment}: {error}") from error
                targets[row, offset:offset + 6] = joints
                targets[row, offset + 6] = first["gripper"] + local * (
                    second["gripper"] - first["gripper"])
                max_position_error = max(max_position_error, position_error)
                max_rotation_error = max(max_rotation_error, rotation_error)
            current_gripper[side] = float(targets[end - 1, offset + 6])

        duration = max(float(np.median(np.diff(source_timestamps))),
                       float(source_timestamps[end - 1] - source_timestamps[start]))
        scale = 1.0
        has_arm_motion = any(
            phase.get(side, {}).get("mode", "hold_current") == "waypoints"
            for side in ("left", "right"))
        if end - start > 2 and has_arm_motion:
            boundary = max(0, start - 1)
            joint_path = np.concatenate((targets[boundary:end, :6], targets[boundary:end, 7:13]), axis=1)
            phase_dt = np.diff(source_timestamps[boundary:end])
            velocity = np.diff(joint_path, axis=0) / phase_dt[:, None]
            velocity_ratio = float(np.max(np.abs(velocity)) / float(runtime.get(
                "compiler_joint_velocity_limit_radps", 1.0)))
            acceleration_dt = np.maximum(phase_dt[1:], 1e-6)
            acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
            acceleration_ratio = float(np.max(np.abs(acceleration), initial=0.0) / float(runtime.get(
                "compiler_joint_acceleration_limit_radps2", 4.0)))
            scale = max(1.0, velocity_ratio, math.sqrt(acceleration_ratio))
            if scale > 1:
                original = timestamps[start:end].copy()
                anchor = timestamps[start - 1] if start else timestamps[start]
                timestamps[start:end] = anchor + (original - anchor) * scale
                extra = float(timestamps[end - 1] - original[-1])
                timestamps[end:] += extra
                # Preserve the compiled planar path when a moving-base phase is
                # lengthened for arm limits. The rollout integrates velocity
                # over the scaled row duration.
                # Scale interval commands once after all phase timing is known.
        phase_scales.append(scale)
        data.qpos[base_addresses] = base_pose[end - 1]
        for side, offset in (("left", 0), ("right", 7)):
            addresses, _, _ = _arm(model, side)
            data.qpos[addresses] = targets[end - 1, offset:offset + 6]
        mujoco.mj_forward(model, data)

    base_targets[:-1] *= (np.diff(source_timestamps) / np.diff(timestamps))[:, None]
    return CompiledMobileTrajectory(
        timestamps, targets, base_targets, base_pose, phase_index,
        tuple(phase_scales), max_position_error, max_rotation_error,
        tuple(expansion_summary))
