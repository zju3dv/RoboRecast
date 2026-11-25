from __future__ import annotations
import math
from typing import Any, Dict, Tuple, List
import numpy as np
import mujoco
from .errors import StructuredToolError
from .geometry_predicates import mesh_support_gap, point_inside_geom, support_surface
from .robot_runtime_view import RobotRuntimeView
from .gripper_control import width_to_control

def model_signature(model):
    import hashlib
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()

def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str, field: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise StructuredToolError("SUCCESS_NAME_MISSING", "Success predicate references an absent MJCF name.",
                                  field, "A destination name in simulation/name_map.json.", name, None)
    return value

def _contact_count(model: mujoco.MjModel, data: mujoco.MjData,
                   pair: Tuple[int, int]) -> int:
    return sum(1 for contact in data.contact[:data.ncon]
               if {int(contact.geom1), int(contact.geom2)} == set(pair))

def _resolve_contact_groups(model, pattern_a, pattern_b):
    """Resolve stable geom names/globs once, preserving unknown empty groups."""
    import fnmatch
    names = [(index, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index))
             for index in range(model.ngeom)]
    groups = tuple(frozenset(index for index, name in names if name is not None and fnmatch.fnmatchcase(name, pattern))
                   for pattern in (pattern_a, pattern_b))
    return groups if all(groups) else None

def _group_contact_count(data, groups):
    left, right = groups
    return sum(1 for contact in data.contact[:data.ncon]
               if (int(contact.geom1) in left and int(contact.geom2) in right) or
                  (int(contact.geom2) in left and int(contact.geom1) in right))

def _geom_extent_in_frame(model: mujoco.MjModel, data: mujoco.MjData,
                          geom_id: int, frame_world_to_local: np.ndarray) -> np.ndarray:
    """Return the geom's projected half extent in an arbitrary local frame."""
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        local_extent = np.repeat(float(size[0]), 3)
    elif geom_type in {int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)}:
        local_extent = np.asarray(size[:3], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        local_extent = np.asarray([size[0], size[0], size[1]], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        local_extent = np.asarray([size[0], size[0], size[0] + size[1]], dtype=np.float64)
    else:
        return np.repeat(float(model.geom_rbound[geom_id]), 3)
    geom_local_to_world = data.geom_xmat[geom_id].reshape(3, 3)
    relative = frame_world_to_local.dot(geom_local_to_world)
    return np.abs(relative).dot(local_extent)

def _geom_bounds_in_frame(model, data, geom_id, frame_world_to_local, origin):
    """Exact mesh vertex bounds; avoid treating long thin meshes as spheres."""
    center = frame_world_to_local.dot(data.geom_xpos[geom_id] - origin)
    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh])
        vertices = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh])]
        relative = frame_world_to_local.dot(data.geom_xmat[geom_id].reshape(3, 3))
        points = vertices.dot(relative.T) + center
        return points.min(axis=0), points.max(axis=0)
    extent = _geom_extent_in_frame(model, data, geom_id, frame_world_to_local)
    return center - extent, center + extent

def _evaluate_success(model: mujoco.MjModel, dense: Dict[str, np.ndarray],
                      success_spec: Dict[str, Any], body_history: Dict[str, np.ndarray],
                      contact_history: Dict[Tuple[str, str], np.ndarray],
                      max_penetration: float,
                      penetration_detail: Dict[str, Any] = None) -> Dict[str, Any]:
    results = []
    required_count = len(success_spec["required_checks"])
    for check_index, row in enumerate(success_spec["checks"]):
        check_type = row["type"]
        if check_type == "finite_state":
            value = bool(np.all(np.isfinite(dense["qpos"])) and np.all(np.isfinite(dense["qvel"])))
            result = {"type": check_type, "value": value, "passed": value}
        elif check_type == "maximum_penetration":
            result = {"type": check_type, "value_m": max_penetration,
                      "limit_m": float(row["max_m"]), "passed": max_penetration <= float(row["max_m"])}
            if penetration_detail:
                result.update(penetration_detail)
        elif check_type == "body_displacement":
            positions = body_history[row["body"]]
            value = float(np.linalg.norm(positions[-1] - positions[0]))
            result = {"type": check_type, "body": row["body"], "value_m": value,
                      "range_m": [float(row["min_m"]), float(row["max_m"])],
                      "passed": float(row["min_m"]) <= value <= float(row["max_m"])}
        elif check_type == "joint_final_range":
            joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, row["joint"], "$.checks[*].joint")
            address = int(model.jnt_qposadr[joint_id])
            value = float(dense["qpos"][-1, address])
            result = {"type": check_type, "joint": row["joint"], "value": value,
                      "range": [float(row["min"]), float(row["max"])],
                      "passed": float(row["min"]) <= value <= float(row["max"])}
        elif check_type in {"body_inside_geom", "body_count_inside_geom", "body_count_inside_volume"}:
            bodies = [row["body"]] if check_type == "body_inside_geom" else row["bodies"]
            geom_name = row.get("geom") or row.get("volume")
            geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name,
                          "$.checks[*].{}".format("geom" if row.get("geom") else "volume"))
            replay = mujoco.MjData(model)
            replay.qpos[:] = dense["qpos"][-1]
            mujoco.mj_forward(model, replay)
            membership = {}
            positions = {}
            predicates = {}
            for body in bodies:
                body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, body, "$.checks[*].body")
                predicate = point_inside_geom(model, replay, geom_id, replay.xpos[body_id], row["margin_m"])
                predicates[body] = predicate
                membership[body] = predicate["inside"]
                positions[body] = predicate["local_position_m"]
            if check_type == "body_inside_geom":
                result = {"type": check_type, "body": row["body"], "geom": row["geom"],
                          "local_position_m": positions[row["body"]],
                          "shape": predicates[row["body"]]["shape"],
                          "dimensions": predicates[row["body"]]["dimensions"],
                          "confidence": predicates[row["body"]]["confidence"],
                          "warning": predicates[row["body"]]["warning"],
                          "passed": membership[row["body"]]}
            else:
                inside = [body for body in bodies if membership[body]]
                result = {"type": check_type, "bodies": bodies,
                          "geom" if row.get("geom") else "volume": geom_name,
                          "inside_bodies": inside, "outside_bodies": [body for body in bodies if body not in inside],
                          "inside_count": len(inside), "minimum": int(row["min_count"]),
                          "local_positions_m": positions,
                          "shape": predicates[bodies[0]]["shape"],
                          "dimensions": predicates[bodies[0]]["dimensions"],
                          "confidence": predicates[bodies[0]]["confidence"],
                          "warning": predicates[bodies[0]]["warning"],
                          "passed": len(inside) >= int(row["min_count"])}
        elif check_type == "body_supported_by_geom":
            body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, row["body"], "$.checks[*].body")
            geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, row["geom"], "$.checks[*].geom")
            replay = mujoco.MjData(model); replay.qpos[:] = dense["qpos"][-1]; mujoco.mj_forward(model, replay)
            rotation = replay.geom_xmat[geom_id].reshape(3, 3).T
            members = []
            for candidate in range(model.ngeom):
                ancestor = int(model.geom_bodyid[candidate])
                while ancestor > 0 and ancestor != body_id:
                    ancestor = int(model.body_parentid[ancestor])
                if ancestor == body_id:
                    members.append(candidate)
            surface = support_surface(model, replay, geom_id)
            probe = {}
            if surface["shape"] == "mesh":
                horizontal, gap, probe = mesh_support_gap(model, replay, geom_id, members,
                                                          float(row["horizontal_margin_m"]))
            else:
                bounds = [_geom_bounds_in_frame(model, replay, candidate, rotation, replay.geom_xpos[geom_id])
                          for candidate in members]
                if not bounds:
                    point = rotation.dot(replay.xpos[body_id] - replay.geom_xpos[geom_id])
                    bounds = [(point, point)]
                local_centers = [(lower + upper) / 2 for lower, upper in bounds]
                extents = [(upper - lower) / 2 for lower, upper in bounds]
                half_xy = np.asarray(surface["half_extents_xy_m"], dtype=np.float64)
                horizontal = any(abs(center[0]) <= half_xy[0] + extents[index][0] + float(row["horizontal_margin_m"]) and
                                 abs(center[1]) <= half_xy[1] + extents[index][1] + float(row["horizontal_margin_m"])
                                 for index, center in enumerate(local_centers))
                bottom = min(center[2] - extents[index][2] for index, center in enumerate(local_centers))
                gap = float(bottom - surface["top_height_m"])
            contact_count = int(np.count_nonzero(contact_history.get(("support:" + row["body"], row["geom"]), [])))
            minimum = int(row.get("min_contact_samples", 0))
            result = {"type": check_type, "body": row["body"], "geom": row["geom"],
                      "horizontal_overlap": bool(horizontal), "vertical_gap_m": gap,
                      "contact_sample_count": contact_count, "minimum_contact_samples": minimum,
                      **probe,
                      "support_shape": surface["shape"],
                      "support_confidence": surface["confidence"],
                      "support_warning": surface["warning"],
                      "passed": bool(horizontal and abs(gap) <= float(row["max_vertical_gap_m"]) and
                                     contact_count >= minimum)}
        else:
            values = contact_history[(row["geom_a"], row["geom_b"])]
            count = int(np.count_nonzero(values > 0))
            result = {"type": check_type, "geom_a": row["geom_a"], "geom_b": row["geom_b"],
                      "sample_count": count, "minimum": int(row["min_samples"]),
                      "passed": count >= int(row["min_samples"])}
        result["role"] = "required" if check_index < required_count else "diagnostic"
        results.append(result)
    return {"schema_version": "real2sim-success-metrics-v1",
            "status": "PASS" if all(row["passed"] for row in results[:required_count]) else "FAIL",
            "checks": results}

def _required_observables(model: mujoco.MjModel, success_spec: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[Tuple[str, str], Tuple[int, int]]]:
    bodies, contacts = {}, {}
    for row in success_spec["checks"]:
        if row["type"] in {"body_displacement", "body_inside_geom", "body_supported_by_geom"}:
            bodies[row["body"]] = _id(model, mujoco.mjtObj.mjOBJ_BODY, row["body"], "$.checks[*].body")
        if row["type"] in {"body_count_inside_geom", "body_count_inside_volume"}:
            for body in row["bodies"]:
                bodies[body] = _id(model, mujoco.mjtObj.mjOBJ_BODY, body, "$.checks[*].bodies")
        if row["type"] == "contact_samples":
            key = (row["geom_a"], row["geom_b"])
            contacts[key] = (_id(model, mujoco.mjtObj.mjOBJ_GEOM, key[0], "$.checks[*].geom_a"),
                             _id(model, mujoco.mjtObj.mjOBJ_GEOM, key[1], "$.checks[*].geom_b"))
        if row["type"] == "body_supported_by_geom":
            body_id = bodies[row["body"]]
            support_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, row["geom"], "$.checks[*].geom")
            member_geoms = set()
            for geom in range(model.ngeom):
                ancestor = int(model.geom_bodyid[geom])
                while ancestor > 0 and ancestor != body_id:
                    ancestor = int(model.body_parentid[ancestor])
                if ancestor == body_id:
                    member_geoms.add(geom)
            contacts[("support:" + row["body"], row["geom"])] = (support_id, member_geoms)
    return bodies, contacts

def _execute(model: mujoco.MjModel, runtime: Dict[str, Any], expanded: ExpandedTrajectory,
             success_spec: Dict[str, Any], requirements=None, resume=None, contact_policy=None, additional_body_positions=()) -> Tuple[Dict[str, np.ndarray], Dict[str, Any],
                                                     Dict[str, np.ndarray]]:
    data = mujoco.MjData(model)
    settled_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "settled_initial")
    if settled_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, settled_id)
    runtime_view = RobotRuntimeView.from_contract(model, runtime)
    bodies, contacts = _required_observables(model, success_spec)
    logs: Dict[str, List[Any]] = {name: [] for name in (
        "timestamp_sec", "qpos", "qvel", "ctrl", "arm_joint_position",
        "arm_joint_velocity", "arm_joint_target", "ee_position_m", "ee_quaternion_wxyz")}
    if contact_policy is not None:
        logs['unexpected_robot_contact'] = []
    body_logs = {name: [] for name in bodies}
    contact_logs = {name: [] for name in contacts}
    requirement_observables = {}
    if requirements is not None:
        predicates = [p for e in requirements['I'] for p in e['predicates']] + requirements['c'] + requirements['G']
        for predicate in predicates:
            key = predicate['key']
            parts = key.split(':')
            if len(parts) == 3 and parts[0] == 'contact':
                groups = _resolve_contact_groups(model, parts[1], parts[2])
                if groups is not None:
                    requirement_observables[key] = ('contact', groups)
            elif len(parts) == 3 and parts[0] == 'body' and parts[2] == 'position':
                index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parts[1])
                if index >= 0:
                    requirement_observables[key] = ('body', index)
            elif len(parts) == 2 and parts[0] == 'joint':
                index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, parts[1])
                if index >= 0 and int(model.jnt_type[index]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    requirement_observables[key] = ('joint', int(model.jnt_qposadr[index]))
        for name in additional_body_positions:
            index = _id(model, mujoco.mjtObj.mjOBJ_BODY, name, '$.tail_speed_observables')
            requirement_observables[f'body:{name}:position'] = ('body', index)
        for key in requirement_observables:
            logs[key] = []
        logs['integration_state'] = []
        logs['penetration_m'] = []
        logs['contacted_geom_bits'] = []
        from roborecast.checkpoints import supports_local_friction
        touched = np.full(model.ngeom, not supports_local_friction(model), dtype=np.uint8)
    max_penetration = 0.0
    penetration_detail = None
    final_time = float(expanded.timestamp_sec[-1])
    steps = int(math.ceil(final_time / model.opt.timestep)) + 1
    target_velocity = (np.gradient(expanded.joint_position, expanded.timestamp_sec, axis=0)
                       if runtime.get("velocity_feedforward") else None)
    start_step = 0
    if resume is not None and additional_body_positions:
        with np.load(resume['evidence_path'], allow_pickle=False) as prefix:
            if any(f'body:{name}:position' not in prefix for name in additional_body_positions):
                # A new derived metric cannot reuse a prefix that never measured it.
                resume = None
    if resume is not None:
        from roborecast.checkpoints import restore
        restore(resume['path'], model, data, resume['identity'],
                local_friction=bool(resume.get('changed_static_friction_geoms')))
        with np.load(resume['evidence_path'], allow_pickle=False) as prefix:
            count = resume['index'] + 1
            for key in logs:
                if key == 'contacted_geom_bits' and key not in prefix:
                    # Historical checkpoints have no solver-contact ledger.
                    logs[key] = [np.packbits(np.ones(model.ngeom, dtype=np.uint8))] * count
                else:
                    logs[key] = list(prefix[key][:count])
            if requirements is not None:
                touched[:] = np.unpackbits(logs['contacted_geom_bits'][-1])[:model.ngeom]
            for key in body_logs:
                body_logs[key] = list(prefix['legacy_body:' + key][:count])
            for key in contact_logs:
                contact_logs[key] = list(prefix['legacy_contact:' + ':'.join(key)][:count])
            max_penetration = float(np.max(prefix['penetration_m'][:count]))
        start_step = count
    for step in range(start_step, steps):
        time_value = min(step * model.opt.timestep, final_time)
        target = np.asarray([np.interp(time_value, expanded.timestamp_sec, expanded.joint_position[:, column])
                             for column in range(runtime_view.arm_dof)])
        width = float(np.interp(time_value, expanded.timestamp_sec, expanded.gripper_width_m))
        driver = width_to_control(runtime["gripper"], width)
        data.ctrl[list(runtime_view.position_actuator_ids)] = target
        data.ctrl[list(runtime_view.velocity_actuator_ids)] = (
            [np.interp(time_value, expanded.timestamp_sec, target_velocity[:, j])
             for j in range(runtime_view.arm_dof)] if target_velocity is not None else 0.0)
        data.ctrl[runtime_view.gripper_actuator_id] = driver
        mujoco.mj_step(model, data)
        used_contact = (-1, -1)
        if contact_policy is not None:
            from roborecast.motion_validity import unexpected_contact
            used_contact = unexpected_contact(data, contact_policy, time_value)
        if requirements is not None:
            # Euler/implicit integrators leave the contacts used by this step in MjData.
            # Also cover the post-step forward pass used for logging/warm start.
            touched[data.contact.geom[:data.ncon].ravel()] = 1
            mujoco.mj_forward(model, data)
            touched[data.contact.geom[:data.ncon].ravel()] = 1
            logs['contacted_geom_bits'].append(np.packbits(touched))
            from roborecast.checkpoints import capture
            logs['integration_state'].append(capture(model, data))
            for key, (kind, index) in requirement_observables.items():
                value = (_group_contact_count(data, index) if kind == 'contact' else
                         data.xpos[index].copy() if kind == 'body' else float(data.qpos[index]))
                logs[key].append(value)
        if contact_policy is not None:
            from roborecast.motion_validity import unexpected_contact
            logs['unexpected_robot_contact'].append(used_contact if used_contact[0] >= 0 else
                unexpected_contact(data, contact_policy, float(data.time)))
        logs["timestamp_sec"].append(float(data.time))
        logs["qpos"].append(data.qpos.copy())
        logs["qvel"].append(data.qvel.copy())
        logs["ctrl"].append(data.ctrl.copy())
        logs["arm_joint_position"].append(data.qpos[list(runtime_view.arm_qpos_addresses)].copy())
        logs["arm_joint_velocity"].append(data.qvel[list(runtime_view.arm_dof_addresses)].copy())
        logs["arm_joint_target"].append(target)
        logs["ee_position_m"].append(data.xpos[runtime_view.end_effector_body_id].copy())
        logs["ee_quaternion_wxyz"].append(data.xquat[runtime_view.end_effector_body_id].copy())
        for name, body_id in bodies.items():
            body_logs[name].append(data.xpos[body_id].copy())
        for name, pair in contacts.items():
            if isinstance(pair[1], set):
                contact_logs[name].append(sum(1 for contact in data.contact[:data.ncon]
                                              if (int(contact.geom1) == pair[0] and int(contact.geom2) in pair[1]) or
                                              (int(contact.geom2) == pair[0] and int(contact.geom1) in pair[1])))
            else:
                contact_logs[name].append(_contact_count(model, data, pair))
        penetrating = [row for row in data.contact[:data.ncon] if row.dist < 0]
        if requirements is not None:
            logs['penetration_m'].append(max((-float(row.dist) for row in penetrating), default=0.))
        if penetrating:
            deepest = min(penetrating, key=lambda row: row.dist)
            value = -float(deepest.dist)
            if value > max_penetration:
                max_penetration = value
                penetration_detail = {
                    "time_sec": float(data.time),
                    "geom_a": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, deepest.geom1),
                    "geom_b": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, deepest.geom2),
                }
    dense = {name: np.asarray(value) for name, value in logs.items()}
    body_history = {name: np.asarray(value) for name, value in body_logs.items()}
    contact_history = {name: np.asarray(value) for name, value in contact_logs.items()}
    if requirements is not None:
        dense.update({'legacy_body:' + key: value for key, value in body_history.items()})
        dense.update({'legacy_contact:' + ':'.join(key): value for key, value in contact_history.items()})
        dense['executed_step_count'] = np.asarray(steps - start_step)
        dense['reused_step_count'] = np.asarray(start_step)
    success = _evaluate_success(model, dense, success_spec, body_history, contact_history,
                                max_penetration, penetration_detail)
    return dense, success, body_history
