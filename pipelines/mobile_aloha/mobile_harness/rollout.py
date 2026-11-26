from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from mobile_input.writer import sha256, write_json
from mobile_input.loader import load_episode
from mobile_robot.control import step_mobile
from .evidence import render_rollout_sheet, RolloutSheetRenderer
from .goals import TERMINAL_TYPES, evaluate_terminal_goals
from .stages import (contact_side, full_task_object_quality, object_geom, object_tilt_deg,
                     relative_pinch, restore, snapshot)
from .semantic_trajectory import COMPILER_VERSION
from .store import Store
from .bindings import current_scene
from .hand_objects import HandObjectMonitor, intended_contact
from .compatibility import execution_hash, longest_prefix


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"model name not found: {name}")
    return value


def _geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        return name
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return f"{body_name}/geom_{geom_id}"


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _evaluate(model: mujoco.MjModel, record: dict, checks: list[dict],
              qpos: np.ndarray, qvel: np.ndarray, body: np.ndarray,
              route: np.ndarray, contacts: dict[tuple[str, str, str], int], phase_end: dict) -> list[dict]:
    results = []
    path_length = float(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum())
    for check in checks:
        kind = check["type"]
        if kind in TERMINAL_TYPES:
            continue
        observed = None
        if kind == "base_path_length":
            observed = path_length
            passed = observed >= float(check["min_m"])
        elif kind == "base_stop":
            state = phase_end[check["phase"]]
            observed = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(check["xy_m"])))
            passed = observed <= float(check["max_error_m"])
        elif kind == "body_displacement":
            body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, check["body"])
            observed = float(np.linalg.norm(body[-1, body_id, :2] - body[0, body_id, :2]))
            passed = observed >= float(check["min_m"])
        elif kind == "joint_displacement":
            joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, check["joint"])
            address = model.jnt_qposadr[joint_id]
            observed = float(abs(qpos[-1, address] - qpos[0, address]))
            passed = observed >= float(check["min_rad"])
        elif kind == "contact":
            phase_name = check["phase"]
            robot = check["robot_geom_prefix"]
            obj = check["object_geom_prefix"]
            observed = sum(value for (phase, first, second), value in contacts.items()
                           if phase == phase_name and
                           ((first.startswith(robot) and second.startswith(obj)) or
                            (second.startswith(robot) and first.startswith(obj))))
            passed = observed >= int(check["min_steps"])
        elif kind == "terminal_base_rest":
            observed = float(np.linalg.norm(qvel[-1, :2]))
            passed = observed <= float(check["max_mps"])
        else:
            raise ValueError(f"unsupported check type: {kind}")
        results.append({"type": kind, "passed": bool(passed), "observed": observed,
                        "target": {key: value for key, value in check.items() if key != "type"}})
    return results


def simulate_rollout(input_dir: Path, store: Store, request: dict) -> dict:
    probe_only = bool(request.get("_probe_only", False))
    state = store.read()
    if not state["merges"] or not state["trajectories"]:
        raise ValueError("scene and trajectory required")
    merge, trajectory = current_scene(store, state), state["trajectories"][-1]
    pending_spec = state.get("pending_trajectory_spec_sha256")
    if pending_spec is not None and pending_spec != trajectory.get("source_spec_sha256"):
        raise ValueError("latest trajectory edit did not compile; fix compilation before probing or rollout")
    if (trajectory["scene_sha256"] != merge["scene_sha256"]
            or trajectory["settled_sha256"] != merge["settled_sha256"]):
        raise ValueError("trajectory is for an earlier scene")
    for field in ("source_spec", "targets"):
        if sha256(store.artifact(trajectory[field + "_path"])) != trajectory[field + "_sha256"]:
            raise ValueError("Compiled trajectory files changed. Compile the trajectory again.")
    scene_path = store.artifact(merge["scene_xml_path"])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    with np.load(store.artifact(merge["settled_state_path"])) as settled:
        data.qpos[:] = settled["qpos"]
        data.qvel[:] = settled["qvel"]
        data.ctrl[:] = settled["ctrl"]
    mujoco.mj_forward(model, data)
    with np.load(store.artifact(trajectory["targets_path"])) as targets:
        arm_targets = np.asarray(targets["arm_targets"])
        base_targets = np.asarray(targets["base_targets"])
        phase_indices = np.asarray(targets["phase_index"])
        timestamps = np.asarray(targets["timestamp_sec"], dtype=float)
        compiler_version = str(targets["compiler_version"])
    if compiler_version != COMPILER_VERSION:
        raise ValueError("trajectory was not compiled by the active semantic compiler")
    source_spec = json.loads(store.artifact(trajectory["source_spec_path"]).read_text())
    phases = source_spec["phases"]
    success_path = store.artifact(request["success_json_path"])
    success = json.loads(success_path.read_text())
    checks = success.get("required_checks", [])
    if not checks:
        raise ValueError("success contract requires checks")
    kinds = {check["type"] for check in checks}
    if not {"base_path_length", "base_stop", "contact", "terminal_base_rest"} <= kinds or not (
            {"body_displacement", "joint_displacement"} & kinds):
        raise ValueError("success contract must cover route, stop, contact, task motion, and rest")
    phase_names = {phase["name"] for phase in phases}
    for check in checks:
        if check["type"] in ("base_stop", "contact") and check.get("phase") not in phase_names:
            raise ValueError(f"{check['type']} check needs a valid phase")
    frozen = state.get("required_success_contract")
    if probe_only:
        pass  # A local probe cannot freeze or revise the authoritative task contract.
    elif frozen is None:
        state["required_success_contract"] = checks
        store.save(state)
    elif frozen != checks:
        revision_path = request.get("success_revision_path")
        if not revision_path:
            raise ValueError("changing success checks requires success_revision_path with rationale")
        revision = json.loads(store.artifact(revision_path).read_text())
        if not revision.get("rationale"):
            raise ValueError("success revision needs a rationale")
        state["required_success_contract"] = checks
        state.setdefault("success_revisions", []).append({"path": revision_path,
                                                          "rationale": revision["rationale"]})
        store.save(state)
    loaded_episode = load_episode(input_dir)
    contract = dict(loaded_episode.runtime_contract)
    if contract.get("base_drive") != "planar_velocity_xyyaw":
        raise ValueError("input robot runtime uses an obsolete base drive; regenerate robot runtime")
    base_joint_ids = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                      for name in contract["base_pose_joints"]]
    base_qpos = [int(model.jnt_qposadr[joint]) for joint in base_joint_ids]
    base_dofs = [int(model.jnt_dofadr[joint]) for joint in base_joint_ids]
    if len(base_qpos) != 3:
        raise ValueError("merged scene uses an obsolete robot asset; merge the scene again")
    arm_actuators = {side: [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                            for name in contract["arms"][side]["actuator_names"]]
                     for side in ("left", "right")}
    grippers = {side: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                         contract["arms"][side]["gripper_actuator"])
                for side in ("left", "right")}
    gripper_ranges = {side: contract["arms"][side].get(
        "gripper_control_range_m", [.021, .057]) for side in ("left", "right")}
    base_prefixes = tuple(contract.get("collision_groups", {}).get(
        "base_geom_prefixes", ["tracer/"]))
    arm_prefixes = tuple(contract.get("collision_groups", {}).get(
        "arm_geom_prefixes", ["left/", "right/"]))
    robot_prefixes = arm_prefixes + base_prefixes
    arm_dofs = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for side in ("left", "right")
                for name in contract["arms"][side]["joint_names"]]
    arm_dofs = [int(model.jnt_dofadr[joint]) for joint in arm_dofs]
    qpos_log, qvel_log, ctrl_log, body_log, route_log, contact_counts = [], [], [], [], [], {}
    initial_state = snapshot(model, data)
    step_control, step_twist, row_end_step = [], [], []
    phase_end = {}
    stage_metrics: list[dict] = []
    checkpoint_paths: list[str] = []
    max_penetration = 0.0
    arm_speed_rows = np.zeros(len(phase_indices))
    desired_jump_rows = np.zeros(len(phase_indices))
    contact_rows: dict[str, set[int]] = {}
    arm_actuator_ids = [actuator for side in ("left", "right")
                        for actuator in arm_actuators[side]]
    arm_target_rate = float(contract.get("compiler_joint_velocity_limit_radps", 1.0))
    gripper_target_rate = float(contract.get("gripper_target_rate_mps", .12))
    revision = (len(state.get("probes", [])) + 1 if probe_only
                else len(state["rollouts"]) + 1)
    rollout_dir = store.root / (f"simulation/probes/probe_{revision}" if probe_only
                                else f"simulation/rollout_r{revision}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    stop_index = None
    if probe_only:
        stop_index = next((index for index, phase in enumerate(phases)
                           if phase["name"] == request["phase_name"]), None)
        if stop_index is None:
            raise ValueError(f"phase not found: {request['phase_name']}")
        validated = trajectory.get("validated_through_index")
        if validated is not None and stop_index > int(validated):
            raise ValueError("probe phase exceeds compile_through_phase; compile a longer prefix")
    elif trajectory.get("validated_through_index") is not None:
        raise ValueError("partial trajectory is probe-only; compile without compile_through_phase for full rollout")
    engine_hash = execution_hash()
    scene_hash = merge["scene_sha256"]
    settled_hash = sha256(store.artifact(merge["settled_state_path"]))
    success_hash = sha256(success_path)
    reused_stages = 0
    prior_records = [*state["rollouts"], *state.get("probes", [])]
    previous, reused_stages = longest_prefix(
        prior_records, trajectory["stage_hashes"],
        stop_index + 1 if probe_only else len(phases),
        scene_hash, settled_hash, engine_hash)
    if reused_stages:
        prior_diag = json.loads(store.artifact(previous["diagnostics_path"]).read_text())
        prior_metrics = prior_diag.get("stage_metrics", [])
        if len(prior_metrics) < reused_stages:
            reused_stages = 0
        else:
            prefix_rows = phases[reused_stages - 1]["end_row"]
            with np.load(store.artifact(previous["rollout_path"])) as prior:
                step_count = int(prior["row_end_step"][prefix_rows - 1])
                step_control = list(prior["step_control"][:step_count])
                step_twist = list(prior["step_twist"][:step_count])
                row_end_step = list(prior["row_end_step"][:prefix_rows])
                qpos_log = list(prior["qpos"][:prefix_rows].copy())
                qvel_log = list(prior["qvel"][:prefix_rows].copy())
                ctrl_log = list(prior["ctrl"][:prefix_rows].copy())
                body_log = list(prior["body_positions"][:prefix_rows].copy())
                route_log = list(prior["base_xyyaw"][:prefix_rows].copy())
                arm_speed_rows[:prefix_rows] = prior["arm_speed_radps"][:prefix_rows]
                desired_jump_rows[:prefix_rows] = prior["desired_target_jump_rad"][:prefix_rows]
            stage_metrics = prior_metrics[:reused_stages]
            checkpoint_paths = previous["checkpoint_paths"][:reused_stages]
            with np.load(store.artifact(checkpoint_paths[-1])) as checkpoint:
                restore(model, data, checkpoint["integration_state"])
            for metric in stage_metrics:
                phase_end[metric["phase"]] = metric["end_xyyaw"]
                max_penetration = max(max_penetration, metric["max_penetration_m"])
                for item in metric["contacts"]:
                    key = (metric["phase"], *item["geoms"])
                    contact_counts[key] = item["pair_substeps"]
                contact_rows[metric["phase"]] = set(metric["contact_rows"])
    start_row = phases[reused_stages - 1]["end_row"] if reused_stages else 0
    stage = None
    end_row = phases[stop_index]["end_row"] if probe_only else len(phase_indices)
    physics_started = time.monotonic()
    for row in range(start_row, end_row):
        phase_index = int(phase_indices[row])
        phase = phases[int(phase_index)]
        phase_name = phase["name"]
        action = phase.get("action", "legacy")
        if row == phase["start_row"]:
            object_name = phase.get("object")
            hand_monitor = HandObjectMonitor(model, data, phase)
            evidence_sides = list(dict.fromkeys(phase.get("grasp_sides", []) +
                                               phase.get("contact_sides", [])))
            object_id = (_id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
                         if object_name else None)
            result_joint = phase.get("result_joint")
            joint_address = (int(model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                      result_joint)]) if result_joint else None)
            stage = {"phase": phase_name, "action": action, "object": object_name,
                     "start_xyyaw": data.qpos[base_qpos].copy(),
                     "object_start": data.xpos[object_id].copy() if object_id is not None else None,
                     "minimum_object_z_m": (float(data.xpos[object_id, 2])
                                            if object_id is not None else None),
                     "result_joint": result_joint, "joint_address": joint_address,
                     "joint_start": float(data.qpos[joint_address]) if joint_address is not None else None,
                     "relative_start": {side: relative_pinch(model, data, object_name, side)
                                        for side in phase.get("grasp_sides", [])},
                     "contact_substeps": {side: 0 for side in evidence_sides},
                     "hand_contact_substeps": {side: 0 for side in evidence_sides},
                     "pad_pair_substeps": {side: 0 for side in evidence_sides},
                     "last_row_pad_pair_substeps": {side: 0 for side in phase.get("grasp_sides", [])},
                     "max_slip_m": {side: 0.0 for side in phase.get("grasp_sides", [])},
                     "max_tilt_deg": 0.0, "self_collision_substeps": 0, "self_collision_pairs": {},
                     "base_object_contact_substeps": 0, "base_object_pairs": {},
                     "payload_structure_contact_substeps": 0,
                     "payload_structure_pairs": {}, "peak_goal_progress_m": 0.0,
                     "max_goal_progress_reversal_m": 0.0,
                     "arm_scene_contact_substeps": 0, "arm_scene_pairs": {},
                     "contact_pairs": {}, "contact_rows": set(), "max_penetration_m": 0.0}
        for side in stage["last_row_pad_pair_substeps"]:
            stage["last_row_pad_pair_substeps"][side] = 0
        desired_ctrl = data.ctrl.copy()
        for side, offset in (("left", 0), ("right", 7)):
            values = arm_targets[row, offset:offset + 6]
            for index, actuator in enumerate(arm_actuators[side]):
                desired_ctrl[actuator] = values[index]
            opening = float(np.clip(arm_targets[row, offset + 6], 0, 1))
            low, high = map(float, gripper_ranges[side])
            desired_ctrl[grippers[side]] = low + (high - low) * opening
        desired_jump_rows[row] = float(np.max(np.abs(desired_ctrl[arm_actuator_ids] -
                                                     data.ctrl[arm_actuator_ids])))
        row_dt = float(timestamps[row] - timestamps[row - 1]) if row else (
            float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else .02)
        previous_time = float(timestamps[row - 1] - timestamps[0]) if row else 0.0
        next_time = float(timestamps[row] - timestamps[0])
        steps = max(0, round(next_time / model.opt.timestep) -
                    round(previous_time / model.opt.timestep))
        for _ in range(steps):
            delta = desired_ctrl[arm_actuator_ids] - data.ctrl[arm_actuator_ids]
            data.ctrl[arm_actuator_ids] += np.clip(
                delta, -arm_target_rate * model.opt.timestep,
                arm_target_rate * model.opt.timestep)
            for actuator in grippers.values():
                data.ctrl[actuator] += float(np.clip(
                    desired_ctrl[actuator] - data.ctrl[actuator],
                    -gripper_target_rate * model.opt.timestep,
                    gripper_target_rate * model.opt.timestep))
            yaw = float(data.qpos[base_qpos[2]])
            target_vx, target_vy, target_w = map(float, base_targets[max(0, row - 1)])
            base_spec = phase.get("base", {})
            if "stop_xy_m" in base_spec:
                dx, dy = (np.asarray(base_spec["stop_xy_m"], dtype=float) -
                          data.qpos[base_qpos[:2]])
                forward_error = math.cos(yaw) * dx + math.sin(yaw) * dy
                target_vx += np.clip(.8 * forward_error, -.25, .25)

            if "stop_yaw_rad" in base_spec:
                target_w += np.clip(1.0 * _wrap(float(base_spec["stop_yaw_rad"]) - yaw), -.5, .5)
            if action == "operate_arm":
                target_vx = target_vy = target_w = 0.0
            command_vx = float(np.clip(target_vx, -.6, .6))
            command_vy = 0.0
            command_w = float(np.clip(target_w, -1.4, 1.4))
            twist = (command_vx, command_vy, command_w)
            step_mobile(model, data, base_qpos, base_dofs, twist)
            step_twist.append(twist)
            step_control.append(data.ctrl.copy())
            arm_speed_rows[row] = max(arm_speed_rows[row],
                                      float(np.max(np.abs(data.qvel[arm_dofs]))))
            contacts_this_step = set()
            contacted = {side: set() for side in hand_monitor.hand}
            self_hit = False
            self_pairs = set()
            base_object_pairs = set()
            payload_structure_pairs = set()
            arm_scene_pairs = set()
            for contact in data.contact:
                names = sorted((_geom_label(model, contact.geom1),
                                _geom_label(model, contact.geom2)))
                contacts_this_step.add((phase_name, *names))
                max_penetration = max(max_penetration, -float(contact.dist))
                stage["max_penetration_m"] = max(stage["max_penetration_m"], -float(contact.dist))
                first, second = contact_side(model, contact.geom1), contact_side(model, contact.geom2)
                tracer_contact = any(name.startswith(base_prefixes) for name in names)
                if (first and second) or (
                        tracer_contact and (first or second)):
                    self_hit = True
                    self_pairs.add(tuple(names))
                if tracer_contact and not (first or second or "floor" in names):
                    base_object_pairs.add(tuple(names))
                if tracer_contact and any(
                        object_geom(model, geom, obj)
                        for obj in {b["object"] for b in hand_monitor.hand.values()}
                        for geom in (contact.geom1, contact.geom2)):
                    payload_structure_pairs.add(tuple(names))
                for robot_geom, task_geom in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                    label = contact_side(model, robot_geom)
                    if label and label[0] in contacted:
                        binding = hand_monitor.hand[label[0]]
                        prefix = binding.get("contact_geom_prefix", phase.get("contact_geom_prefix", "")
                                             if binding["object"] == stage["object"] else "")
                        if object_geom(model, task_geom, binding["object"]) and _geom_label(model, task_geom).startswith(prefix):
                            contacted[label[0]].add(label[1])
                for robot_geom, scene_geom in ((contact.geom1, contact.geom2),
                                               (contact.geom2, contact.geom1)):
                    robot_label = contact_side(model, robot_geom)
                    scene_name = _geom_label(model, scene_geom)
                    if not robot_label or scene_name.startswith(robot_prefixes):
                        continue
                    if scene_name == "floor":
                        continue
                    intended = intended_contact(model, scene_geom, robot_label, hand_monitor.hand)
                    if not intended:
                        arm_scene_pairs.add(tuple(names))
            stage["self_collision_substeps"] += int(self_hit)
            for pair in self_pairs:
                stage["self_collision_pairs"][pair] = stage["self_collision_pairs"].get(pair, 0) + 1
            stage["base_object_contact_substeps"] += int(bool(base_object_pairs))
            stage["payload_structure_contact_substeps"] += int(bool(payload_structure_pairs))
            stage["arm_scene_contact_substeps"] += int(bool(arm_scene_pairs))
            for pair in base_object_pairs:
                stage["base_object_pairs"][pair] = stage["base_object_pairs"].get(pair, 0) + 1
            for pair in payload_structure_pairs:
                stage["payload_structure_pairs"][pair] = (
                    stage["payload_structure_pairs"].get(pair, 0) + 1)
            for pair in arm_scene_pairs:
                stage["arm_scene_pairs"][pair] = stage["arm_scene_pairs"].get(pair, 0) + 1
            hand_monitor.step(model, data, contacted)
            for side, parts in contacted.items():
                if side not in stage["contact_substeps"]:
                    continue
                stage["contact_substeps"][side] += int(bool(parts))
                stage["hand_contact_substeps"][side] += int(bool(
                    parts & {"left_pad", "right_pad", "hand"}))
                stage["pad_pair_substeps"][side] += int(
                    "left_pad" in parts and "right_pad" in parts)
                if side in stage["last_row_pad_pair_substeps"]:
                    stage["last_row_pad_pair_substeps"][side] += int(
                        "left_pad" in parts and "right_pad" in parts)
            for key in contacts_this_step:
                contact_counts[key] = contact_counts.get(key, 0) + 1
                stage["contact_pairs"][key[1:]] = stage["contact_pairs"].get(key[1:], 0) + 1
                if (any(name.startswith(arm_prefixes) for name in key[1:]) and
                        not any(name.startswith(base_prefixes) for name in key[1:])):
                    contact_rows.setdefault(phase_name, set()).add(row)
                    stage["contact_rows"].add(row)
        if not np.isfinite(data.qpos).all():
            raise ValueError(f"nonfinite rollout at row {row}")
        row_end_step.append(len(step_control))
        qpos_log.append(data.qpos.copy())
        qvel_log.append(data.qvel.copy())
        ctrl_log.append(data.ctrl.copy())
        body_log.append(data.xpos.copy())
        route_log.append(data.qpos[base_qpos].copy())
        if stage["object"]:
            stage["minimum_object_z_m"] = min(stage["minimum_object_z_m"],
                float(data.xpos[_id(model, mujoco.mjtObj.mjOBJ_BODY, stage["object"]), 2]))
            stage["max_tilt_deg"] = max(stage["max_tilt_deg"],
                                        object_tilt_deg(model, data, stage["object"]))
            for side, start in stage["relative_start"].items():
                stage["max_slip_m"][side] = max(stage["max_slip_m"][side],
                    float(np.linalg.norm(relative_pinch(model, data, stage["object"], side) - start)))
            if "goal_direction_xy" in phase:
                direction = np.asarray(phase["goal_direction_xy"], dtype=float)
                if direction.shape == (2,) and np.linalg.norm(direction) > 1e-8:
                    progress = float(np.dot(
                        data.xpos[_id(model, mujoco.mjtObj.mjOBJ_BODY, stage["object"]), :2] -
                        stage["object_start"][:2], direction / np.linalg.norm(direction)))
                    stage["peak_goal_progress_m"] = max(stage["peak_goal_progress_m"], progress)
                    stage["max_goal_progress_reversal_m"] = max(
                        stage["max_goal_progress_reversal_m"],
                        stage["peak_goal_progress_m"] - progress)
        if row + 1 == phase["end_row"]:
            phase_end[phase_name] = route_log[-1].tolist()
            base_delta = data.qpos[base_qpos[:2]] - stage["start_xyyaw"][:2]
            object_delta = (data.xpos[_id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         stage["object"])] - stage["object_start"]
                            if stage["object"] else np.zeros(3))
            phase_duration = max(.02, float(timestamps[phase["end_row"] - 1] -
                                             timestamps[phase["start_row"]]))
            stage_steps = max(1, round(phase_duration / model.opt.timestep))
            failures = hand_monitor.failures()
            if action == "operate_arm" and np.linalg.norm(base_delta) > .015:
                failures.append("base_moved_during_arm_operation")
            if action == "operate_arm" and phase.get("grasp_sides") and not hand_monitor.enabled:
                for side in phase.get("grasp_sides", []):
                    if stage["last_row_pad_pair_substeps"][side] == 0:
                        failures.append(f"{side}_grasp_not_ready")
            if action != "legacy" and stage["self_collision_substeps"] > max(4, .05 * stage_steps):
                failures.append("robot_self_collision")
            if action != "legacy" and stage["base_object_contact_substeps"] > int(
                    phase.get("max_base_object_contact_substeps", 4)):
                failures.append("base_collided_with_task_scene")
            if action != "legacy" and stage["arm_scene_contact_substeps"] > int(
                    phase.get("max_arm_scene_contact_substeps", 4)):
                failures.append("arm_collided_with_non_target_scene")
            if ((phase.get("intent") == "supported_carry" or any(b["mode"] == "hold" for b in hand_monitor.hand.values())) and
                    stage["payload_structure_contact_substeps"] > int(
                        phase.get("max_payload_structure_contact_substeps", 4))):
                failures.append("payload_collided_with_robot_structure")
            if stage["max_goal_progress_reversal_m"] > float(
                    phase.get("max_progress_reversal_m", .04)):
                failures.append("object_progress_reversed")
            for requirement in phase.get("preserve_joint_ranges", []):
                joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, requirement["joint"])
                observed = float(data.qpos[model.jnt_qposadr[joint]])
                if not float(requirement.get("min_rad", -math.inf)) <= observed <= float(
                        requirement.get("max_rad", math.inf)):
                    failures.append(f"preserved_joint_out_of_range:{requirement['joint']}")
            if action in ("push_with_base", "pull_with_base", "carry_with_base"):
                local_time = np.diff(timestamps[phase["start_row"]:phase["end_row"]], prepend=
                                     timestamps[phase["start_row"]])
                signed_forward = float(np.sum(
                    base_targets[phase["start_row"]:phase["end_row"], 0] * local_time))
                if action == "push_with_base" and signed_forward < .02:
                    failures.append("push_requires_forward_base_motion")
                if action == "pull_with_base" and signed_forward > -.02:
                    failures.append("pull_requires_reverse_base_motion")
                if np.linalg.norm(base_delta) < float(phase.get("min_base_motion_m", .05)):
                    failures.append("insufficient_base_motion")
                if stage["joint_address"] is not None:
                    if abs(float(data.qpos[stage["joint_address"]]) - stage["joint_start"]) < float(phase.get("min_joint_motion_rad", .1)):
                        failures.append("joint_did_not_move")
                elif np.dot(object_delta[:2], base_delta) < float(phase.get("min_object_progress_m", .02)) * max(np.linalg.norm(base_delta), 1e-9):
                    failures.append("object_did_not_follow_base")
                for side in phase.get("grasp_sides", []):
                    if stage["pad_pair_substeps"][side] < int(phase.get("min_grasp_substeps", 5)):
                        failures.append(f"{side}_finger_grasp_missing")
                    if stage["max_slip_m"][side] > float(phase.get("max_slip_m", .08)):
                        failures.append(f"{side}_grasp_slipped")
                for side in phase.get("contact_sides", []):
                    minimum = int(phase.get("min_hand_contact_substeps",
                                            .25 * stage_steps))
                    if stage["hand_contact_substeps"][side] < minimum:
                        failures.append(f"{side}_hand_contact_insufficient")
                if action == "carry_with_base" and stage["minimum_object_z_m"] < float(phase.get("carry_height_m", .15)):
                    failures.append("object_not_lifted")
                if stage["max_tilt_deg"] > float(phase.get("max_tilt_deg", 20)):
                    failures.append("object_tipped")
                if "goal_direction_xy" in phase:
                    direction = np.asarray(phase["goal_direction_xy"], dtype=float)
                    if direction.shape != (2,) or np.linalg.norm(direction) < 1e-6:
                        raise ValueError("goal_direction_xy must be a nonzero 2-vector")
                    progress = float(np.dot(object_delta[:2], direction / np.linalg.norm(direction)))
                    if progress < float(phase.get("min_goal_progress_m", .02)):
                        failures.append("object_moved_wrong_direction")
            object_speed = None
            if stage["object"]:
                task_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, stage["object"])
                object_speed = float(np.linalg.norm(data.cvel[task_body, 3:]))
                if "max_object_end_speed_mps" in phase and object_speed > float(phase["max_object_end_speed_mps"]):
                    failures.append("object_not_resting")
            metric = {"phase": phase_name, "action": action, "object": stage["object"],
                      "hand_objects": hand_monitor.metrics if hand_monitor.enabled else {},
                      "end_xyyaw": phase_end[phase_name],
                      "base_displacement_m": float(np.linalg.norm(base_delta)),
                      "object_displacement_xyz_m": object_delta.tolist(),
                      "minimum_object_z_m": stage["minimum_object_z_m"],
                      "object_end_speed_mps": object_speed,
                      "result_joint": stage["result_joint"],
                      "contact_substeps": stage["contact_substeps"],
                      "hand_contact_substeps": stage["hand_contact_substeps"],
                      "pad_pair_substeps": stage["pad_pair_substeps"],
                      "last_row_pad_pair_substeps": stage["last_row_pad_pair_substeps"],
                      "max_slip_m": stage["max_slip_m"],
                      "max_tilt_deg": stage["max_tilt_deg"],
                      "self_collision_substeps": stage["self_collision_substeps"],
                      "self_collision_top_pairs": [{"geoms": list(pair), "substeps": count}
                                                   for pair, count in sorted(
                                                       stage["self_collision_pairs"].items(),
                                                       key=lambda item: -item[1])[:5]],
                      "base_object_contact_substeps": stage["base_object_contact_substeps"],
                      "payload_structure_contact_substeps": stage["payload_structure_contact_substeps"],
                      "payload_structure_top_pairs": [{"geoms": list(pair), "substeps": count}
                                                        for pair, count in sorted(
                                                            stage["payload_structure_pairs"].items(),
                                                            key=lambda item: -item[1])[:5]],
                      "arm_scene_contact_substeps": stage["arm_scene_contact_substeps"],
                      "arm_scene_top_pairs": [{"geoms": list(pair), "substeps": count}
                                                for pair, count in sorted(
                                                    stage["arm_scene_pairs"].items(),
                                                    key=lambda item: -item[1])[:5]],
                      "max_goal_progress_reversal_m": stage["max_goal_progress_reversal_m"],
                      "base_object_top_pairs": [{"geoms": list(pair), "substeps": count}
                                                for pair, count in sorted(
                                                    stage["base_object_pairs"].items(),
                                                    key=lambda item: -item[1])[:5]],
                      "max_penetration_m": stage["max_penetration_m"],
                      "contact_rows": sorted(stage["contact_rows"]),
                      "contacts": [{"geoms": list(pair), "pair_substeps": count}
                                   for pair, count in stage["contact_pairs"].items()],
                      "passed": not failures, "failures": failures}
            stage_metrics.append(metric)
            checkpoint_path = rollout_dir / f"checkpoint_stage{phase_index + 1}.npz"
            np.savez_compressed(checkpoint_path, integration_state=snapshot(model, data))
            checkpoint_paths.append(str(checkpoint_path.relative_to(store.root)))
            if probe_only and failures and request.get("stop_at_first_failure", True):
                break
    physics_elapsed = time.monotonic() - physics_started
    qpos, qvel, body, route = map(np.asarray, (qpos_log, qvel_log, body_log, route_log))
    passed_prefix = next((index for index, metric in enumerate(stage_metrics)
                          if not metric["passed"]), len(stage_metrics))
    commands = dict(initial_state=initial_state, row_end_step=row_end_step,
                    step_control=np.asarray(step_control).reshape(-1, model.nu),
                    step_twist=np.asarray(step_twist).reshape(-1, 3))
    if probe_only:
        if not len(qpos):
            raise ValueError("local probe has no simulated rows")
        output = rollout_dir / "states.npz"
        np.savez_compressed(output, **commands, qpos=qpos, qvel=qvel, ctrl=np.asarray(ctrl_log),
                            body_positions=body, base_xyyaw=route,
                            phase_index=phase_indices[:len(qpos)],
                            arm_speed_radps=arm_speed_rows[:len(qpos)],
                            desired_target_jump_rad=desired_jump_rows[:len(qpos)],
                            timestamp_sec=timestamps[:len(qpos)])
        blocker = next((metric for metric in stage_metrics if not metric["passed"]), None)
        terminal_results = (evaluate_terminal_goals(model, qpos, checks)
                            if stop_index == len(phases) - 1 and len(qpos) == len(phase_indices)
                            else [])
        failed_terminal = [item for item in terminal_results if not item["passed"]]
        probe_failed = bool(blocker or failed_terminal)
        focus = blocker or stage_metrics[-1]
        focus_index = next(index for index, phase in enumerate(phases)
                           if phase["name"] == focus["phase"])
        focus_phase = phases[focus_index]
        image = rollout_dir / "local_comparison.jpg"
        first, end = focus_phase["start_row"], focus_phase["end_row"]
        rows = sorted({first, (first + min(end, len(qpos)) - 1) // 2, min(end, len(qpos)) - 1})
        render_started = time.monotonic()
        render_rollout_sheet(input_dir, scene_path, output, phases, image,
                             rows_override=rows)
        diagnostics_path = rollout_dir / "diagnostics.json"
        write_json(diagnostics_path, {"schema_version": "mobile-stage-probe-v1.3",
                                      "status": "FAIL" if probe_failed else "PASS",
                                      "authoritative": False, "stage_metrics": stage_metrics,
                                      "terminal_goal_results": terminal_results,
                                      "first_failed_phase": (blocker["phase"] if blocker else
                                                             phases[-1]["name"] if failed_terminal else None),
                                      "stage_sheet_paths": [None] * focus_index +
                                      [str(image.relative_to(store.root))],
                                      "reused_stages": reused_stages,
                                      "timing": {"physics_sec": round(physics_elapsed, 3),
                                                 "render_sec": round(time.monotonic() - render_started, 3)}})
        record = {"revision": revision, "scene_sha256": scene_hash,
                  "settled_sha256": settled_hash, "engine_sha256": engine_hash,
                  "stage_hashes": trajectory["stage_hashes"],
                  "checkpoint_paths": checkpoint_paths, "passed_prefix": passed_prefix,
                  "rollout_path": str(output.relative_to(store.root)),
                  "diagnostics_path": str(diagnostics_path.relative_to(store.root))}
        state.setdefault("probes", []).append(record)
        store.save(state)
        response = {"status": "FAIL" if probe_failed else "PASS", "authoritative": False,
                    "phase": focus["phase"],
                    "first_failed_phase": (blocker["phase"] if blocker else
                                           phases[-1]["name"] if failed_terminal else None),
                    "failures": focus["failures"], "failed_terminal_goals": failed_terminal[:3],
                    "first_blocker": ({"phase": blocker["phase"], "failures": blocker["failures"],
                        "hand_objects": blocker.get("hand_objects", {}),
                        "contacts": blocker.get("contacts", [])[:8],
                        "collision_pairs": (blocker.get("self_collision_top_pairs", []) +
                            blocker.get("arm_scene_top_pairs", []) +
                            blocker.get("payload_structure_top_pairs", []) +
                            blocker.get("base_object_top_pairs", []))[:4],
                        "local_sheet_path": str(image.relative_to(store.root))} if blocker else None),
                    "hand_objects": focus.get("hand_objects", {}),
                    "scene_revision": state["merges"][-1]["revision"],
                    "grasp_scope": "PASS covers executed phases only; probe through a short lift/clearance phase before committing to carry",
                    "reused_stages": reused_stages,
                    "local_sheet_path": str(image.relative_to(store.root)),
                    "diagnostics_path": record["diagnostics_path"]}
        write_json(rollout_dir / "summary.json", response)
        return response
    results = _evaluate(model, {}, checks, qpos, qvel, body, route, contact_counts, phase_end)
    results.extend(evaluate_terminal_goals(model, qpos, checks))
    object_quality = full_task_object_quality(model, qpos, phase_indices, phases)
    max_arm_speed = float(arm_speed_rows.max())
    sample_dt = np.maximum(np.diff(timestamps[:len(qvel)]), model.opt.timestep)
    arm_acceleration = np.max(
        np.abs(np.diff(qvel[:, arm_dofs], axis=0) / sample_dt[:, None]), axis=1)
    max_arm_acceleration = float(arm_acceleration.max())
    speed_limit = float(contract.get("compiler_joint_velocity_limit_radps", 1.0)) * 1.5
    acceleration_limit = float(contract.get("compiler_joint_acceleration_limit_radps2", 4.0)) * 3
    motion_quality = {"passed": max_arm_speed <= speed_limit and max_arm_acceleration <= acceleration_limit,
                      "max_arm_speed_radps": max_arm_speed,
                      "max_arm_speed_row": int(arm_speed_rows.argmax()),
                      "max_arm_acceleration_radps2": max_arm_acceleration,
                      "max_arm_acceleration_row": int(arm_acceleration.argmax() + 1),
                      "max_desired_target_jump_rad": float(desired_jump_rows.max()),
                      "max_desired_target_jump_row": int(desired_jump_rows.argmax()),
                      "arm_target_execution": "frozen_compiled_position_targets",
                      "arm_speed_limit_radps": speed_limit,
                      "arm_acceleration_limit_radps2": acceleration_limit}
    passed = (all(check["passed"] for check in results) and motion_quality["passed"]
              and all(metric["passed"] for metric in stage_metrics)
              and all(item["passed"] for item in object_quality))
    failed_checks = [check for check in results if not check["passed"]]
    failed_phase_names = {check["target"].get("phase") for check in failed_checks}
    failed_phase_names.update(metric["phase"] for metric in stage_metrics if not metric["passed"])
    failed_object_phases = {item["first_bad_phase"] for item in object_quality if not item["passed"]}
    failed_phase_names.update(failed_object_phases)
    failed_motion_phases = set()
    if not motion_quality["passed"]:
        for key, limit, value in (("max_arm_speed_row", speed_limit, max_arm_speed),
                                   ("max_arm_acceleration_row", acceleration_limit, max_arm_acceleration)):
            if value > limit:
                name = phases[int(phase_indices[motion_quality[key]])]["name"]
                failed_phase_names.add(name)
                failed_motion_phases.add(name)
    first_failed_phase = next((phase["name"] for phase in phases
                               if phase["name"] in failed_phase_names), None)
    if first_failed_phase is None and failed_checks:
        first = failed_checks[0]
        target_name = first["target"].get("body") or first["target"].get("subject")
        joint_name = first["target"].get("joint")
        related = [phase for phase in phases
                   if phase.get("object") == target_name or phase.get("result_joint") == joint_name]
        first_failed_phase = (related[-1] if related else phases[-1])["name"]
    output = rollout_dir / "states.npz"
    np.savez_compressed(output, **commands, qpos=qpos, qvel=qvel, ctrl=np.asarray(ctrl_log),
                        body_positions=body, base_xyyaw=route,
                        phase_index=phase_indices, arm_speed_radps=arm_speed_rows,
                        desired_target_jump_rad=desired_jump_rows,
                        timestamp_sec=timestamps[:len(qpos)])
    sheet_path = rollout_dir / "comparison.jpg"
    stage_sheet_paths = [None] * len(phases)
    if reused_stages:
        for index, path in enumerate(prior_diag.get("stage_sheet_paths", [])[:reused_stages]):
            stage_sheet_paths[index] = path
    render_started = time.monotonic()
    with RolloutSheetRenderer(input_dir, scene_path, output, phases) as render_context:
        if passed:
            sheet_rows = render_rollout_sheet(input_dir, scene_path, output, phases, sheet_path, context=render_context)
            render_indices = range(len(phases))
        else:
            focus_index = next((index for index, phase in enumerate(phases)
                                if phase["name"] == first_failed_phase), len(phases) - 1)
            focus = phases[focus_index]
            first, end = focus["start_row"], focus["end_row"]
            sheet_rows = sorted({first, (first + end - 1) // 2, end - 1})
            render_rollout_sheet(input_dir, scene_path, output, phases, sheet_path,
                                 rows_override=sheet_rows, context=render_context)
            render_indices = range(max(0, focus_index - 1), min(len(phases), focus_index + 2))
        for index in render_indices:
            inherited = stage_sheet_paths[index]
            if index < reused_stages and inherited and store.artifact(inherited).is_file():
                continue
            phase = phases[index]
            first, end = phase["start_row"], phase["end_row"]
            stage_sheet = rollout_dir / f"stage{index + 1}_comparison.jpg"
            rows = sorted({first, (first + end - 1) // 2, end - 1})
            render_rollout_sheet(input_dir, scene_path, output, phases, stage_sheet,
                                 rows_override=rows, context=render_context)
            stage_sheet_paths[index] = str(stage_sheet.relative_to(store.root))
    diagnostics_path = rollout_dir / "diagnostics.json"
    phase_motion = [{"phase": phase["name"],
                     "max_arm_speed_radps": float(arm_speed_rows[phase["start_row"]:phase["end_row"]].max()),
                     "end_xyyaw": phase_end[phase["name"]],
                     "contact_rows": sorted(contact_rows.get(phase["name"], set()))[:8]}
                    for phase in phases]
    round_delta = None
    if state["rollouts"]:
        prior = json.loads(store.artifact(state["rollouts"][-1]["diagnostics_path"]).read_text())
        round_delta = {"previous_status": prior["status"],
                       "path_length_change_m": float(
                           np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum() -
                           prior["path_length_m"]),
                       "failed_check_count_change": len(failed_checks) - sum(
                           not check["passed"] for check in prior["required_checks"]),
                       "failed_stage_count_change": sum(not m["passed"] for m in stage_metrics) -
                           sum(not m["passed"] for m in prior.get("stage_metrics", []))}
        if "motion_quality" in prior:
            round_delta["max_arm_speed_change_radps"] = (
                max_arm_speed - prior["motion_quality"]["max_arm_speed_radps"])
    diagnostics = {"schema_version": "mobile-rollout-diagnostics-v1.3",
                   "status": "PASS" if passed else "FAIL", "required_checks": results,
                   "motion_quality": motion_quality, "phase_motion": phase_motion,
                   "stage_metrics": stage_metrics, "reused_stages": reused_stages,
                   "timing": {"physics_sec": round(physics_elapsed, 3),
                              "render_sec": round(time.monotonic() - render_started, 3)},
                   "object_quality": object_quality,
                   "stage_sheet_paths": stage_sheet_paths,
                   "first_failed_phase": first_failed_phase, "round_delta": round_delta,
                   "comparison_sheet_path": str(sheet_path.relative_to(store.root)),
                   "comparison_rows": sheet_rows,
                   "path_length_m": float(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum()),
                   "net_xyyaw": route[-1].tolist(), "phase_end_xyyaw": phase_end,
                   "max_penetration_m": max_penetration,
                   "top_contacts": [{"phase": key[0], "geoms": list(key[1:]),
                                     "pair_substeps": value}
                                    for key, value in sorted(contact_counts.items(), key=lambda x: -x[1])
                                    if "floor" not in key[1:]][:20]}
    write_json(diagnostics_path, diagnostics)
    record = {"revision": revision, "status": diagnostics["status"],
              "scene_sha256": scene_hash, "settled_sha256": settled_hash,
              "engine_sha256": engine_hash,
              "trajectory_revision": trajectory["revision"],
              "success_contract_path": request["success_json_path"],
              "success_contract_sha256": success_hash,
              "stage_hashes": trajectory["stage_hashes"],
              "checkpoint_paths": checkpoint_paths, "passed_prefix": passed_prefix,
              "rollout_path": str(output.relative_to(store.root)),
              "comparison_sheet_path": str(sheet_path.relative_to(store.root)),
              "diagnostics_path": str(diagnostics_path.relative_to(store.root))}
    state["rollouts"].append(record)
    state["stage"] = "ROLLOUT_PASS" if passed else "ROLLOUT_FAIL"
    store.save(state)
    first_blocker = None
    if not passed:
        metric = next((item for item in stage_metrics
                       if item["phase"] == first_failed_phase and not item["passed"]), None)
        if metric is not None:
            first_blocker = {"phase": first_failed_phase, "source": "stage",
                             "code": metric["failures"][0],
                             "hand_objects": metric.get("hand_objects", {}),
                             "contacts": metric.get("contacts", [])[:8],
                             "contact_substeps": metric["contact_substeps"],
                             "hand_contact_substeps": metric["hand_contact_substeps"],
                             "pad_pair_substeps": metric["pad_pair_substeps"],
                             "collision_pairs": (metric.get("self_collision_top_pairs", []) +
                                                 metric["arm_scene_top_pairs"] +
                                                 metric["payload_structure_top_pairs"] +
                                                 metric["base_object_top_pairs"])[:3],
                             "local_sheet_path": stage_sheet_paths[next(
                                 index for index, phase in enumerate(phases)
                                 if phase["name"] == first_failed_phase)]}
        elif failed_checks and first_failed_phase not in (failed_motion_phases | failed_object_phases):
            check = next((item for item in failed_checks
                          if item["target"].get("phase") == first_failed_phase), failed_checks[0])
            first_blocker = {"phase": first_failed_phase, "source": "success_contract",
                             "code": check["type"], "observed": check["observed"],
                             "required": check["target"],
                             "local_sheet_path": stage_sheet_paths[next(
                                 index for index, phase in enumerate(phases)
                                 if phase["name"] == first_failed_phase)]}
        else:
            first_blocker = {"phase": first_failed_phase, "source": "motion_or_object",
                             "code": "motion_quality" if first_failed_phase in failed_motion_phases else "object_quality",
                             "local_sheet_path": stage_sheet_paths[next(
                                 index for index, phase in enumerate(phases)
                                 if phase["name"] == first_failed_phase)]}
    response = {"status": record["status"], "rollout_revision": revision,
            "first_blocker": first_blocker,
            "failed_checks": failed_checks[:3], "failed_check_count": len(failed_checks),
            "first_failed_phase": first_failed_phase,
            "failed_stages": [{"phase": metric["phase"], "failures": metric["failures"],
                               "contact_substeps": metric["contact_substeps"],
                               "hand_contact_substeps": metric["hand_contact_substeps"],
                               "pad_pair_substeps": metric["pad_pair_substeps"],
                               "max_slip_m": metric["max_slip_m"],
                               "self_collision_substeps": metric["self_collision_substeps"],
                               "base_object_contact_substeps": metric["base_object_contact_substeps"]}
                              for metric in stage_metrics if not metric["passed"]][:1],
            "failed_stage_count": sum(not metric["passed"] for metric in stage_metrics),
            "failed_objects": [item for item in object_quality if not item["passed"]],
            "reused_stages": reused_stages,
            "failed_stage_sheet_path": (stage_sheet_paths[next(
                i for i, phase in enumerate(phases) if phase["name"] == first_failed_phase)]
                if first_failed_phase else None),
            "path_length_m": diagnostics["path_length_m"], "round_delta": round_delta,
            "motion_quality": motion_quality,
            "comparison_sheet_path": record["comparison_sheet_path"],
            "max_penetration_m": max_penetration,
            "diagnostics_path": record["diagnostics_path"]}
    write_json(rollout_dir / "summary.json", response)
    return response


def probe_mobile_stage(input_dir: Path, store: Store, request: dict) -> dict:
    """Run one bounded stage diagnosis without granting an authoritative PASS."""
    return simulate_rollout(input_dir, store, {**request, "_probe_only": True})
