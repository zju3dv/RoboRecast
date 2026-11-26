from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from mobile_input.writer import sha256, write_json
from .goals import validate_action_intent
from .semantic_trajectory import COMPILER_VERSION, compile_semantic_plan
from .stages import restore
from .store import Store
from .bindings import current_scene
from .hand_objects import normalize
from .compatibility import execution_hash, physical_match


ACTIONS = {"move_base", "operate_arm", "push_with_base", "pull_with_base",
           "carry_with_base"}
COUPLED_ACTIONS = {"push_with_base", "pull_with_base", "carry_with_base"}


def _action(phase: dict) -> str:
    action = phase.get("action")
    if action not in ACTIONS:
        raise ValueError(f"unknown mobile action: {action}")
    grasp = phase.get("grasp_sides", [])
    contact = phase.get("contact_sides", [])
    if action in ("push_with_base", "pull_with_base") and not (grasp or contact):
        raise ValueError(f"{action} requires contact_sides or grasp_sides")
    if action == "carry_with_base" and not grasp:
        raise ValueError("carry_with_base requires grasp_sides")
    if grasp or contact or action in COUPLED_ACTIONS:
        if not phase.get("object") or any(
                side not in ("left", "right") for side in [*grasp, *contact]):
            raise ValueError(f"{action} contact/grasp sides require an object and left/right names")
    if action == "operate_arm" and phase.get("base", {}).get("mode", "stop") != "stop":
        raise ValueError("operate_arm requires base.mode=stop")
    return action


def _canonical_phase(phase: dict) -> dict:
    value = json.loads(json.dumps(phase))
    for key in ("repair_note", "description", "notes"):
        value.pop(key, None)
    for side in ("left", "right"):
        if side not in value:
            value[side] = {"mode": "hold_current"}
        elif value[side].get("mode") == "hold":
            value[side]["mode"] = "hold_current"
    return normalize(value)


def _strategy_advisories(phases: list[dict]) -> list[dict]:
    """Flag likely copied corrections without making the harness task-specific."""
    advisories = []
    released: dict[str, str] = {}
    for phase in phases:
        object_name = phase.get("object")
        intent = phase.get("intent")
        if not object_name:
            continue
        if intent == "release":
            released[object_name] = phase["name"]
            continue
        reconnects = bool(phase.get("grasp_sides") or phase.get("contact_sides") or
                          intent in ("pinch_grasp", "contact_push", "contact_pull"))
        if object_name in released and reconnects:
            if not phase.get("necessity"):
                advisories.append({
                    "code": "redundant_release_recontact",
                    "object": object_name,
                    "release_phase": released[object_name],
                    "recontact_phase": phase["name"],
                    "message": ("remove copied retry or declare necessity such as changed_grasp_region, "
                                "obstacle_clearance, or contact_recovery"),
                })
            released.pop(object_name, None)
    return advisories


def _checkpoint_override(store: Store, state: dict, model: mujoco.MjModel,
                         phases: list[dict], after_phase: str
                         ) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    names = [phase["name"] for phase in phases]
    if after_phase not in names or names.index(after_phase) + 1 >= len(phases):
        raise ValueError("calibrate_after_phase must name a non-final phase")
    index = names.index(after_phase)
    candidate = prior = None
    merge = current_scene(store, state)
    scene_hash = merge["scene_sha256"]
    settled_hash = sha256(store.artifact(merge["settled_state_path"]))
    engine_hash = execution_hash()
    for probe in reversed(state.get("probes", [])):
        if not physical_match(probe, scene_hash, settled_hash, engine_hash):
            continue
        if len(probe.get("checkpoint_paths", [])) <= index:
            continue
        diagnostics = json.loads(store.artifact(probe["diagnostics_path"]).read_text())
        metrics = diagnostics.get("stage_metrics", [])
        # A failed suffix may still list inherited checkpoint paths. Only use a
        # checkpoint when every stage through the calibration boundary actually
        # passed in that probe.
        if len(metrics) <= index or not all(
                metric.get("passed", False) for metric in metrics[:index + 1]):
            continue
        matching = next((trajectory for trajectory in reversed(state["trajectories"])
                         if trajectory.get("stage_hashes") == probe.get("stage_hashes")), None)
        if matching is None or matching.get("compiler_version") != COMPILER_VERSION:
            continue
        prior_phases = json.loads(
            store.artifact(matching["source_spec_path"]).read_text())["phases"]
        if ([_canonical_phase(p) for p in prior_phases[:index + 1]] !=
                [_canonical_phase(p) for p in phases[:index + 1]]):
            continue
        candidate, prior = probe, matching
        break
    if candidate is None or prior is None:
        raise ValueError(
            f"no compatible passing checkpoint after {after_phase}; probe the unchanged prefix in the current scene")
    data = mujoco.MjData(model)
    checkpoint_path = store.artifact(candidate["checkpoint_paths"][index])
    with np.load(checkpoint_path) as checkpoint:
        restore(model, data, checkpoint["integration_state"])
    next_name = phases[index + 1]["name"]
    override = {next_name: {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
                            "ctrl": data.ctrl.copy()}}
    provenance = {"after_phase": after_phase, "next_phase": next_name,
                  "checkpoint_path": str(checkpoint_path.relative_to(store.root)),
                  "probe_diagnostics_path": candidate["diagnostics_path"],
                  "_prefix_index": index, "_prefix_hash": candidate["stage_hashes"][index]}
    return override, provenance


def _checkpoint_overrides(store: Store, state: dict, model: mujoco.MjModel,
                          phases: list[dict], after_phases: list[str]
                          ) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    overrides: dict[str, dict[str, np.ndarray]] = {}
    provenance = []
    for after_phase in after_phases:
        resolved, detail = _checkpoint_override(store, state, model, phases, after_phase)
        overlap = set(overrides) & set(resolved)
        if overlap:
            raise ValueError(f"duplicate checkpoint calibration boundary: {sorted(overlap)}")
        overrides.update(resolved)
        provenance.append(detail)
    return overrides, provenance


def compile_trajectory(input_dir: Path, store: Store, request: dict) -> dict:
    state = store.read()
    if not state["merges"]:
        raise ValueError("merge scene first")
    spec_path = store.artifact(request["trajectory_json_path"])
    spec = json.loads(spec_path.read_text())
    state["pending_trajectory_spec_sha256"] = sha256(spec_path)
    state["stage"] = "TRAJECTORY_COMPILING"
    store.save(state)
    if spec.get("schema_version", "mobile-semantic-action-v1.4") != "mobile-semantic-action-v1.4":
        raise ValueError("v1.4 requires schema_version mobile-semantic-action-v1.4")
    phases = [_canonical_phase(phase) for phase in spec.get("phases", [])]
    for phase in phases:
        _action(phase)
        validate_action_intent(phase)
    compile_through_phase = request.get("compile_through_phase")
    compile_phases = json.loads(json.dumps(phases))
    validated_through_index = None
    if compile_through_phase is not None:
        phase_names = [phase["name"] for phase in phases]
        if compile_through_phase not in phase_names:
            raise ValueError("compile_through_phase must name a trajectory phase")
        validated_through_index = phase_names.index(compile_through_phase)
        for phase in compile_phases[validated_through_index + 1:]:
            phase.pop("motion", None)
            phase["left"] = {"mode": "hold_current"}
            phase["right"] = {"mode": "hold_current"}

    merge = current_scene(store, state)
    scene_path = store.artifact(merge["scene_xml_path"])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    with np.load(store.artifact(merge["settled_state_path"])) as archive:
        settled = {key: np.asarray(archive[key]).copy() for key in ("qpos", "qvel", "ctrl")}
    from mobile_input.loader import load_episode
    loaded = load_episode(input_dir)
    source = {"qpos": loaded.arm_position.copy(), "action": loaded.arm_command.copy(),
              "base_action": loaded.base_twist_body.copy(),
              "timestamp_sec": loaded.timestamp_sec.copy()}
    runtime = dict(loaded.runtime_contract)
    requested_calibrations = request.get("calibrate_after_phases")
    if requested_calibrations is None:
        singular = request.get("calibrate_after_phase")
        requested_calibrations = [singular] if singular else []
    if not isinstance(requested_calibrations, list) or not all(
            isinstance(value, str) for value in requested_calibrations):
        raise ValueError("calibrate_after_phases must be a list of phase names")
    overrides, calibrations = _checkpoint_overrides(
        store, state, model, phases, requested_calibrations)
    calibration = (calibrations[0] if len(calibrations) == 1 else calibrations) or None
    compiled = compile_semantic_plan(
        model, settled, source, compile_phases, runtime, phase_start_overrides=overrides)
    strategy_advisories = _strategy_advisories(phases)
    strategy_advisories.extend({"code": "large_time_scaling", "phase": phase["name"],
                               "scale": scale,
                               "message": "Inspect IK posture/path before accepting a very slow phase."}
                              for phase, scale in zip(phases, compiled.phase_scales) if scale > 3)

    revision = len(state["trajectories"]) + 1
    target_path = store.root / f"simulation/targets_r{revision}.npz"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_path,
        timestamp_sec=compiled.timestamp_sec,
        arm_targets=compiled.arm_targets,
        base_targets=compiled.base_targets,
        base_nominal_xyyaw=compiled.base_nominal_xyyaw,
        phase_index=compiled.phase_index,
        compiler_version=np.asarray(COMPILER_VERSION),
    )
    normalized = {"schema_version": "mobile-semantic-action-v1.4", "phases": phases}
    snapshot_path = store.root / f"simulation/trajectory_spec_r{revision}.json"
    write_json(snapshot_path, normalized)
    stage_hashes = []
    previous_hash = merge["scene_sha256"]
    for phase in phases:
        start, end = int(phase["start_row"]), int(phase["end_row"])
        digest = hashlib.sha256()
        digest.update(COMPILER_VERSION.encode())
        digest.update(str(runtime.get("asset_sha256", "")).encode())
        digest.update(previous_hash.encode())
        digest.update(json.dumps(phase, sort_keys=True).encode())
        digest.update(compiled.arm_targets[start:end].tobytes())
        digest.update(compiled.base_targets[start:end].tobytes())
        digest.update(compiled.timestamp_sec[max(0, start - 1):end].tobytes())
        previous_hash = digest.hexdigest()
        stage_hashes.append(previous_hash)
    for detail in calibrations:
        index = detail.pop("_prefix_index")
        expected = detail.pop("_prefix_hash")
        if stage_hashes[index] != expected:
            raise ValueError(
                f"checkpoint prefix changed through {phases[index]['name']}; "
                "compile without calibration and probe this prefix again")
    diagnostic_path = store.root / f"simulation/trajectory_compile_r{revision}.json"
    write_json(diagnostic_path, {
        "schema_version": COMPILER_VERSION,
        "phase_time_scales": list(compiled.phase_scales),
        "max_ik_position_error_m": compiled.max_ik_position_error_m,
        "max_ik_rotation_error_rad": compiled.max_ik_rotation_error_rad,
        "semantic_expansions": list(compiled.expansion_summary),
        "strategy_advisories": strategy_advisories,
        "checkpoint_calibration": calibration,
        "compile_through_phase": compile_through_phase,
        "source_prior": "qpos/action used only for initial state, IK seed, timing, and gripper prior",
    })
    record = {
        "revision": revision,
        "compiler_version": COMPILER_VERSION,
        "source_spec_path": str(snapshot_path.relative_to(store.root)),
        "source_spec_sha256": sha256(snapshot_path),
        "targets_path": str(target_path.relative_to(store.root)),
        "targets_sha256": sha256(target_path),
        "settled_sha256": merge["settled_sha256"],
        "compile_diagnostics_path": str(diagnostic_path.relative_to(store.root)),
        "scene_sha256": merge["scene_sha256"],
        "phase_count": len(phases),
        "rows": len(compiled.timestamp_sec),
        "duration_sec": float(compiled.timestamp_sec[-1]),
        "max_phase_time_scale": float(max(compiled.phase_scales, default=1)),
        "semantic_expansion_count": len(compiled.expansion_summary),
        "strategy_advisories": strategy_advisories,
        "checkpoint_calibration": calibration,
        "compile_through_phase": compile_through_phase,
        "validated_through_index": validated_through_index,
        "stage_hashes": stage_hashes,
        "actions": [_action(phase) for phase in phases],
        "phase_names": [phase["name"] for phase in phases],
    }
    write_json(store.root / f"simulation/trajectory_r{revision}.json", record)
    state["trajectories"].append(record)
    state.pop("pending_trajectory_spec_sha256", None)
    state["stage"] = "TRAJECTORY_READY"
    store.save(state)
    compact = {key: record[key] for key in (
        "revision", "compiler_version", "phase_count", "rows", "duration_sec",
        "max_phase_time_scale", "semantic_expansion_count", "strategy_advisories",
        "checkpoint_calibration",
        "compile_through_phase",
        "stage_hashes", "phase_names")}
    return {"status": "PASS", **compact}
