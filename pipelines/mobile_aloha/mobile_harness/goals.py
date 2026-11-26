"""Episode-declared action intent and terminal task goals."""

from __future__ import annotations

import math

import mujoco
import numpy as np


TERMINAL_TYPES = {"joint_final_range", "terminal_body_relation",
                  "terminal_support_contact", "terminal_contact_absent"}


def validate_action_intent(phase: dict) -> None:
    intent = phase.get("intent")
    if intent is None:
        return
    action = phase.get("action")
    contact = phase.get("contact_sides", [])
    grasp = phase.get("grasp_sides", [])
    if intent == "contact_push":
        valid = action in ("push_with_base", "operate_arm") and bool(contact) and not grasp
    elif intent == "contact_pull":
        valid = action in ("pull_with_base", "operate_arm") and bool(contact) and not grasp
    elif intent in ("pinch_grasp", "supported_carry"):
        valid = bool(grasp) and action in (
            "operate_arm", "push_with_base", "pull_with_base", "carry_with_base")
    elif intent == "release":
        valid = not grasp and action == "operate_arm"
    else:
        raise ValueError(f"unknown action intent: {intent}")
    if not valid:
        raise ValueError(f"{intent} is incompatible with action/contact/grasp fields")


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    identifier = mujoco.mj_name2id(model, kind, name)
    if identifier < 0:
        raise ValueError(f"terminal goal name not found: {name}")
    return identifier


def _contact_rows(model: mujoco.MjModel, qpos: np.ndarray, window: int,
                  prefix_a: str, prefix_b: str) -> int:
    data = mujoco.MjData(model)
    hits = 0
    for state in qpos[-window:]:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        if any((a.startswith(prefix_a) and b.startswith(prefix_b)) or
               (a.startswith(prefix_b) and b.startswith(prefix_a))
               for contact in data.contact
               for a, b in ((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                               contact.geom1) or "",
                             mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                               contact.geom2) or ""),)):
            hits += 1
    return hits


def evaluate_terminal_goals(model: mujoco.MjModel, qpos: np.ndarray,
                            checks: list[dict]) -> list[dict]:
    """Evaluate only terminal checks against actual simulated states."""
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) == 0:
        raise ValueError("terminal qpos must be a nonempty [T,nq] array")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[-1]
    mujoco.mj_forward(model, data)
    results = []
    for check in checks:
        kind = check["type"]
        if kind not in TERMINAL_TYPES:
            continue
        if kind == "joint_final_range":
            joint = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, check["joint"])
            observed = float(data.qpos[model.jnt_qposadr[joint]])
            low = float(check.get("min_rad", -math.inf))
            high = float(check.get("max_rad", math.inf))
            passed = low <= observed <= high
        elif kind == "terminal_body_relation":
            subject = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, check["subject"])
            target = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, check["target"])
            relation = check["relation"]
            if relation == "near":
                if "subject_geom" in check and "target_geom" in check:
                    geom_a = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, check["subject_geom"])
                    geom_b = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, check["target_geom"])
                    fromto = np.zeros(6)
                    observed = float(mujoco.mj_geomDistance(
                        model, data, geom_a, geom_b, 10.0, fromto))
                    maximum = float(check["max_distance_m"])
                else:
                    observed = float(np.linalg.norm(data.xpos[subject, :2] - data.xpos[target, :2]))
                    maximum = float(check["max_xy_distance_m"])
                passed = observed <= maximum
            elif relation == "inside_xy":
                geom = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, check["target_geom"])
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                    raise ValueError("inside_xy target_geom must be a box")
                local = data.geom_xmat[geom].reshape(3, 3).T @ (
                    data.xpos[subject] - data.geom_xpos[geom])
                margin = float(check.get("margin_m", 0.0))
                observed = np.abs(local[:2]).tolist()
                passed = bool(np.all(np.abs(local[:2]) <= model.geom_size[geom, :2] + margin))
            elif relation == "above":
                observed = float(data.xpos[subject, 2] - data.xpos[target, 2])
                passed = observed >= float(check["min_z_difference_m"])
            else:
                raise ValueError(f"unknown terminal relation: {relation}")
        else:
            window = int(check.get("window_rows", 10))
            if window < 1 or window > len(qpos):
                raise ValueError("terminal window_rows is out of range")
            if kind == "terminal_support_contact":
                first, second = check["object_geom_prefix"], check["support_geom_prefix"]
                observed = _contact_rows(model, qpos, window, first, second)
                passed = observed >= int(check.get("min_rows", 1))
            else:
                first, second = check["group_a_prefix"], check["group_b_prefix"]
                observed = _contact_rows(model, qpos, window, first, second)
                passed = observed <= int(check.get("max_rows", 0))
        results.append({"type": kind, "passed": bool(passed), "observed": observed,
                        "target": {key: value for key, value in check.items() if key != "type"}})
    return results
