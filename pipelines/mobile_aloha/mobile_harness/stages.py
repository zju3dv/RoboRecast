from __future__ import annotations

import math

import mujoco
import numpy as np


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def restore(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    if len(state) != mujoco.mj_stateSize(model, STATE_SPEC):
        raise ValueError("checkpoint does not match scene")
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    warmstart = data.qacc_warmstart.copy()
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = warmstart


def body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise ValueError(f"missing task body: {name}")
    return result


def object_geom(model: mujoco.MjModel, geom: int, body_name: str) -> bool:
    body = int(model.geom_bodyid[geom])
    while body > 0:
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) == body_name:
            return True
        body = int(model.body_parentid[body])
    return False


def contact_side(model: mujoco.MjModel, geom: int) -> tuple[str, str] | None:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
    for side in ("left", "right"):
        if name.startswith(f"{side}/left_finger_pad"):
            return side, "left_pad"
        if name.startswith(f"{side}/right_finger_pad"):
            return side, "right_pad"
        body = int(model.geom_bodyid[geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if body_name in (f"{side}/gripper_link", f"{side}/gripper_prop_link",
                         f"{side}/left_finger_link", f"{side}/right_finger_link",
                         f"{side}/left_finger", f"{side}/right_finger", f"{side}/eef"):
            return side, "hand"
        if body_name.startswith(f"{side}/"):
            return side, "arm"
    return None


def relative_pinch(model: mujoco.MjModel, data: mujoco.MjData,
                   body_name: str, side: str) -> np.ndarray:
    body = body_id(model, body_name)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/pinch")
    rotation = data.xmat[body].reshape(3, 3)
    return rotation.T @ (data.site_xpos[site] - data.xpos[body])


def object_tilt_deg(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    body = body_id(model, body_name)
    z = data.xmat[body].reshape(3, 3)[:, 2]
    return math.degrees(math.acos(float(np.clip(z[2], -1, 1))))


def full_task_object_quality(model: mujoco.MjModel, qpos: np.ndarray,
                             phase_index: np.ndarray, phases: list[dict]) -> list[dict]:
    """Track manipulated free bodies through release, transfer, and task end."""
    tracked: dict[str, tuple[int, float]] = {}
    for phase in phases:
        name = phase.get("object")
        action = phase.get("action")
        if not name or action not in ("push_with_base", "pull_with_base", "carry_with_base"):
            continue
        if action == "carry_with_base" and "global_max_tilt_deg" not in phase:
            continue
        limit = float(phase.get("global_max_tilt_deg", 20))
        previous = tracked.get(name)
        tracked[name] = (min(previous[0], phase["start_row"]) if previous else phase["start_row"],
                         min(previous[1], limit) if previous else limit)
    for phase in phases:
        for binding in phase.get("hand_objects", {}).values():
            if "global_max_tilt_deg" not in binding:
                continue
            name, limit = binding["object"], float(binding["global_max_tilt_deg"])
            previous = tracked.get(name)
            tracked[name] = (min(previous[0], phase["start_row"]) if previous else phase["start_row"],
                             min(previous[1], limit) if previous else limit)
    results = []
    for name, (start, limit) in tracked.items():
        body = body_id(model, name)
        joint = int(model.body_jntadr[body])
        if model.body_jntnum[body] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        address = int(model.jnt_qposadr[joint])
        quat = qpos[start:, address + 3:address + 7]
        z_axis_z = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        angles = np.degrees(np.arccos(np.clip(z_axis_z, -1, 1)))
        peak = int(np.argmax(angles))
        bad = np.flatnonzero(angles > limit)
        bad_row = start + int(bad[0]) if len(bad) else None
        results.append({"object": name, "limit_deg": limit,
                        "max_tilt_deg": float(angles[peak]),
                        "max_tilt_row": start + peak,
                        "terminal_tilt_deg": float(angles[-1]),
                        "first_bad_row": bad_row,
                        "first_bad_phase": (phases[int(phase_index[bad_row])]["name"]
                                            if bad_row is not None else None),
                        "passed": bad_row is None})
    return results


def overview_camera(model: mujoco.MjModel, data: mujoco.MjData,
                    focus_body: str | None = None) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    base_name = next((name for name in ("base", "tracer/base")
                      if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0), None)
    if base_name is None:
        raise ValueError("runtime base body is missing")
    base = data.xpos[body_id(model, base_name)].copy()
    point = base.copy()
    if not focus_body:
        candidates = []
        for joint in range(model.njnt):
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body = int(model.jnt_bodyid[joint])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            distance = float(np.linalg.norm(data.xpos[body, :2] - base[:2]))
            if name and distance < 2.2:
                candidates.append((distance, name))
        if candidates:
            focus_body = min(candidates)[1]
    if focus_body:
        target = data.xpos[body_id(model, focus_body)]
        distance = float(np.linalg.norm(target[:2] - base[:2]))
        if distance < 2.2:
            point = .55 * base + .45 * target
    point[2] = .55
    camera.lookat[:] = point
    camera.distance = 2.25 if focus_body else 2.55
    camera.azimuth = 105
    camera.elevation = -55
    return camera
