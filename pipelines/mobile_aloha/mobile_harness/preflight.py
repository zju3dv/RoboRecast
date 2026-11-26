"""Read-only geometric and short-contact probes for mobile task scenes."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
from .rendering import presentation_renderer, presentation_options
import numpy as np
from PIL import Image, ImageDraw

from .semantic_trajectory import _solve
from .stages import contact_side, object_geom, relative_pinch
from .store import Store


ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def _loaded_scene(store: Store) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    state = store.read()
    if not state["merges"]:
        raise ValueError("merge scene before preflight")
    merge = state["merges"][-1]
    model = mujoco.MjModel.from_xml_path(str(store.artifact(merge["scene_xml_path"])))
    data = mujoco.MjData(model)
    with np.load(store.artifact(merge["settled_state_path"])) as settled:
        data.qpos[:] = settled["qpos"]
        data.qvel[:] = settled["qvel"]
        data.ctrl[:] = settled["ctrl"]
    mujoco.mj_forward(model, data)
    return model, data, merge


def _collision_pairs(model: mujoco.MjModel, data: mujoco.MjData,
                     body_filter: set[int] | None = None) -> list[dict]:
    pairs = {}
    for contact in data.contact:
        first_body, second_body = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if body_filter is not None and not ((first_body in body_filter) != (second_body in body_filter)):
            continue
        if contact.dist >= 0:
            continue
        names = sorted((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or
                        f"geom_{contact.geom1}",
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or
                        f"geom_{contact.geom2}"))
        key = tuple(names)
        pairs[key] = max(pairs.get(key, 0.0), -float(contact.dist))
    return [{"geoms": list(key), "penetration_m": value}
            for key, value in sorted(pairs.items(), key=lambda row: -row[1])]


def _subtree(model: mujoco.MjModel, root_body: int) -> set[int]:
    result = {root_body}
    for body in range(root_body + 1, model.nbody):
        parent = int(model.body_parentid[body])
        if parent in result:
            result.add(body)
    return result


def articulation_sweep(model: mujoco.MjModel, data: mujoco.MjData,
                       joint_name: str, positions: list[float]) -> dict:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint < 0 or int(model.jnt_type[joint]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ValueError(f"expected a named hinge or slide joint: {joint_name}")
    if len(positions) < 3 or not np.isfinite(positions).all():
        raise ValueError("joint_positions needs at least three finite values")
    if model.jnt_limited[joint]:
        low, high = model.jnt_range[joint]
        if any(value < low - 1e-6 or value > high + 1e-6 for value in positions):
            raise ValueError("joint_positions exceed the modeled joint range")
    address = int(model.jnt_qposadr[joint])
    original = data.qpos.copy()
    affected = _subtree(model, int(model.jnt_bodyid[joint]))
    samples = []
    for value in positions:
        data.qpos[:] = original
        data.qpos[address] = value
        mujoco.mj_forward(model, data)
        pairs = _collision_pairs(model, data, affected)
        samples.append({"joint_position": float(value), "collision_pairs": pairs,
                        "max_penetration_m": max((row["penetration_m"] for row in pairs), default=0.0)})
    data.qpos[:] = original
    mujoco.mj_forward(model, data)
    return {"joint_name": joint_name, "samples": samples,
            "blocking_sample_count": sum(bool(row["collision_pairs"]) for row in samples),
            "advisory_only": True}


def inspect_articulation_sweep(store: Store, request: dict) -> dict:
    model, data, merge = _loaded_scene(store)
    name = request["joint_name"]
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise ValueError(f"joint not found: {name}")
    initial = float(data.qpos[model.jnt_qposadr[joint]])
    if "joint_positions" in request:
        positions = [float(value) for value in request["joint_positions"]]
    else:
        if model.jnt_limited[joint]:
            low, high = map(float, model.jnt_range[joint])
            end = high if abs(high - initial) >= abs(low - initial) else low
        else:
            end = initial + (math.pi / 2 if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE else .2)
        positions = np.linspace(initial, end, int(request.get("sample_count", 7))).tolist()
    result = articulation_sweep(model, data, name, positions)
    if request.get("render", True):
        selected = [positions[0], positions[len(positions) // 2], positions[-1]]
        output = store.root / f"simulation/preflight/articulation_{name.replace('/', '_')}_scene{merge['revision']}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (3 * 320, 240), "black")
        draw = ImageDraw.Draw(canvas)
        renderer = presentation_renderer(model, width=320, height=240)
        try:
            for column, value in enumerate(selected):
                data.qpos[model.jnt_qposadr[joint]] = value
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="top", scene_option=presentation_options())
                canvas.paste(Image.fromarray(renderer.render().copy()), (column * 320, 0))
                draw.text((column * 320 + 6, 6), f"{name}: {value:.2f}", fill="yellow")
        finally:
            renderer.close()
        canvas.save(output, quality=85)
        result["preview_path"] = str(output.relative_to(store.root))
    return {"status": "PASS", **result}


def _arm_joint_addresses(model: mujoco.MjModel, side: str) -> list[int]:
    addresses = []
    legacy = [f"{side}/{name}" for name in ARM_JOINTS]
    generic = [f"{side}/joint{index}" for index in range(1, 7)]
    names = legacy if mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, legacy[0]) >= 0 else generic
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"robot arm joint missing: {name}")
        addresses.append(int(model.jnt_qposadr[joint]))
    return addresses


def reachability(model: mujoco.MjModel, data: mujoco.MjData, side: str,
                 target_position_m: list[float], base_xyyaw: list[float] | None = None) -> dict:
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    target = np.asarray(target_position_m, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_position_m must have three finite numbers")
    if base_xyyaw is not None:
        if len(base_xyyaw) != 3:
            raise ValueError("base_xyyaw must have three numbers")
        names = (["base/x", "base/y", "base/yaw"] if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "base/x") >= 0 else
            ["tracer/x", "tracer/y", "tracer/yaw"])
        addresses = [int(model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in names]
        data.qpos[addresses] = base_xyyaw
    mujoco.mj_forward(model, data)
    addresses = _arm_joint_addresses(model, side)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/pinch")
    if site < 0:
        raise ValueError(f"pinch site missing for {side}")
    try:
        _solve(model, data, side, target, None, position_tolerance=.03,
               max_step_from_reference=None)
    except ValueError:
        pass
    error = float(np.linalg.norm(data.site_xpos[site] - target))
    pairs = _collision_pairs(model, data)
    joint_names = ([f"{side}/{name}" for name in ARM_JOINTS]
                   if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{side}/{ARM_JOINTS[0]}") >= 0 else
                   [f"{side}/joint{index}" for index in range(1, 7)])
    margins = [float(min(data.qpos[address] - model.jnt_range[joint, 0],
                         model.jnt_range[joint, 1] - data.qpos[address]))
               for address, joint in zip(addresses, [mujoco.mj_name2id(
                   model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])]
    return {"side": side, "target_error_m": error, "reachable": error <= .03,
            "minimum_joint_limit_margin_rad": min(margins),
            "collision_pairs": pairs[:12], "advisory_only": True}


def probe_reachability(store: Store, request: dict) -> dict:
    model, data, _ = _loaded_scene(store)
    result = reachability(model, data, request["side"], request["target_position_m"],
                          request.get("base_xyyaw"))
    return {"status": "PASS", **result}


def probe_contact(store: Store, request: dict) -> dict:
    model, data, _ = _loaded_scene(store)
    side, object_name = request["side"], request["object"]
    reach = reachability(model, data, side, request["target_position_m"],
                         request.get("base_xyyaw"))
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}/gripper")
    if gripper < 0:
        raise ValueError(f"gripper actuator missing for {side}")
    opening = float(request.get("gripper_opening", .1))
    if not 0 <= opening <= 1:
        raise ValueError("gripper_opening must be in [0,1]")
    low, high = model.actuator_ctrlrange[gripper]
    data.ctrl[gripper] = float(low + (high - low) * opening)
    duration = float(request.get("duration_sec", .3))
    if not .01 <= duration <= 1.0:
        raise ValueError("duration_sec must be in [.01,1.0]")
    before = relative_pinch(model, data, object_name, side)
    joint_names = ([f"{side}/{name}" for name in ARM_JOINTS]
                   if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{side}/{ARM_JOINTS[0]}") >= 0 else
                   [f"{side}/joint{index}" for index in range(1, 7)])
    for full_name, address in zip(joint_names, _arm_joint_addresses(model, side)):
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    full_name)
        if actuator >= 0:
            data.ctrl[actuator] = data.qpos[address]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    counts = {"arm": 0, "hand": 0, "left_pad": 0, "right_pad": 0}
    pinch_pair_substeps = 0
    steps = max(1, math.ceil(duration / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        touched = set()
        for contact in data.contact:
            for robot_geom, task_geom in ((contact.geom1, contact.geom2),
                                          (contact.geom2, contact.geom1)):
                role = contact_side(model, robot_geom)
                if role and role[0] == side and object_geom(model, task_geom, object_name):
                    touched.add(role[1])
        for part in touched:
            counts[part] += 1
        pinch_pair_substeps += int({"left_pad", "right_pad"} <= touched)
    slip = float(np.linalg.norm(relative_pinch(model, data, object_name, side) - before))
    return {"status": "PASS", "advisory_only": True, "reachability": reach,
            "contact_substeps": counts, "pinch_pair_substeps": pinch_pair_substeps,
            "pinch_pair_observed": bool(pinch_pair_substeps), "relative_slip_m": slip,
            "duration_sec": duration}
