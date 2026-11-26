from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .loader import load_episode
from .writer import sha256


def _named(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> bool:
    return bool(name) and mujoco.mj_name2id(model, kind, str(name)) >= 0


def _validate_runtime(model: mujoco.MjModel, contract: dict, failures: list[str]) -> None:
    base_joints = contract.get("base_pose_joints", [])
    if len(base_joints) != 3 or not all(
            _named(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in base_joints):
        failures.append("runtime base_pose_joints")
    if contract.get("base_drive") != "planar_velocity_xyyaw":
        failures.append("runtime base_drive")
    base_body = contract.get("base_body")
    if base_body and not _named(model, mujoco.mjtObj.mjOBJ_BODY, base_body):
        failures.append("runtime base_body")
    for side in ("left", "right"):
        arm = contract.get("arms", {}).get(side, {})
        if len(arm.get("joint_names", [])) != 6 or not all(
                _named(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in arm.get("joint_names", [])):
            failures.append(f"runtime {side} joints")
        if len(arm.get("actuator_names", [])) != 6 or not all(
                _named(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in arm.get("actuator_names", [])):
            failures.append(f"runtime {side} actuators")
        if not _named(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                      arm.get("gripper_actuator", "")):
            failures.append(f"runtime {side} gripper")
        if not _named(model, mujoco.mjtObj.mjOBJ_SITE, arm.get("end_effector_site", "")):
            failures.append(f"runtime {side} end effector")
    for role in ("top", "left_wrist", "right_wrist"):
        name = contract.get("cameras", {}).get(role, "")
        if not _named(model, mujoco.mjtObj.mjOBJ_CAMERA, name):
            failures.append(f"runtime camera {role}")


def validate_episode(episode_dir: Path) -> dict:
    episode_dir = Path(episode_dir).resolve()
    failures: list[str] = []
    warnings: list[str] = []
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    for item in manifest.get("files", []):
        path = episode_dir / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            failures.append(f"file hash: {item['path']}")
    loaded = load_episode(episode_dir)
    if np.any(np.diff(loaded.timestamp_sec) <= 0):
        failures.append("timestamps not monotonic")
    expected = (loaded.rows, 14)
    for name, array in (("arm_position", loaded.arm_position),
                        ("arm_velocity", loaded.arm_velocity),
                        ("arm_command", loaded.arm_command)):
        if array.shape != expected or not np.isfinite(array).all():
            failures.append(f"{name} shape or values")
    if loaded.base_twist_body.shape != (loaded.rows, 3):
        failures.append("base_twist_body shape")
    if loaded.route_prior_xyyaw.shape != (loaded.rows, 3):
        failures.append("route prior shape")
    frame_counts = {}
    for role, path in loaded.camera_paths.items():
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = capture.read()
        capture.release()
        frame_counts[role] = count
        if not ok or count != loaded.rows:
            failures.append(f"{role} video: {count} frames")
    model = mujoco.MjModel.from_xml_path(str(loaded.runtime_scene))
    _validate_runtime(model, dict(loaded.runtime_contract), failures)
    with np.load(loaded.initial_state) as initial:
        if initial["qpos"].shape != (model.nq,) or initial["ctrl"].shape != (model.nu,):
            failures.append("initial state dimensions")
    calibration = json.loads((episode_dir / "specs/camera_calibration.json").read_text())
    if calibration.get("status") in ("missing", "approximate"):
        warnings.append(f"camera calibration is {calibration.get('status')}")
    dt = np.diff(loaded.timestamp_sec)
    speed = np.linalg.norm(loaded.base_twist_body[:-1, :2], axis=1)
    travel = float(np.sum(speed * dt))
    return {
        "schema_version": "mobile-real2sim-validation-v2",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "rows": loaded.rows,
            "duration_sec": float(loaded.timestamp_sec[-1] - loaded.timestamp_sec[0]),
            "path_length_prior_m": travel,
            "net_route_prior_xyyaw": loaded.route_prior_xyyaw[-1].tolist(),
            "video_frame_counts": frame_counts,
            "robot_nq": model.nq,
            "robot_nv": model.nv,
            "robot_nu": model.nu,
            "robot_ncam": model.ncam,
        },
    }
