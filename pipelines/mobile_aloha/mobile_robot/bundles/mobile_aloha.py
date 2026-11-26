from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mobile_input.writer import sha256, write_json
from mobile_robot.asset import BASE_JOINTS, JOINTS, build_robot_asset, initial_state


def install(episode_root: Path, source_qpos: np.ndarray) -> None:
    runtime = Path(episode_root) / "derived/robot_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    scene = build_robot_asset(runtime / "robot_base_scene.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    qpos, ctrl = initial_state(model, source_qpos)
    np.savez_compressed(runtime / "initial_state.npz", qpos=qpos, ctrl=ctrl)
    contract = {
        "schema_version": "mobile-robot-runtime-v2",
        "embodiment_id": "mobile_aloha_tracer_vx300s",
        "asset_sha256": sha256(scene), "nq": model.nq, "nv": model.nv,
        "nu": model.nu, "ncam": model.ncam,
        "base_body": "tracer/base", "base_pose_joints": list(BASE_JOINTS),
        "base_drive": "planar_velocity_xyyaw",
        "compiler_joint_velocity_limit_radps": 1.0,
        "compiler_joint_acceleration_limit_radps2": 4.0,
        "arms": {side: {
            "joint_names": [f"{side}/{name}" for name in JOINTS],
            "actuator_names": [f"{side}/{name}" for name in JOINTS],
            "gripper_actuator": f"{side}/gripper", "gripper_control_range_m": [.021, .057],
            "end_effector_site": f"{side}/pinch",
            "finger_pad_sites": [],
            "wrist_camera": f"wrist_cam_{side}"} for side in ("left", "right")},
        "cameras": {"top": "top", "left_wrist": "wrist_cam_left",
                    "right_wrist": "wrist_cam_right"},
        "collision_groups": {"base_geom_prefixes": ["tracer/"],
                             "arm_geom_prefixes": ["left/", "right/"],
                             "finger_geom_prefixes": ["left/left_finger", "left/right_finger",
                                                      "right/left_finger", "right/right_finger"]},
        "camera_mount_policy": "fixed_bundle_approximation",
    }
    write_json(runtime / "robot_runtime_contract.json", contract)
    write_json(runtime / "bundle_provenance.json", {
        "embodiment_id": "mobile_aloha_tracer_vx300s",
        "source_repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "source_component": "trossen_vx300s plus project Tracer reconstruction",
        "asset_sha256": sha256(scene),
        "camera_mount_policy": "fixed_bundle_approximation",
        "adaptations": ["three virtual planar joints", "visual non-driving wheels",
                        "mature ALOHA2 actuator control profile", "semantic sites and cameras"]})
