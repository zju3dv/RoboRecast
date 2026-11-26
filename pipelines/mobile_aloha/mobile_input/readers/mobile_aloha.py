from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mobile_input.schema import CameraStream, CanonicalEpisodeV2, integrate_body_twist
from mobile_input.writer import sha256
from .common import (HDF5FrameReader, dataset_inventory,
                     decode_jpeg_bgr_payload_as_source_rgb)


CAMERAS = {"top": "observations/images/cam_high",
           "left_wrist": "observations/images/cam_left_wrist",
           "right_wrist": "observations/images/cam_right_wrist"}


def read_mobile_aloha(source_path: Path, episode_id: str, task_name: str,
                      source_url: str = "") -> CanonicalEpisodeV2:
    source_path = Path(source_path).resolve()
    with h5py.File(source_path, "r") as source:
        qpos = np.asarray(source["observations/qpos"], np.float32)
        qvel = np.asarray(source["observations/qvel"], np.float32)
        command = np.asarray(source["action"], np.float32)
        base = np.asarray(source["base_action"], np.float32)
        rows = len(qpos)
        if qpos.shape != (rows, 14) or qvel.shape != qpos.shape or command.shape != qpos.shape:
            raise ValueError("Mobile ALOHA arm arrays must be aligned [T,14]")
        if base.shape != (rows, 2):
            raise ValueError("Mobile ALOHA base_action must be [T,2]")
        for key in CAMERAS.values():
            if len(source[key]) != rows:
                raise ValueError(f"camera row mismatch: {key}")
        inventory = dataset_inventory(source)
        attrs = {key: str(value) for key, value in source.attrs.items()}
        first_shape = decode_jpeg_bgr_payload_as_source_rgb(source[CAMERAS["top"]][0]).shape
    timestamps = np.arange(rows, dtype=np.float64) / 50.0
    twist = np.column_stack((base[:, 0], np.zeros(rows, np.float32), base[:, 1])).astype(np.float32)
    route = integrate_body_twist(timestamps, twist)
    cameras = {
        role: CameraStream(role, rows,
                           HDF5FrameReader(source_path, key,
                                           decode_jpeg_bgr_payload_as_source_rgb),
                           key, (int(first_shape[1]), int(first_shape[0])), "jpeg_bytes",
                           {"color_transform": "opencv_misencoded_source_rgb_bytes"})
        for role, key in CAMERAS.items()
    }
    dt = np.diff(timestamps)
    return CanonicalEpisodeV2(
        episode_id, "mobile_aloha", source_path, task_name,
        "mobile_aloha_tracer_vx300s", timestamps, qpos, qvel, command, twist, route, cameras,
        {"dataset": "Mobile ALOHA", "source_url": source_url,
         "source_path": str(source_path), "source_sha256": sha256(source_path),
         "source_size_bytes": source_path.stat().st_size, "source_attrs": attrs},
        {"status": "PASS", "resolved_paths": {
            "arm_position": "observations/qpos", "arm_velocity": "observations/qvel",
            "arm_command": "action", "base_twist": "base_action", "cameras": CAMERAS},
         "inventory": inventory, "base_action_order": ["forward_mps", "yaw_rate_radps"],
         "arm_order": "left_6,left_gripper,right_6,right_gripper"},
        {"timestamp_source": "fixed_control_rate", "control_hz": 50,
         "rows": rows, "median_dt_sec": float(np.median(dt))},
        {"status": "approximate", "policy": "fixed_bundle_mounts",
         "depth_world_projection_allowed": False},
        {"method": "midpoint_integration_of_body_twist", "is_ground_truth": False,
         "path_length_m": float(np.sum(np.abs(base[:-1, 0]) * dt)),
         "net_xyyaw": route[-1].tolist()},
        {"task": task_name, "status": "source_folder_prior_not_video_ground_truth",
         "scope": "robot_path_manipulated_objects_and_required_supports_only"},
        {"embodiment_id": "mobile_aloha_tracer_vx300s",
         "arm_channels": "left_6,left_gripper,right_6,right_gripper",
         "gripper_normalization": {"source": "already_normalized", "closed": 0, "open": 1}})
