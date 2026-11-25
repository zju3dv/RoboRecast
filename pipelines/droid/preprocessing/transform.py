from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .config import EpisodeSelection
from .robot_runtime import build_robot_runtime
from .image_geometry import proxy_camera_geometry
from .util import ffprobe_video, read_json, sha256_file, tree_sha256, write_json


CAMERA_AXIS_OPENCV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0, 1.0])
GRIPPER_MAX_WIDTH_M = 0.085


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def pose6_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    result = np.broadcast_to(np.eye(4), pose.shape[:-1] + (4, 4)).copy()
    result[..., :3, :3] = Rotation.from_euler("xyz", pose[..., 3:6]).as_matrix()
    result[..., :3, 3] = pose[..., :3]
    return result


def continuous_quaternions(euler_xyz: np.ndarray) -> np.ndarray:
    quat = Rotation.from_euler("xyz", euler_xyz).as_quat()
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    return quat


def camera_intrinsics(entry: Dict[str, Any]) -> Dict[str, Any]:
    fx, cx, fy, cy = [float(x) for x in entry["cameraMatrix"]]
    return {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": int(entry["width"]), "height": int(entry["height"]),
        "projection_model": "pinhole",
        "distortion_coefficients": [float(x) for x in entry.get("distCoeffs", [])],
        "input_video_is_undistorted": True,
        "apply_distortion_during_rendering": False,
    }


def create_proxy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + ".part.mp4")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-an", "-vf",
        "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,settb=1/60,setpts=N",
        "-r", "60", "-c:v", "libx264", "-preset", "fast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-metadata:s:v:0", "rotate=0", "-movflags", "+faststart",
        str(tmp),
    ]
    import subprocess
    subprocess.run(cmd, check=True)
    os.replace(tmp, destination)


def ensure_robot_link(episode_dir: Path, shared_fr3: Path) -> Path:
    robot_dir = episode_dir / "raw" / "robot"
    robot_dir.mkdir(parents=True, exist_ok=True)
    link = robot_dir / shared_fr3.name
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == shared_fr3.resolve():
            return link
        raise FileExistsError(f"Unexpected robot asset at {link}")
    link.symlink_to(os.path.relpath(shared_fr3, link.parent), target_is_directory=True)
    return link


def _metadata_file(sample_dir: Path) -> Path:
    paths = list(sample_dir.glob("metadata_*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one metadata_*.json in {sample_dir}")
    return paths[0]


def _file_entry(path: Path, episode_dir: Path) -> Dict[str, Any]:
    return {
        "path": path.relative_to(episode_dir).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _build_manifest(
    episode_dir: Path,
    metadata: Dict[str, Any],
    selection: EpisodeSelection,
    shared_fr3: Path,
) -> None:
    files = []
    for path in sorted(episode_dir.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name not in {
            "manifest.json", "validation_report.json"
        }:
            files.append(_file_entry(path, episode_dir))
    runtime_manifest_path = episode_dir / "derived/robot_runtime/build_manifest.json"
    runtime_entry = None
    if runtime_manifest_path.is_file():
        runtime_entry = {"path": runtime_manifest_path.relative_to(episode_dir).as_posix(),
                         "sha256": sha256_file(runtime_manifest_path),
                         "contract": read_json(runtime_manifest_path)}
    manifest = {
        "schema_version": "real2sim-input-manifest-v2",
        "generated_at_utc": now_utc(),
        "episode_id": selection.episode_id,
        "dataset_root": ".",
        "episode": {
            "success": bool(metadata["success"]),
            "robot_family": "Franka Research 3 (FR3)" if shared_fr3.name == "FR3" else "Franka Emika Panda",
            "robot_serial": metadata["robot_serial"],
            "scene_id": int(metadata["scene_id"]),
            "task_semantics_status": "unlabeled_agent_must_infer_from_video",
            "task_semantics_output": "intent.json",
        },
        "raw_data_policy": {
            "immutable": True,
            "agent_may_modify_raw": False,
            "hash_algorithm": "sha256",
            "task_language_annotations_included": False,
        },
        "shared_assets": {
            "robot": {
                "family": shared_fr3.name.lower(),
                "episode_link": "raw/robot/" + shared_fr3.name,
                "resolved_path_from_input_root": "shared/robot_assets/" + shared_fr3.name,
                "tree_sha256": tree_sha256(shared_fr3),
                "kinova_is_available_but_not_selected": True,
            }
        },
        "robot_runtime": runtime_entry,
        "files": files,
    }
    write_json(episode_dir / "manifest.json", manifest)


def convert_episode(
    selection: EpisodeSelection,
    samples_root: Path,
    input_root: Path,
    robot_family_override: str | None = None,
) -> Path:
    sample_dir = samples_root / selection.source_folder_name
    episode_dir = input_root / "episodes" / selection.episode_id
    metadata_path = _metadata_file(sample_dir)
    metadata = read_json(metadata_path)
    normalized_robot = str(metadata.get("robot_serial", "")).lower()
    family = "FR3" if normalized_robot.startswith("fr3-") else "Panda" if normalized_robot.startswith("panda-") else None
    if robot_family_override is not None:
        if robot_family_override not in ('Panda', 'FR3') or (family is not None and family != robot_family_override):
            raise ValueError('Robot identity override conflicts with explicit metadata')
        family = robot_family_override
    if family is None:
        raise ValueError(f"Episode lacks explicit FR3/Panda identity: {metadata.get('robot_serial')}")
    shared_fr3 = input_root / "shared/robot_assets" / family
    if not shared_fr3.is_dir():
        raise FileNotFoundError(shared_fr3)
    if episode_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing episode: {episode_dir}")
    if not metadata.get("success"):
        raise ValueError(f"Episode is not successful: {selection.episode_id}")
    if metadata.get("uuid") != selection.episode_id:
        raise ValueError("Episode ID does not match metadata UUID")

    catalog = samples_root / "_catalog"
    intrinsics_all = read_json(catalog / "intrinsics.json")
    extrinsics_all = read_json(catalog / "cam2base_extrinsic_superset.json")
    intrinsics = intrinsics_all.get(selection.episode_id)
    extrinsics = extrinsics_all.get(selection.episode_id)
    if intrinsics is None or extrinsics is None:
        raise ValueError("Selected episode lacks complete post-hoc calibration")

    episode_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["raw/videos", "raw/actions", "raw/calibration", "raw/metadata",
                "derived/videos", "derived/calibration", "specs"]:
        (episode_dir / sub).mkdir(parents=True, exist_ok=True)
    for stale in [
        episode_dir / "raw" / "metadata" / "episode_metadata.json",
        episode_dir / "raw" / "metadata" / "language_annotations.json",
        episode_dir / "specs" / "semantic_spec.json",
    ]:
        stale.unlink(missing_ok=True)
    write_json(episode_dir / "raw" / "metadata" / "episode_metadata_sanitized.json", {
        "schema_version": "droid-agent-metadata-v1",
        "episode_id": metadata["uuid"],
        "success": bool(metadata["success"]),
        "robot_serial": metadata["robot_serial"],
        "scene_id": int(metadata["scene_id"]),
        "camera_serials": {
            "wrist": str(metadata["wrist_cam_serial"]),
            "ext1": str(metadata["ext1_cam_serial"]),
            "ext2": str(metadata["ext2_cam_serial"]),
        },
        "source_metadata_sha256": sha256_file(metadata_path),
        "task_or_language_fields_removed": True,
    })
    ensure_robot_link(episode_dir, shared_fr3)

    role_to_serial = {
        "wrist": str(metadata["wrist_cam_serial"]),
        "ext1": str(metadata["ext1_cam_serial"]),
        "ext2": str(metadata["ext2_cam_serial"]),
    }
    for role, serial in role_to_serial.items():
        source = sample_dir / "recordings" / "MP4" / f"{serial}.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, episode_dir / "raw" / "videos" / f"{role}.mp4")

    h5_path = sample_dir / "trajectory.h5"
    with h5py.File(h5_path, "r") as h5:
        cartesian = h5["action/cartesian_position"][:].astype(np.float64)
        gripper = h5["action/gripper_position"][:].astype(np.float64)
        timestamps_ms = h5["observation/timestamp/control/control_start"][:].astype(np.int64)
        action = np.concatenate([cartesian, gripper[:, None]], axis=1)
        n = len(action)
        joint_position = h5["action/joint_position"][:].astype(np.float64)
        joint_velocity = h5["action/joint_velocity"][:].astype(np.float64)
        cartesian_velocity = h5["action/cartesian_velocity"][:].astype(np.float64)

        wrist = role_to_serial["wrist"]
        camera_pose6 = h5[f"observation/camera_extrinsics/{wrist}_left"][:].astype(np.float64)
        mount_pose6 = h5[f"observation/camera_extrinsics/{wrist}_left_gripper_offset"][:].astype(np.float64)
        camera_ts_ms = h5[f"observation/timestamp/cameras/{wrist}_estimated_capture"][:].astype(np.int64)
        observed_ee = h5["observation/robot_state/cartesian_position"][:].astype(np.float64)

    arrays = [cartesian, gripper, timestamps_ms, joint_position, joint_velocity,
              cartesian_velocity, camera_pose6, mount_pose6, camera_ts_ms, observed_ee]
    if any(len(x) != n for x in arrays):
        raise ValueError("HDF5 arrays do not share one trajectory length")

    raw_video_info = {role: ffprobe_video(episode_dir / "raw" / "videos" / f"{role}.mp4")
                      for role in role_to_serial}
    frame_counts = {role: int(info.get("nb_read_frames", info.get("nb_frames", -1)))
                    for role, info in raw_video_info.items()}
    if set(frame_counts.values()) != {n - 1}:
        raise ValueError(f"Expected N-1 frames for all videos; N={n}, frames={frame_counts}")

    atomic_savez(
        episode_dir / "raw" / "actions" / "actions.npz",
        action=action,
        timestamp_ms=timestamps_ms,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        cartesian_velocity=cartesian_velocity,
    )
    write_json(episode_dir / "raw" / "actions" / "actions.json", {
        "trajectory_length": n,
        "action_shape": [n, 7],
        "fields": ["cartesian_position[6]", "gripper_position[1]"],
        "cartesian_position": "absolute xyz meters + extrinsic xyz Euler radians",
        "gripper_position": "normalized closure; 0 open, 1 closed",
        "timestamp_ms": "observation/timestamp/control/control_start",
        "source_hdf5": "trajectory.h5",
    })

    timestamp_sec = (timestamps_ms - timestamps_ms[0]).astype(np.float64) / 1000.0
    quat = continuous_quaternions(cartesian[:, 3:6])
    closure = np.clip(gripper, 0.0, 1.0)
    valid_action = np.isfinite(action).all(axis=1)
    valid_action &= np.r_[True, np.diff(timestamps_ms) > 0]
    atomic_savez(
        episode_dir / "derived" / "actions_canonical.npz",
        timestamp_sec=timestamp_sec,
        position_m=cartesian[:, :3],
        quaternion_xyzw=quat,
        gripper_width_m=GRIPPER_MAX_WIDTH_M * (1.0 - closure),
        source_index=np.arange(n, dtype=np.int64),
        valid_mask=valid_action,
        timestamp_unix_sec=timestamps_ms.astype(np.float64) / 1000.0,
        gripper_closure=closure,
        metadata_json=np.array(json.dumps({
            "euler_order": "xyz", "euler_composition": "extrinsic_fixed_axes",
            "translation_unit": "meter", "rotation_unit": "radian",
            "reference_frame": "fr3_robot_base", "controlled_frame": "panda_link8",
        })),
    )

    source_index = np.arange(n, dtype=np.int64)
    video_index = source_index.copy()
    video_index[-1] = -1
    atomic_savez(
        episode_dir / "raw" / "calibration" / "wrist_extrinsics_raw.npz",
        camera_pose_in_fr3_base_xyz_euler=camera_pose6,
        gripper_to_camera_offset_xyz_euler=mount_pose6,
        observed_ee_pose_xyz_euler=observed_ee,
        camera_timestamp_estimated_capture_ms=camera_ts_ms,
        control_timestamp_ms=timestamps_ms,
        source_timestep_index=source_index,
        video_frame_index=video_index,
        valid_mask=np.isfinite(camera_pose6).all(axis=1) & np.isfinite(mount_pose6).all(axis=1),
    )

    t_base_camera = pose6_to_matrix(camera_pose6)
    t_camera_base = np.linalg.inv(t_base_camera)
    t_mount = pose6_to_matrix(mount_pose6)
    t_base_mj = t_base_camera @ CAMERA_AXIS_OPENCV_TO_MUJOCO
    camera_timestamp_sec = (camera_ts_ms - timestamps_ms[0]).astype(np.float64) / 1000.0
    mount_translation_spread = float(np.max(np.linalg.norm(t_mount[:, :3, 3] - t_mount[0, :3, 3], axis=1)))
    mount_rotation_spread = float(np.max(Rotation.from_matrix(t_mount[0, :3, :3].T @ t_mount[:, :3, :3]).magnitude()))
    fixed_mount = t_mount[0]
    wrist_valid = np.isfinite(t_base_camera).all(axis=(1, 2)) & np.isfinite(t_mount).all(axis=(1, 2))
    atomic_savez(
        episode_dir / "derived" / "calibration" / "wrist_extrinsics_canonical.npz",
        timestamp_sec=camera_timestamp_sec,
        camera_timestamp_from_first_capture_sec=(camera_ts_ms - camera_ts_ms[0]).astype(np.float64) / 1000.0,
        control_timestamp_sec=timestamp_sec,
        capture_to_control_latency_sec=(timestamps_ms - camera_ts_ms).astype(np.float64) / 1000.0,
        T_fr3_base_camera=t_base_camera,
        T_camera_fr3_base=t_camera_base,
        T_fr3_base_mujoco_camera=t_base_mj,
        T_panda_link8_camera=t_mount,
        translation_m=t_base_camera[:, :3, 3],
        quaternion_xyzw=continuous_quaternions(camera_pose6[:, 3:6]),
        source_timestep_index=source_index,
        video_frame_index=video_index,
        valid_mask=wrist_valid,
        metadata_json=np.array(json.dumps({"camera_serial": wrist, "pose_order": "xyz_euler"})),
    )
    write_json(episode_dir / "derived" / "calibration" / "wrist_mount_calibration.json", {
        "schema_version": "wrist-fixed-mount-v1",
        "parent_frame": "panda_link8",
        "camera_frame": "wrist_camera_opencv_optical",
        "T_panda_link8_camera": fixed_mount.tolist(),
        "source": f"observation/camera_extrinsics/{wrist}_left_gripper_offset[0]",
        "must_remain_rigid": True,
        "forbid_per_frame_pose_assignment_in_simulation": True,
        "observed_translation_spread_m": mount_translation_spread,
        "observed_rotation_spread_rad": mount_rotation_spread,
    })

    camera_info: Dict[str, Any] = {}
    for role, serial in role_to_serial.items():
        if serial not in intrinsics:
            raise KeyError(f"No intrinsics for camera {serial}")
        camera_info[role] = {
            "serial_number": serial,
            "video": f"{role}.mp4",
            "intrinsics": camera_intrinsics(intrinsics[serial]),
        }
        if role.startswith("ext"):
            if serial not in extrinsics:
                raise KeyError(f"No post-hoc cam2base extrinsics for {serial}")
            t_base_ext = pose6_to_matrix(np.asarray(extrinsics[serial]))
            camera_info[role]["extrinsics"] = {
                "source": "KarlP/droid cam2base_extrinsic_superset.json",
                "source_convention": "T_fr3_base_camera",
                "T_fr3_base_camera": t_base_ext.tolist(),
                "convention": "T_camera_world",
                "matrix": np.linalg.inv(t_base_ext).tolist(),
                "quality_metric": extrinsics.get(f"{serial}_quality_metric"),
                "quality_metric_type": extrinsics.get(f"{serial}_metric_type"),
            }
    write_json(episode_dir / "raw" / "calibration" / "camera_info.json", camera_info)

    for role in role_to_serial:
        create_proxy(episode_dir / "raw" / "videos" / f"{role}.mp4",
                     episode_dir / "derived" / "videos" / f"{role}_proxy.mp4")
    proxy_info = {role: ffprobe_video(episode_dir / "derived" / "videos" / f"{role}_proxy.mp4")
                  for role in role_to_serial}
    frame_count = n - 1
    frame_idx = np.arange(frame_count, dtype=np.int64)
    atomic_savez(
        episode_dir / "derived" / "videos" / "proxy_frame_mapping.npz",
        camera_name=np.array(["ext1", "ext2", "wrist"]),
        source_frame_index=frame_idx,
        proxy_frame_index=frame_idx,
        source_pts_sec=frame_idx.astype(np.float64) / 60.0,
        proxy_pts_sec=frame_idx.astype(np.float64) / 60.0,
        action_index=frame_idx,
        control_timestamp_sec=timestamp_sec[:-1],
        wrist_capture_timestamp_sec=camera_timestamp_sec[:-1],
        valid_mask=np.ones(frame_count, dtype=bool),
        metadata_json=np.array(json.dumps({"mapping_type": "one-to-one identity", "terminal_action": n - 1})),
    )

    generated = now_utc()
    action_spec = {
        "schema_version": "real2sim-action-spec-v2", "generated_at_utc": generated,
        "primary_action": {"shape": [n, 7], "semantics": "absolute commanded EE pose and gripper closure"},
        "translation": {"unit": "meter", "reference_frame": "fr3_robot_base", "controlled_frame": "panda_link8"},
        "rotation": {"representation": "euler", "sequence": "xyz", "composition": "extrinsic_fixed_axes", "unit": "radian", "canonical": "quaternion_xyzw"},
        "gripper": {"open_value": 0.0, "closed_value": 1.0, "max_width_m": GRIPPER_MAX_WIDTH_M, "width_formula": "0.085 * (1 - closure)"},
        "timestamps": {"source": "observation/timestamp/control/control_start", "unit": "millisecond", "strictly_increasing": bool(np.all(np.diff(timestamps_ms) > 0))},
        "companion_arrays": {"cartesian_velocity": "commanded", "joint_position": "commanded", "joint_velocity": "commanded"},
    }
    camera_spec = {
        "schema_version": "real2sim-camera-spec-v2", "generated_at_utc": generated,
        "videos_are_undistorted": True, "render_projection": "pinhole_without_distortion",
        "fixed_external_cameras": ["ext1", "ext2"],
        "external_transform_convention": "camera_info extrinsics.matrix is T_camera_world; T_fr3_base_camera is also retained",
        "wrist": {"parent_frame": "panda_link8", "use_fixed_mount": True, "per_frame_extrinsics_for_validation_only": True},
        "opencv_to_mujoco_axes": CAMERA_AXIS_OPENCV_TO_MUJOCO.tolist(),
        "roles": role_to_serial,
        "image_geometry": {role: proxy_camera_geometry(camera_info[role]["intrinsics"], 640, 360)
                           for role in role_to_serial},
    }
    robot_spec = {
        "schema_version": "real2sim-robot-spec-v2", "generated_at_utc": generated,
        "selected_robot": {"family": family, "serial": metadata["robot_serial"], "dof": 7},
        "arm_asset": {"episode_path": "raw/robot/" + family, "shared_path_from_input_root": "shared/robot_assets/" + family, "tree_sha256": tree_sha256(shared_fr3)},
        "gripper": {"model": "Robotiq 2F-85", "maximum_opening_width_m": GRIPPER_MAX_WIDTH_M},
        "action_pose_frame": "panda_link8",
        "excluded_for_episode": ["Franka Emika Panda" if family == "FR3" else "FR3", "Kinova Gen3 7DoF"],
    }
    time_sync = {
        "schema_version": "real2sim-time-sync-v2", "generated_at_utc": generated,
        "action_sample_count": n, "video_frame_count_per_stream": frame_count,
        "mapping": "video frame i maps to action i; final action is terminal boundary without a displayed frame",
        "physical_timeline_sec": {"start": 0.0, "end": float(timestamp_sec[-1]), "source": "actions_canonical.npz:timestamp_sec"},
        "presentation_timeline": {"fps": 60, "frame_count": frame_count, "duration_sec": frame_count / 60.0},
        "do_not_use_video_fps_as_physics_timestep": True,
    }
    wrist_spec = {
        "schema_version": "real2sim-wrist-extrinsics-spec-v2", "generated_at_utc": generated,
        "camera_serial": wrist, "pose6_layout": ["x", "y", "z", "rx", "ry", "rz"],
        "rotation": {"sequence": "xyz", "composition": "extrinsic_fixed_axes", "unit": "radian"},
        "camera_pose": "T_fr3_base_camera", "projection_transform": "T_camera_fr3_base",
        "mount": "T_panda_link8_camera", "fixed_mount_required": True,
        "sample_count": n,
    }
    proxy_spec = {
        "schema_version": "real2sim-proxy-video-spec-v2", "generated_at_utc": generated,
        "profile": {"codec": "h264", "pixel_format": "yuv420p", "width": 640, "height": 360, "fps": 60, "frame_count": frame_count},
        "aspect_ratio_policy": "scale to fit and pad; never stretch, crop, flip, or mirror",
        "mapping": "derived/videos/proxy_frame_mapping.npz",
        "streams": {role: {"raw": raw_video_info[role], "proxy": proxy_info[role],
                           "image_geometry": proxy_camera_geometry(camera_info[role]["intrinsics"], 640, 360)}
                    for role in role_to_serial},
    }
    input_contract = {
        "schema_version": "real2sim-input-contract-v2", "generated_at_utc": generated,
        "path_base": "episode_root", "self_contained_with_shared_assets": True,
        "allowed_read_scope": ["this episode/**", "input/shared/robot_assets/" + family + "/**"],
        "forbidden_read_scope": ["reconstruction/**", "DROID_samples/**", "paths outside input/**"],
        "authoritative_sources": {"inventory": "manifest.json", "actions": "specs/action_spec.json", "cameras": "specs/camera_spec.json", "time": "specs/time_sync.json", "robot": "specs/robot_spec.json"},
        "task_semantics": {
            "status": "not_provided_in_input",
            "must_be_inferred_from": ["derived/videos/ext1_proxy.mp4", "derived/videos/ext2_proxy.mp4", "derived/videos/wrist_proxy.mp4"],
            "output_contract": "specs/semantic_annotation_contract.json",
            "output_path_relative_to_run": "intent.json",
        },
        "excluded_semantic_sources": ["DROID post-hoc language annotations", "source metadata task text"],
        "completion_gate": "All blocking contracts are resolved and validate must pass before reconstruction.",
    }
    semantic_annotation_contract = {
        "schema_version": "real2sim-semantic-annotation-contract-v2",
        "generated_at_utc": generated,
        "input_contains_task_label": False,
        "annotation_stage": "first_stage_of_reconstruction",
        "evidence_policy": {
            "primary": ["ext1_proxy", "ext2_proxy", "wrist_proxy"],
            "action_use": "phase timing, motion, and gripper-state evidence only; never infer object identity from action",
            "full_resolution_escalation": "only for a task-blocking visual ambiguity",
            "ignore_external_language_annotations": True,
        },
        "required_output": {
            "path_relative_to_reconstruction": "intent.json",
            "fields": ["I", "c", "G"],
        },
        "acceptance": {
            "view_selection": "Use the views that resolve the task-relevant contacts and state changes.",
            "minimum_claim_policy": "When object identity is uncertain, describe only visible geometry and behavior; do not invent a specific name.",
            "scene_scope_policy": "Infer the smallest local support or fixture needed by the operation; do not assume a tabletop scene.",
        },
    }
    for name, value in {
        "action_spec.json": action_spec, "camera_spec.json": camera_spec,
        "robot_spec.json": robot_spec, "time_sync.json": time_sync,
        "wrist_extrinsics_spec.json": wrist_spec, "proxy_video_spec.json": proxy_spec,
        "input_contract.json": input_contract,
        "semantic_annotation_contract.json": semantic_annotation_contract,
    }.items():
        write_json(episode_dir / "specs" / name, value)

    build_robot_runtime(input_root, episode_dir)

    data_format = f"""# DROID episode {selection.episode_id}

Read the three camera proxies in `derived/videos`, metric depth in
`derived/depth/depth.h5`, and `derived/actions_canonical.npz`. Interpret
interaction timing with the action timestamps and the source-frame mapping.
The arm is {family} with a Robotiq 2F-85. Camera conventions and intrinsics
are recorded under `specs`. The wrist camera is fixed to the flange.
"""
    if family == "Panda":
        data_format += "\nLegacy array keys containing fr3 denote the selected robot base, not an FR3 model.\n"
    (episode_dir / "DATA_FORMAT.md").write_text(data_format, encoding="utf-8")
    _build_manifest(episode_dir, metadata, selection, shared_fr3)
    return episode_dir
