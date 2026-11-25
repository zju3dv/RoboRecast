from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import mujoco
import numpy as np

from .util import ffprobe_video, read_json, sha256_file, write_json


REQUIRED = [
    "manifest.json", "DATA_FORMAT.md",
    "raw/metadata/episode_metadata_sanitized.json",
    "raw/videos/ext1.mp4", "raw/videos/ext2.mp4", "raw/videos/wrist.mp4",
    "raw/actions/actions.npz", "raw/actions/actions.json",
    "raw/calibration/camera_info.json", "raw/calibration/wrist_extrinsics_raw.npz",
    "derived/actions_canonical.npz",
    "derived/calibration/wrist_extrinsics_canonical.npz",
    "derived/calibration/wrist_mount_calibration.json",
    "derived/videos/ext1_proxy.mp4", "derived/videos/ext2_proxy.mp4",
    "derived/videos/wrist_proxy.mp4", "derived/videos/proxy_frame_mapping.npz",
    "specs/input_contract.json", "specs/action_spec.json", "specs/camera_spec.json",
    "specs/robot_spec.json", "specs/time_sync.json", "specs/wrist_extrinsics_spec.json",
    "specs/proxy_video_spec.json", "specs/semantic_annotation_contract.json",
    "derived/robot_runtime/robot_base_scene.xml",
    "derived/robot_runtime/robot_runtime_contract.json",
    "derived/robot_runtime/name_map.json",
    "derived/robot_runtime/build_manifest.json",
]

REQUIRED_V3 = ["derived/depth/depth.h5", "specs/depth_spec.json"]


def validate_episode(episode_dir: Path, input_root: Path, write_report: bool = True) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None, severity: str = "error") -> None:
        checks.append({"name": name, "passed": bool(ok), "severity": severity, "detail": detail})

    manifest_preview = read_json(episode_dir / "manifest.json") if (episode_dir / "manifest.json").is_file() else {}
    is_v3 = manifest_preview.get("schema_version") == "real2sim-input-manifest-v3"
    required = REQUIRED + (REQUIRED_V3 if is_v3 else [])
    for rel in required:
        check(f"required:{rel}", (episode_dir / rel).exists())
    if not all((episode_dir / rel).exists() for rel in required):
        report = {"schema_version": "real2sim-validation-v1", "episode_id": episode_dir.name,
                  "status": "FAIL", "checks": checks}
        if write_report:
            write_json(episode_dir / "validation_report.json", report)
        return report

    manifest = read_json(episode_dir / "manifest.json")
    check("manifest_schema", manifest.get("schema_version") in {
        "real2sim-input-manifest-v2", "real2sim-input-manifest-v3",
    })
    check("episode_id_matches_directory", manifest.get("episode_id") == episode_dir.name)
    robot_serial = str(manifest.get("episode", {}).get("robot_serial", ""))
    robot_spec = read_json(episode_dir / "specs/robot_spec.json")
    family = robot_spec.get("selected_robot", {}).get("family", "FR3")
    explicit_identity = family in {"FR3", "Panda"} and robot_serial.lower().startswith(family.lower() + "-")
    evidence_path = episode_dir / 'specs/robot_identity_evidence.json'
    if not explicit_identity and evidence_path.exists() and family in {'FR3', 'Panda'}:
        evidence = read_json(evidence_path)
        source = Path(evidence['path'])
        known_serial = str(read_json(source).get('robot_serial', '')).lower()
        explicit_identity = (sha256_file(source) == evidence['sha256'] and
            robot_serial.lower() == str(evidence['raw_serial']).lower() and
            known_serial == family.lower() + '-' + robot_serial.lower())
    check("robot_is_explicit_selected_family", explicit_identity, robot_serial)
    check("episode_success", manifest.get("episode", {}).get("success") is True)
    episode_info = manifest.get("episode", {})
    check("task_semantics_deferred", episode_info.get("task_semantics_status") == "unlabeled_agent_must_infer_from_video")
    check("manifest_has_no_task_label", not any(key in episode_info for key in ("visual_task", "task_class", "source_current_task")))
    check("posthoc_language_absent", not (episode_dir / "raw" / "metadata" / "language_annotations.json").exists())
    check("unsanitized_metadata_absent", not (episode_dir / "raw" / "metadata" / "episode_metadata.json").exists())
    sanitized_metadata = read_json(episode_dir / "raw" / "metadata" / "episode_metadata_sanitized.json")
    check("metadata_declares_task_fields_removed", sanitized_metadata.get("task_or_language_fields_removed") is True)
    check("metadata_has_no_task_language_fields", not any(
        token in key.lower() for key in sanitized_metadata for token in ("task", "language", "instruction")
        if key != "task_or_language_fields_removed"
    ))

    for entry in manifest.get("files", []):
        path = episode_dir / entry["path"]
        check(f"manifest_file:{entry['path']}", path.is_file() and path.stat().st_size == entry["size_bytes"] and sha256_file(path) == entry["sha256"])

    robot_link = episode_dir / "raw" / "robot" / family
    expected_robot = input_root / "shared" / "robot_assets" / family
    check("robot_asset_is_shared_link", robot_link.is_symlink())
    check("robot_asset_resolves_to_selected_robot", robot_link.resolve() == expected_robot.resolve(), str(robot_link.resolve()))
    try:
        robot_link.resolve().relative_to(input_root.resolve())
        inside = True
    except ValueError:
        inside = False
    check("robot_asset_stays_inside_input", inside)
    check("kinova_not_linked_by_episode", not (episode_dir / "raw" / "robot" / "Kinova_Gen3_7DoF").exists())

    runtime_dir = episode_dir / "derived/robot_runtime"
    runtime_manifest = read_json(runtime_dir / "build_manifest.json")
    runtime_contract = read_json(runtime_dir / "robot_runtime_contract.json")
    check("robot_runtime_matches_selected_family", runtime_contract.get("robot_family") == family)
    runtime_entry = manifest.get("robot_runtime") or {}
    check("robot_runtime_manifest_schema",
          runtime_manifest.get("schema_version") == "real2sim-robot-runtime-manifest-v1")
    check("robot_runtime_contract_schema",
          runtime_contract.get("schema_version") == "real2sim-robot-runtime-v1")
    check("robot_runtime_manifest_link",
          runtime_entry.get("path") == "derived/robot_runtime/build_manifest.json" and
          runtime_entry.get("sha256") == sha256_file(runtime_dir / "build_manifest.json"))
    for filename, digest in runtime_manifest.get("files", {}).items():
        path = runtime_dir / filename
        check("robot_runtime_hash:{}".format(filename),
              path.is_file() and sha256_file(path) == digest)
    try:
        runtime_model = mujoco.MjModel.from_xml_path(str(runtime_dir / "robot_base_scene.xml"))
    except Exception as exc:
        check("robot_runtime_compiles", False, "{}: {}".format(type(exc).__name__, exc))
    else:
        expected_compiled = runtime_manifest.get("compiled", {})
        actual_compiled = {"nq": runtime_model.nq, "nv": runtime_model.nv, "nu": runtime_model.nu,
                           "nbody": runtime_model.nbody, "ngeom": runtime_model.ngeom,
                           "ncam": runtime_model.ncam}
        check("robot_runtime_compiles", True, actual_compiled)
        check("robot_runtime_structure", all(expected_compiled.get(key) == value
                                               for key, value in actual_compiled.items()),
              {"expected": expected_compiled, "actual": actual_compiled})
    check("robot_runtime_controller_regression",
          runtime_manifest.get("controller_regression", {}).get("status") == "PASS",
          runtime_manifest.get("controller_regression"))

    with np.load(episode_dir / "raw" / "actions" / "actions.npz") as raw:
        action = raw["action"]
        timestamps_ms = raw["timestamp_ms"]
    n = len(action)
    check("raw_action_shape", action.shape == (n, 7), list(action.shape))
    check("raw_action_finite", np.isfinite(action).all())
    check("timestamps_strictly_increasing", bool(np.all(np.diff(timestamps_ms) > 0)))
    with np.load(episode_dir / "derived" / "actions_canonical.npz") as canonical:
        expected_keys = {"timestamp_sec", "position_m", "quaternion_xyzw", "gripper_width_m", "source_index", "valid_mask"}
        check("canonical_action_keys", expected_keys.issubset(canonical.files), canonical.files)
        check("canonical_action_length", len(canonical["timestamp_sec"]) == n)
        qnorm = np.linalg.norm(canonical["quaternion_xyzw"], axis=1)
        check("canonical_quaternions_normalized", bool(np.allclose(qnorm, 1.0, atol=1e-8)), float(np.max(np.abs(qnorm - 1))))
        widths = canonical["gripper_width_m"]
        check("gripper_width_range", bool(np.all((widths >= 0) & (widths <= 0.085 + 1e-12))))
        check("canonical_valid_mask", bool(np.all(canonical["valid_mask"])))

    raw_counts = {}
    proxy_counts = {}
    raw_geometry = {}
    for role in ("ext1", "ext2", "wrist"):
        raw_info = ffprobe_video(episode_dir / "raw" / "videos" / f"{role}.mp4")
        proxy_info = ffprobe_video(episode_dir / "derived" / "videos" / f"{role}_proxy.mp4")
        raw_counts[role] = int(raw_info.get("nb_read_frames", raw_info.get("nb_frames", -1)))
        raw_geometry[role] = [int(raw_info.get("width", 0)), int(raw_info.get("height", 0))]
        proxy_counts[role] = int(proxy_info.get("nb_read_frames", proxy_info.get("nb_frames", -1)))
        check(f"proxy_{role}_geometry", proxy_info.get("width") == 640 and proxy_info.get("height") == 360)
        check(f"proxy_{role}_codec", proxy_info.get("codec_name") == "h264")
        check(f"proxy_{role}_pix_fmt", proxy_info.get("pix_fmt") == "yuv420p")
        check(f"proxy_{role}_fps", proxy_info.get("avg_frame_rate") == "60/1")
        check(f"proxy_{role}_no_rotation", not proxy_info.get("tags", {}).get("rotate"))
    check("raw_frame_count_is_n_minus_one", set(raw_counts.values()) == {n - 1}, raw_counts)
    check("proxy_frame_count_matches_raw", proxy_counts == raw_counts, {"raw": raw_counts, "proxy": proxy_counts})

    with np.load(episode_dir / "derived" / "videos" / "proxy_frame_mapping.npz") as mapping:
        check("mapping_length", len(mapping["source_frame_index"]) == n - 1)
        check("mapping_identity", bool(np.array_equal(mapping["source_frame_index"], mapping["proxy_frame_index"])))
        check("mapping_action_identity", bool(np.array_equal(mapping["source_frame_index"], mapping["action_index"])))

    if is_v3:
        depth_spec = read_json(episode_dir / "specs" / "depth_spec.json")
        check("depth_spec_schema", depth_spec.get("schema_version") == "real2sim-depth-spec-v1")
        check("depth_unit_mm", depth_spec.get("unit") == "millimeter")
        check("depth_uint16", depth_spec.get("dtype") == "uint16")
        check("depth_zero_invalid", depth_spec.get("invalid_value") == 0)
        check("depth_proxy_identity", depth_spec.get("pixel_mapping", {}).get("proxy_to_depth", "").startswith("identity"))
        with h5py.File(str(episode_dir / "derived" / "depth" / "depth.h5"), "r") as depth:
            check("depth_h5_schema", depth.attrs.get("schema_version") == "droid-proxy-depth-v1")
            check("depth_h5_geometry", int(depth.attrs.get("width", 0)) == 640 and int(depth.attrs.get("height", 0)) == 360)
            for role in ("ext1", "ext2", "wrist"):
                prefix = "cameras/{}".format(role)
                check("depth_{}_group".format(role), prefix in depth)
                if prefix not in depth:
                    continue
                group = depth[prefix]
                array = group.get("depth_mm")
                expected_shape = (proxy_counts[role], 360, 640)
                check("depth_{}_shape".format(role), array is not None and array.shape == expected_shape,
                      list(array.shape) if array is not None else None)
                check("depth_{}_dtype".format(role), array is not None and array.dtype == np.uint16,
                      str(array.dtype) if array is not None else None)
                check("depth_{}_timestamps".format(role), "timestamp_ns" in group and len(group["timestamp_ns"]) == proxy_counts[role])
                check("depth_{}_grab_valid".format(role), "grab_valid" in group and len(group["grab_valid"]) == proxy_counts[role])
                check("depth_{}_positions".format(role), "svo_frame_index" in group and
                      np.array_equal(group["svo_frame_index"][:], np.arange(proxy_counts[role])))
                check("depth_{}_intrinsics".format(role), all(name in group.attrs for name in ("fx", "fy", "cx", "cy")))
                check("depth_{}_pixel_transform".format(role),
                      group.attrs.get("svo_to_proxy_pixel_transform", "identity") in {
                          "identity", "rotate_180", "flip_horizontal", "flip_vertical",
                      }, group.attrs.get("svo_to_proxy_pixel_transform", "identity (implicit)"))
                check("depth_{}_stereo_view".format(role),
                      group.attrs.get("stereo_view", "left") in {"left", "right"},
                      group.attrs.get("stereo_view", "left (implicit)"))
                if "valid_ratio" in group:
                    ratios = group["valid_ratio"][:]
                    check("depth_{}_has_valid_pixels".format(role), bool(np.any(ratios > 0)), float(np.mean(ratios)))
        records = depth_spec.get("extraction", {}).get("records", {})
        for role in ("ext1", "ext2", "wrist"):
            score = records.get(role, {}).get("minimum_alignment_ncc")
            check("depth_{}_proxy_alignment".format(role), isinstance(score, (int, float)) and score >= 0.80,
                  score)
            failure_fraction = records.get(role, {}).get("grab_failure_fraction")
            check("depth_{}_grab_failure_fraction".format(role),
                  isinstance(failure_fraction, (int, float)) and failure_fraction <= 0.01,
                  failure_fraction)

    camera_info = read_json(episode_dir / "raw" / "calibration" / "camera_info.json")
    check("camera_roles", set(camera_info) == {"ext1", "ext2", "wrist"}, list(camera_info))
    for role in ("ext1", "ext2"):
        matrix = np.asarray(camera_info[role]["extrinsics"]["matrix"], dtype=float)
        rot = matrix[:3, :3]
        check(f"{role}_homogeneous_matrix", matrix.shape == (4, 4) and np.allclose(matrix[3], [0, 0, 0, 1]))
        check(f"{role}_rotation_valid", np.allclose(rot.T @ rot, np.eye(3), atol=1e-8) and np.isclose(np.linalg.det(rot), 1.0, atol=1e-8))
        source = camera_info[role]["extrinsics"].get("source", "")
        check(f"{role}_calibration_source_declared", bool(source), source)

    runtime_contract = read_json(episode_dir / "derived/robot_runtime/robot_runtime_contract.json")
    for role in ("ext1", "ext2", "wrist"):
        camera = runtime_contract.get("cameras", {}).get(role, {})
        source, proxy = camera.get("source_intrinsics", {}), camera.get("proxy_intrinsics", {})
        transform = camera.get("source_to_proxy_pixel", {})
        scale, offset = transform.get("scale_xy", []), transform.get("offset_xy", [])
        complete = len(scale) == 2 and len(offset) == 2 and source and proxy
        consistent = bool(complete and [source.get("width"), source.get("height")] == raw_geometry[role] and
                          proxy.get("width") == 640 and proxy.get("height") == 360 and
                          np.isclose(proxy.get("fx"), source.get("fx") * scale[0]) and
                          np.isclose(proxy.get("fy"), source.get("fy") * scale[1]) and
                          np.isclose(proxy.get("cx"), source.get("cx") * scale[0] + offset[0]) and
                          np.isclose(proxy.get("cy"), source.get("cy") * scale[1] + offset[1]))
        check("{}_proxy_camera_geometry".format(role), consistent, camera)

    with np.load(episode_dir / "derived" / "calibration" / "wrist_extrinsics_canonical.npz") as wrist:
        check("wrist_length", len(wrist["timestamp_sec"]) == n)
        matrices = wrist["T_panda_link8_camera"]
        check("wrist_mount_constant", bool(np.allclose(matrices, matrices[0], atol=1e-10)))
        check("wrist_valid_mask", bool(np.all(wrist["valid_mask"])))
    mount = read_json(episode_dir / "derived" / "calibration" / "wrist_mount_calibration.json")
    check("wrist_fixed_mount_required", mount.get("must_remain_rigid") is True)
    check("wrist_per_frame_assignment_forbidden", mount.get("forbid_per_frame_pose_assignment_in_simulation") is True)

    semantic_contract = read_json(episode_dir / "specs" / "semantic_annotation_contract.json")
    check("semantic_annotation_contract_schema", semantic_contract.get("schema_version") in ("real2sim-semantic-annotation-contract-v1", "real2sim-semantic-annotation-contract-v2"))
    check("input_contains_no_task_label", semantic_contract.get("input_contains_task_label") is False)
    check("semantic_annotation_is_reconstruction_stage", semantic_contract.get("annotation_stage") == "first_stage_of_reconstruction")
    check("posthoc_language_ignored", semantic_contract.get("evidence_policy", {}).get("ignore_external_language_annotations") is True)
    output = semantic_contract["required_output"]
    if semantic_contract["schema_version"].endswith("-v2"):
        declared = read_json(episode_dir / "specs/input_contract.json")["task_semantics"]
        check("intent_output_contract", output["fields"] == ["I", "c", "G"] and
              output["path_relative_to_reconstruction"] == declared["output_path_relative_to_run"]
              == episode_info.get("task_semantics_output") == "intent.json")
    else:
        check("legacy_semantics_contract", {"observed_task", "phases", "goal_state"}.issubset(output["fields"]))

    failures = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    report = {
        "schema_version": "real2sim-validation-v1",
        "episode_id": episode_dir.name,
        "status": "FAIL" if failures else "PASS",
        "summary": {"passed": len(checks) - len(failures) - len(warnings), "failed": len(failures), "warnings": len(warnings)},
        "checks": checks,
    }
    if write_report:
        write_json(episode_dir / "validation_report.json", report)
    return report


def validate_all(input_root: Path) -> Dict[str, Any]:
    episodes_dir = input_root / "episodes"
    reports = [validate_episode(path, input_root) for path in sorted(episodes_dir.iterdir()) if path.is_dir()]
    aggregate = {
        "schema_version": "real2sim-validation-aggregate-v1",
        "status": "PASS" if reports and all(row["status"] == "PASS" for row in reports) else "FAIL",
        "episode_count": len(reports),
        "passed_episode_count": sum(row["status"] == "PASS" for row in reports),
        "episodes": [{"episode_id": row["episode_id"], "status": row["status"], "summary": row.get("summary", {})} for row in reports],
    }
    write_json(input_root / "validation_report.json", aggregate)
    return aggregate
