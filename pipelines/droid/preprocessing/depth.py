from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from .transform import now_utc
from .util import read_json, sha256_file, write_json


CAMERAS = ("ext1", "ext2", "wrist")
WIDTH = 640
HEIGHT = 360
MAX_DEPTH_MM = 10000


def _source_episode(samples_root: Path, episode_id: str) -> Tuple[Path, Dict[str, Any]]:
    matches = []
    # Catalogs may contain metadata_*.json lists. Select only this exact
    # episode filename, then validate the embedded UUID; never parse catalogs.
    for metadata_path in samples_root.glob("*/metadata_{}.json".format(episode_id)):
        if metadata_path.parent.name.startswith("_"):
            continue
        metadata = read_json(metadata_path)
        if isinstance(metadata, dict) and metadata.get("uuid") == episode_id:
            matches.append((metadata_path.parent, metadata))
    if len(matches) != 1:
        raise ValueError("expected one raw sample for {}, found {}".format(episode_id, len(matches)))
    return matches[0]


def _proxy_frame_count(episode_dir: Path) -> int:
    with np.load(str(episode_dir / "derived/videos/proxy_frame_mapping.npz"), allow_pickle=False) as archive:
        return int(len(archive["proxy_frame_index"]))


def _proxy_frame(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("cannot decode proxy frame {} from {}".format(index, video))
        return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    finally:
        capture.release()


def _alignment_score(first_bgr: np.ndarray, second_bgr: np.ndarray) -> float:
    first = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    second = cv2.cvtColor(second_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.sum(first * second) / denominator) if denominator else 0.0


def _apply_pixel_transform(array: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return array
    if transform == "rotate_180":
        return np.flip(array, axis=(0, 1))
    if transform == "flip_horizontal":
        return np.flip(array, axis=1)
    if transform == "flip_vertical":
        return np.flip(array, axis=0)
    raise ValueError("unknown pixel transform {}".format(transform))


def _detect_pixel_transform(proxy_bgr: np.ndarray, svo_bgr: np.ndarray) -> Tuple[str, Dict[str, float]]:
    scores = {
        transform: _alignment_score(proxy_bgr, _apply_pixel_transform(svo_bgr, transform))
        for transform in ("identity", "rotate_180", "flip_horizontal", "flip_vertical")
    }
    return max(scores, key=scores.get), scores


def _camera_intrinsics(camera: Any, resolution: Any, stereo_view: str = "left") -> Dict[str, float]:
    info = camera.get_camera_information(resolution)
    parameters = info.camera_configuration.calibration_parameters
    lens = parameters.left_cam if stereo_view == "left" else parameters.right_cam
    return {"fx": float(lens.fx), "fy": float(lens.fy), "cx": float(lens.cx), "cy": float(lens.cy),
            "width": WIDTH, "height": HEIGHT}


def _sdk_version(sl: Any) -> str:
    try:
        return str(sl.Camera.get_sdk_version())
    except Exception:
        return "unknown"


def _extract_stream(sl: Any, svo_path: Path, proxy_path: Path, group: h5py.Group,
                    frame_count: int, max_depth_mm: int, settings_dir: Optional[Path]) -> Dict[str, Any]:
    camera = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    if settings_dir is not None:
        calibration = settings_dir / "SN{}.conf".format(svo_path.stem)
        if not calibration.is_file():
            raise FileNotFoundError("missing ZED calibration file {}".format(calibration))
        init.optional_settings_path = os.path.join(str(settings_dir), "")
    init.svo_real_time_mode = False
    init.enable_right_side_measure = True
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("cannot open {}: {}".format(svo_path, status))
    resolution = sl.Resolution(WIDTH, HEIGHT)
    runtime = sl.RuntimeParameters()
    depth = sl.Mat()
    left = sl.Mat()
    right = sl.Mat()
    available = int(camera.get_svo_number_of_frames())
    if available < frame_count:
        camera.close()
        raise ValueError("{} has {} SVO frames but proxy requires {}".format(svo_path.name, available, frame_count))
    dataset = group.create_dataset("depth_mm", shape=(frame_count, HEIGHT, WIDTH), dtype=np.uint16,
                                   chunks=(1, HEIGHT, WIDTH), compression="lzf", shuffle=True)
    timestamps = group.create_dataset("timestamp_ns", shape=(frame_count,), dtype=np.int64)
    positions = group.create_dataset("svo_frame_index", data=np.arange(frame_count, dtype=np.int32))
    valid_ratios = group.create_dataset("valid_ratio", shape=(frame_count,), dtype=np.float32)
    grab_valid = group.create_dataset("grab_valid", shape=(frame_count,), dtype=np.bool_)
    alignment_indices = sorted(set([0, max(0, frame_count // 2), max(0, frame_count - 2)]))
    alignment = []
    grab_failures = []
    pixel_transform: Optional[str] = None
    transform_scores: Dict[str, float] = {}
    stereo_view: Optional[str] = None
    intrinsics: Dict[str, float] = {}
    try:
        for index in range(frame_count):
            camera.set_svo_position(index)
            grab_status = camera.grab(runtime)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                dataset[index] = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
                valid_ratios[index] = 0.0
                timestamps[index] = -1
                grab_valid[index] = False
                grab_failures.append({"frame_id": index, "status": str(grab_status)})
                continue
            grab_valid[index] = True
            if pixel_transform is None:
                camera.retrieve_image(left, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
                camera.retrieve_image(right, sl.VIEW.RIGHT, sl.MEM.CPU, resolution)
                proxy = _proxy_frame(proxy_path, index)
                eye_images = {}
                for name, image in (("left", left), ("right", right)):
                    rgba = np.asarray(image.get_data())
                    eye_images[name] = (cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
                                        if rgba.shape[-1] == 4 else rgba[..., :3])
                candidates = {}
                for name, image in eye_images.items():
                    _, scores = _detect_pixel_transform(proxy, image)
                    for transform, score in scores.items():
                        candidates["{}:{}".format(name, transform)] = score
                selected = max(candidates, key=candidates.get)
                stereo_view, pixel_transform = selected.split(":", 1)
                transform_scores = candidates
                group.attrs["stereo_view"] = stereo_view
                group.attrs["svo_to_proxy_pixel_transform"] = pixel_transform
                intrinsics = _camera_intrinsics(camera, resolution, stereo_view)
                for name, value in intrinsics.items():
                    group.attrs[name] = value
            measure = sl.MEASURE.DEPTH if stereo_view == "left" else sl.MEASURE.DEPTH_RIGHT
            camera.retrieve_measure(depth, measure, sl.MEM.CPU, resolution)
            values = np.asarray(depth.get_data(), dtype=np.float32)
            if values.ndim == 3:
                values = values[..., 0]
            values = _apply_pixel_transform(values, pixel_transform)
            valid = np.isfinite(values) & (values > 0) & (values <= max_depth_mm)
            encoded = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
            encoded[valid] = np.rint(values[valid]).astype(np.uint16)
            dataset[index] = encoded
            valid_ratios[index] = float(valid.mean())
            timestamps[index] = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if index in alignment_indices:
                selected_image = left if stereo_view == "left" else right
                camera.retrieve_image(selected_image,
                                      sl.VIEW.LEFT if stereo_view == "left" else sl.VIEW.RIGHT,
                                      sl.MEM.CPU, resolution)
                rgba = np.asarray(selected_image.get_data())
                svo_bgr = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR) if rgba.shape[-1] == 4 else rgba[..., :3]
                alignment.append({"frame_id": index, "normalized_cross_correlation": _alignment_score(
                    _proxy_frame(proxy_path, index), _apply_pixel_transform(svo_bgr, pixel_transform))})
    finally:
        camera.close()
    return {
        "source_svo_name": svo_path.name,
        "source_svo_sha256": sha256_file(svo_path),
        "source_svo_frame_count": available,
        "output_frame_count": frame_count,
        "intrinsics": intrinsics,
        "mean_valid_ratio": float(np.asarray(valid_ratios[:]).mean()),
        "grab_failures": grab_failures,
        "grab_failure_fraction": float(len(grab_failures) / frame_count),
        "stereo_view": stereo_view,
        "svo_to_proxy_pixel_transform": pixel_transform,
        "pixel_transform_detection_scores": transform_scores,
        "alignment_samples": alignment,
        "minimum_alignment_ncc": min(row["normalized_cross_correlation"] for row in alignment),
    }


def _manifest_v3(episode_dir: Path) -> None:
    manifest_path = episode_dir / "manifest.json"
    manifest = read_json(manifest_path)
    files = []
    for path in sorted(episode_dir.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name not in {"manifest.json", "validation_report.json"}:
            files.append({"path": path.relative_to(episode_dir).as_posix(), "size_bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    manifest["schema_version"] = "real2sim-input-manifest-v3"
    manifest["generated_at_utc"] = now_utc()
    manifest["depth"] = {
        "path": "derived/depth/depth.h5", "spec": "specs/depth_spec.json",
        "unit": "millimeter", "dtype": "uint16", "invalid_value": 0,
        "purpose": "coarse static-object pose initialization only",
    }
    manifest["files"] = files
    write_json(manifest_path, manifest)


def extract_episode_depth(episode_dir: Path, samples_root: Path,
                          max_depth_mm: int = MAX_DEPTH_MM,
                          settings_dir: Optional[Path] = None) -> Dict[str, Any]:
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise RuntimeError("pyzed is required; run this command in the pinned ZED SDK environment") from exc
    episode_id = episode_dir.name
    sample_dir, metadata = _source_episode(samples_root, episode_id)
    serials = {"wrist": str(metadata["wrist_cam_serial"]), "ext1": str(metadata["ext1_cam_serial"]),
               "ext2": str(metadata["ext2_cam_serial"])}
    frame_count = _proxy_frame_count(episode_dir)
    destination = episode_dir / "derived" / "depth" / "depth.h5"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("depth.part.h5")
    temporary.unlink(missing_ok=True)
    records = {}
    with h5py.File(str(temporary), "w") as archive:
        archive.attrs["schema_version"] = "droid-proxy-depth-v1"
        archive.attrs["depth_unit"] = "millimeter"
        archive.attrs["invalid_value"] = 0
        archive.attrs["max_depth_mm"] = int(max_depth_mm)
        archive.attrs["width"] = WIDTH
        archive.attrs["height"] = HEIGHT
        archive.attrs["zed_sdk_version"] = _sdk_version(sl)
        cameras = archive.create_group("cameras")
        for role in CAMERAS:
            serial = serials[role]
            svo = sample_dir / "recordings" / "SVO" / "{}.svo".format(serial)
            if not svo.is_file():
                raise FileNotFoundError(svo)
            group = cameras.create_group(role)
            group.attrs["serial_number"] = serial
            records[role] = _extract_stream(
                sl, svo, episode_dir / "derived/videos/{}_proxy.mp4".format(role), group,
                frame_count, int(max_depth_mm),
                settings_dir,
            )
        archive.flush()
    os.replace(str(temporary), str(destination))
    spec = {
        "schema_version": "real2sim-depth-spec-v1", "generated_at_utc": now_utc(),
        "episode_id": episode_id, "storage": "derived/depth/depth.h5",
        "shape_per_camera": [frame_count, HEIGHT, WIDTH], "dtype": "uint16",
        "unit": "millimeter", "invalid_value": 0, "maximum_depth_mm": int(max_depth_mm),
        "camera_roles": list(CAMERAS), "pixel_mapping": {
            "proxy_to_depth": "identity after verified 640x360 SVO-left retrieval",
            "origin": "top-left", "x_axis": "right", "y_axis": "down",
        },
        "extraction": {"zed_sdk_version": _sdk_version(sl), "depth_mode": "ULTRA",
                       "offline_svo_playback": True,
                       "calibration_source": "per-camera SN*.conf via optional_settings_path",
                       "records": records},
        "quality_warning": (
            "DROID ZED stereo depth is noisy and may contain holes and edge outliers. "
            "Use local robust statistics only for coarse pose initialization; wrist depth is optional."
        ),
    }
    write_json(episode_dir / "specs" / "depth_spec.json", spec)
    input_contract_path = episode_dir / "specs" / "input_contract.json"
    contract = read_json(input_contract_path)
    contract["schema_version"] = "real2sim-input-contract-v3"
    contract["depth"] = {
        "path": "derived/depth/depth.h5", "spec": "specs/depth_spec.json",
        "allowed_use": "coarse static-scene pose initialization",
        "proxy_pixel_mapping": "identity",
    }
    write_json(input_contract_path, contract)
    data_format = episode_dir / "DATA_FORMAT.md"
    existing = data_format.read_text(encoding="utf-8")
    marker = "## Depth (v3)"
    if marker not in existing:
        data_format.write_text(existing.rstrip() + "\n\n" + marker + "\n\n"
                               "Use `derived/depth/depth.h5` according to `specs/depth_spec.json`. "
                               "Pixels match the 640x360 proxies. Values are uint16 millimetres; zero is invalid. "
                               "Depth is noisy supporting evidence, not ground truth.\n", encoding="utf-8")
    _manifest_v3(episode_dir)
    return {"episode_id": episode_id, "depth_path": str(destination), "size_bytes": destination.stat().st_size,
            "cameras": records}
