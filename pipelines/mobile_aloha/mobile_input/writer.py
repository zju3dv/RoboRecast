from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .schema import CameraStream, CanonicalEpisodeV2


BundleInstaller = Callable[[str, Path, np.ndarray], None]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_proxy(camera: CameraStream, path: Path, media_fps: float) -> None:
    first = camera.frame(0)
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{media_fps:.8g}",
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-pix_fmt", "yuv420p", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for row in range(camera.rows):
            frame = first if row == 0 else camera.frame(row)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {camera.role}: {error[-2000:]}")
    except Exception:
        process.kill()
        process.wait()
        raise


def _review_media(root: Path, rows: int, route: np.ndarray) -> None:
    review = root / "review"
    review.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, rows - 1, 5).round().astype(int)
    roles = ("top", "left_wrist", "right_wrist")
    sheet = Image.new("RGB", (5 * 320, 3 * 240), "black")
    draw = ImageDraw.Draw(sheet)
    for line, role in enumerate(roles):
        capture = cv2.VideoCapture(str(root / f"derived/videos/{role}_proxy.mp4"))
        for column, row in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row))
            ok, bgr = capture.read()
            if not ok:
                capture.release()
                raise ValueError(f"missing review frame {role}:{row}")
            rgb = cv2.cvtColor(cv2.resize(bgr, (320, 240)), cv2.COLOR_BGR2RGB)
            sheet.paste(Image.fromarray(rgb), (column * 320, line * 240))
            draw.text((column * 320 + 8, line * 240 + 8), f"{role} row {row}", fill="yellow")
        capture.release()
    sheet.save(review / "input_contact_sheet.jpg", quality=90)
    canvas = np.full((640, 640, 3), 250, np.uint8)
    xy = np.asarray(route[:, :2], float)
    low, high = xy.min(axis=0), xy.max(axis=0)
    center = (low + high) / 2
    scale = min(550 / max(high[0] - low[0], .2), 550 / max(high[1] - low[1], .2))
    points = np.rint((xy - center) * scale + 320).astype(int)
    points[:, 1] = 640 - points[:, 1]
    cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, (180, 100, 20), 2)
    cv2.circle(canvas, tuple(points[0]), 7, (30, 180, 30), -1)
    cv2.circle(canvas, tuple(points[-1]), 7, (30, 30, 200), -1)
    cv2.imwrite(str(review / "route_prior.png"), canvas)


def write_episode(canonical: CanonicalEpisodeV2, output: Path,
                  install_bundle: BundleInstaller) -> Path:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{canonical.episode_id}.", dir=output.parent))
    try:
        trajectory_dir = temp / "derived/trajectory"
        trajectory_dir.mkdir(parents=True)
        np.savez_compressed(
            trajectory_dir / "canonical_trajectory.npz",
            timestamp_sec=np.asarray(canonical.timestamp_sec, np.float64),
            arm_position=np.asarray(canonical.arm_position, np.float32),
            arm_velocity=np.asarray(canonical.arm_velocity, np.float32),
            arm_command=np.asarray(canonical.arm_command, np.float32),
            base_twist_body=np.asarray(canonical.base_twist_body, np.float32),
            route_prior_xyyaw=np.asarray(canonical.route_prior_xyyaw, np.float64))
        if canonical.source_extras:
            np.savez_compressed(trajectory_dir / "source_extras.npz", **canonical.source_extras)
        median_dt = float(np.median(np.diff(canonical.timestamp_sec)))
        media_fps = float(np.clip(1.0 / median_dt, 1, 120))
        for role, camera in canonical.cameras.items():
            _write_proxy(camera, temp / f"derived/videos/{role}_proxy.mp4", media_fps)
        np.savez_compressed(
            temp / "derived/videos/frame_mapping.npz",
            proxy_frame_index=np.arange(len(canonical.timestamp_sec), dtype=np.int64),
            source_row_index=np.arange(len(canonical.timestamp_sec), dtype=np.int64),
            timestamp_sec=np.asarray(canonical.timestamp_sec, np.float64))
        specs = {
            "source.json": canonical.source_spec,
            "source_audit.json": canonical.source_audit,
            "time_sync.json": canonical.time_sync,
            "camera_calibration.json": canonical.camera_calibration,
            "route_prior.json": canonical.route_spec,
            "task_prior.json": canonical.task_prior,
            "embodiment.json": canonical.embodiment,
        }
        for name, value in specs.items():
            write_json(temp / "specs" / name, value)
        install_bundle(canonical.embodiment_id, temp, canonical.arm_position[0])
        contract = {
            "schema_version": "mobile-episode-contract-v2",
            "dataset": canonical.dataset,
            "source_episode": canonical.source_path.name,
            "task_prior": "specs/task_prior.json",
            "embodiment_id": canonical.embodiment_id,
            "trajectory_path": "derived/trajectory/canonical_trajectory.npz",
            "camera_roles": {role: f"derived/videos/{role}_proxy.mp4" for role in canonical.cameras},
            "frame_mapping_path": "derived/videos/frame_mapping.npz",
            "route_source": canonical.route_spec.get("method", "unknown"),
            "calibration_availability": canonical.camera_calibration.get("status", "unknown"),
            "robot_runtime_contract_path": "derived/robot_runtime/robot_runtime_contract.json",
            "optional_modalities": sorted(canonical.source_extras),
        }
        write_json(temp / "episode_contract.json", contract)
        _review_media(temp, len(canonical.timestamp_sec), canonical.route_prior_xyyaw)
        files = [{"path": path.relative_to(temp).as_posix(), "sha256": sha256(path),
                  "size_bytes": path.stat().st_size}
                 for path in sorted(temp.rglob("*")) if path.is_file()]
        write_json(temp / "manifest.json", {
            "schema_version": "mobile-real2sim-input-v2",
            "episode_id": canonical.episode_id,
            "dataset": canonical.dataset,
            "embodiment_id": canonical.embodiment_id,
            "rows": len(canonical.timestamp_sec),
            "duration_sec": float(canonical.timestamp_sec[-1] - canonical.timestamp_sec[0]),
            "files": files,
        })
        from .validate import validate_episode
        report = validate_episode(temp)
        write_json(temp / "validation_report.json", report)
        if report["status"] != "PASS":
            raise ValueError(f"generated input failed validation: {report}")
        from mobile_harness.bootstrap import bootstrap
        bootstrap(temp, temp)
        os.replace(temp, output)
        return output
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
