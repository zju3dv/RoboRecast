from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from mobile_input.loader import load_episode
from mobile_input.writer import write_json


def _frame(path: Path, index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot read {path} frame {index}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def bootstrap(input_dir: Path, reconstruction_dir: Path) -> dict:
    input_dir, reconstruction_dir = Path(input_dir), Path(reconstruction_dir)
    validation = json.loads((input_dir / "validation_report.json").read_text())
    if validation["status"] != "PASS":
        raise ValueError("input validation failed")
    source = load_episode(input_dir)
    base = source.base_twist_body
    route = source.route_prior_xyyaw
    qpos = source.arm_position
    arm_velocity = source.arm_velocity
    timestamps = source.timestamp_sec
    rows = source.rows
    camera_roles = tuple(source.camera_paths)
    groups = {"early": np.linspace(0, max(0, rows // 3 - 1), 4).astype(int),
              "middle": np.linspace(rows // 3, 2 * rows // 3 - 1, 4).astype(int),
              "late": np.linspace(2 * rows // 3, rows - 1, 4).astype(int)}
    bundle = reconstruction_dir / "review/source"
    bundle.mkdir(parents=True, exist_ok=True)
    sheets = {}
    for period, indices in groups.items():
        sheet = Image.new("RGB", (4 * 320, 3 * 240), "black")
        draw = ImageDraw.Draw(sheet)
        for column, index in enumerate(indices):
            for row, role in enumerate(camera_roles):
                image = _frame(source.camera_paths[role], int(index))
                sheet.paste(image.resize((320, 240)), (column * 320, row * 240))
                draw.text((column * 320 + 8, row * 240 + 8), f"{role} {index}", fill="yellow")
        output = bundle / f"overview_{period}.jpg"
        sheet.save(output, quality=88)
        sheets[period] = str(output.relative_to(reconstruction_dir))
    dt = np.diff(timestamps)
    route_summary = {"duration_sec": float(timestamps[-1] - timestamps[0]),
                     "travel_prior_m": float(np.sum(
                         np.linalg.norm(base[:-1, :2], axis=1) * dt)),
                     "net_xyyaw_prior": route[-1].tolist(),
                     "moving_rows": int(np.sum(np.abs(base[:, 0]) > .03)),
                     "forward_speed_range_mps": [float(base[:, 0].min()), float(base[:, 0].max())],
                     "lateral_speed_range_mps": [float(base[:, 1].min()), float(base[:, 1].max())],
                     "yaw_rate_range_radps": [float(base[:, 2].min()), float(base[:, 2].max())]}
    mode_windows = []
    window_rows = max(1, int(round(.5 / float(np.median(dt)))))
    for start in range(0, rows, window_rows):
        end = min(start + window_rows, rows)
        moving = np.mean((np.linalg.norm(base[start:end, :2], axis=1) > .03) |
                         (np.abs(base[start:end, 2]) > .1)) > .4
        arms = np.mean(np.max(np.abs(arm_velocity[start:end]), axis=1) > .12) > .4
        mode = ("overlap" if moving and arms else "base" if moving else
                "arms" if arms else "idle")
        if mode_windows and mode_windows[-1]["mode"] == mode:
            mode_windows[-1]["rows"][1] = end
        else:
            mode_windows.append({"rows": [start, end], "mode": mode})
    route_summary["source_motion_mode_runs"] = mode_windows
    route_summary["mode_hint_note"] = "Velocity-only hints; video contact semantics decide action names and boundaries."
    write_json(bundle / "motion_summary.json", route_summary)
    context = {"schema_version": "mobile-workspace-context-v1", "episode_id": source.episode_id,
               "contact_sheets": sheets,
               "motion_summary": str((bundle / "motion_summary.json").relative_to(reconstruction_dir)),
               "episode_schema": source.schema_version,
               "robot_contract": "derived/robot_runtime/robot_runtime_contract.json"}
    write_json(reconstruction_dir / "workspace_context.json", context)
    return context
