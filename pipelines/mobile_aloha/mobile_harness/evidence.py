from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
from .rendering import presentation_renderer, presentation_options
import numpy as np
from PIL import Image, ImageDraw

from mobile_input.loader import load_episode
from .stages import overview_camera
from .camera_visibility import ExternalCameraSelector


def evidence_rows(phases: list[dict], arm_speed: np.ndarray, count: int) -> list[int]:
    chosen = {0, count - 1, int(np.argmax(arm_speed))}
    for phase in phases:
        name = phase["name"].lower()
        start, end = int(phase["start_row"]), int(phase["end_row"])
        if "release" in name or "withdraw" in name:
            chosen.add(start)
        elif any(word in name for word in ("contact", "push", "pull", "carry", "grasp")):
            chosen.add((start + end - 1) // 2)
    if len(chosen) < 6:
        chosen.update(np.linspace(0, count - 1, 6).astype(int).tolist())
    if len(chosen) > 8:
        important = {0, count - 1, int(np.argmax(arm_speed))}
        remaining = sorted(chosen - important)
        indices = np.linspace(0, len(remaining) - 1, 8 - len(important)).astype(int)
        chosen = important | {remaining[i] for i in indices}
    return sorted(chosen)


class RolloutSheetRenderer:
    """One rollout render context; cache panels shared by comparison sheets."""

    def __init__(self, input_dir, scene_path, rollout_path, phases):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        with np.load(rollout_path) as record:
            self.qpos = np.asarray(record["qpos"])
            self.ctrl = np.asarray(record["ctrl"])
            self.arm_speed = np.asarray(record["arm_speed_radps"])
            self.phase_index = np.asarray(record["phase_index"])
        self.phases = phases
        self.source = load_episode(input_dir)
        cameras = self.source.runtime_contract["cameras"]
        self.views = (cameras["top"], cameras["left_wrist"], cameras["right_wrist"], "route_overview")
        self.renderer = presentation_renderer(self.model, height=240, width=320)
        self.capture = cv2.VideoCapture(str(self.source.camera_paths["top"]))
        self.cache = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.capture.release()
        self.renderer.close()

    def panels(self, row):
        if row in self.cache:
            return self.cache[row]
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, row)
        ok, frame = self.capture.read()
        if not ok:
            raise ValueError(f"real top video missing row {row}")
        panels = [Image.fromarray(cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB))]
        self.data.qpos[:] = self.qpos[row]
        self.data.ctrl[:] = self.ctrl[row]
        mujoco.mj_forward(self.model, self.data)
        phase = self.phases[int(self.phase_index[row])]
        for view in self.views:
            camera = view
            if view == "route_overview":
                camera = ExternalCameraSelector().select(
                    self.model, self.data, self.renderer,
                    overview_camera(self.model, self.data, phase.get("object")), phase.get("object"))
            self.renderer.update_scene(self.data, camera=camera, scene_option=presentation_options())
            panels.append(Image.fromarray(self.renderer.render().copy()))
        # Bound image memory for long tasks while retaining recently reused rows.
        if len(self.cache) >= 24:
            self.cache.pop(next(iter(self.cache)))
        self.cache[row] = panels
        return panels

    def render(self, output, rows_override=None):
        rows = (rows_override if rows_override is not None else
                evidence_rows(self.phases, self.arm_speed, len(self.qpos)))
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (5 * 320, len(rows) * 240), "black")
        draw = ImageDraw.Draw(sheet)
        for line, row in enumerate(rows):
            label = (f"row {row} / {self.source.timestamp_sec[row]:.2f}s / "
                     f"arm {self.arm_speed[row]:.1f} rad/s")
            for column, panel in enumerate(self.panels(row)):
                left, top = column * 320, line * 240
                sheet.paste(panel, (left, top))
                draw.rectangle((left, top, left + 319, top + 30), fill=(24, 24, 24))
                draw.text((left + 6, top + 4), label if column == 0 else self.views[column - 1],
                          fill=(255, 255, 255))
        sheet.save(output, quality=88)
        return rows


def render_rollout_sheet(input_dir: Path, scene_path: Path, rollout_path: Path,
                         phases: list[dict], output: Path,
                         rows_override: list[int] | None = None, *, context=None) -> list[int]:
    if context is not None:
        return context.render(output, rows_override)
    with RolloutSheetRenderer(input_dir, scene_path, rollout_path, phases) as renderer:
        return renderer.render(output, rows_override)
