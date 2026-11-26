from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import mujoco
from .rendering import presentation_renderer, presentation_options, sample_rows
import numpy as np

from mobile_input.loader import load_episode
from .stages import overview_camera
from .camera_visibility import ExternalCameraSelector
from .store import Store


def _label(frame: np.ndarray, label: str) -> np.ndarray:
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 31), (28, 28, 28), -1)
    cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return frame


def _writer(path: Path, size: str) -> subprocess.Popen:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", size, "-r", "10", "-i", "-",
               "-an", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _finish_writer(process: subprocess.Popen) -> None:
    assert process.stdin is not None
    process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(error[-2000:])


def render_comparison(input_dir: Path, store: Store, output_dir: Path) -> Path:
    state = store.read()
    if state["stage"] != "REVIEW_PASS":
        raise ValueError("Accepted owner review required")
    rollout = state["rollouts"][-1]
    merge = state["merges"][-1]
    model = mujoco.MjModel.from_xml_path(str(store.artifact(merge["scene_xml_path"])))
    data = mujoco.MjData(model)
    with np.load(store.artifact(rollout["rollout_path"])) as record:
        qpos = np.asarray(record["qpos"])
        timestamps = np.asarray(record["timestamp_sec"])
        ctrl = np.asarray(record["ctrl"])
        phase_index = np.asarray(record["phase_index"])
    phases = json.loads(store.artifact(state["trajectories"][-1]["source_spec_path"]).read_text())["phases"]
    episode = load_episode(input_dir)
    camera_pairs = [(role, episode.runtime_contract["cameras"][role])
                    for role in ("top", "left_wrist", "right_wrist")]
    overview = ("route_overview", "follow", "interaction_overview")
    selectors = {name: ExternalCameraSelector() for name in overview}
    captures = [cv2.VideoCapture(str(episode.camera_paths[role]))
                for role, _ in camera_pairs]
    output = output_dir / "comparison.mp4"
    process = _writer(output, "960x720")
    renderer = presentation_renderer(model, height=240, width=320)
    try:
        assert process.stdin is not None
        for row in sample_rows(timestamps):
            data.qpos[:] = qpos[row]
            data.ctrl[:] = ctrl[row]
            mujoco.mj_forward(model, data)
            panels = []
            for capture, (role, _) in zip(captures, camera_pairs):
                capture.set(cv2.CAP_PROP_POS_FRAMES, row)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"real {role} video missing frame {row}")
                panels.append(_label(cv2.resize(frame, (320, 240)), f"ref {role} {episode.timestamp_sec[row]:.1f}s aligned"))
            for role, camera in camera_pairs:
                renderer.update_scene(data, camera=camera, scene_option=presentation_options())
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                panels.append(_label(frame, f"sim {role} {timestamps[row]:.1f}s"))
            for camera in overview:
                focus = phases[int(phase_index[row])].get("object")
                view = overview_camera(model, data, focus if camera != "follow" else None)
                if camera == "follow":
                    view.distance, view.azimuth, view.elevation = 2.0, 145, -35
                elif camera == "interaction_overview":
                    view.distance, view.azimuth, view.elevation = 1.45, 105, -32
                view = selectors[camera].select(
                    model, data, renderer, view, focus,
                    time=float(timestamps[row]), phase=int(phase_index[row]))
                renderer.update_scene(data, camera=view, scene_option=presentation_options())
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                panels.append(_label(frame, f"sim {camera}"))
            grid = np.concatenate([np.concatenate(panels[i:i + 3], axis=1)
                                   for i in (0, 3, 6)], axis=0)
            process.stdin.write(grid.tobytes())
        _finish_writer(process)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        renderer.close()
        for capture in captures:
            capture.release()
    return output
