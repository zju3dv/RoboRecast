"""Shared physical-time sampling and presentation options."""
import mujoco
import numpy as np


def presentation_renderer(model, *, height, width):
    # MuJoCo znear is scaled by scene extent. Keep a physical 3 mm near plane
    # so large rooms do not slice away objects in wrist-camera close-ups.
    model.vis.map.znear = .003 / max(float(model.stat.extent), 1e-6)
    return mujoco.Renderer(model, height=height, width=width)


def presentation_options() -> mujoco.MjvOption:
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[3] = 0  # collision-only proxies
    return options


def sample_rows(timestamps: np.ndarray, fps: float = 10.0) -> np.ndarray:
    times = np.asarray(timestamps, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('recorded timestamps must strictly increase')
    # ceil(duration * fps) frames; final state appears in the final frame.
    count = max(2, int(np.ceil((times[-1] - times[0]) * fps)))
    samples = times[0] + np.arange(count) / fps
    rows = np.searchsorted(times, samples, side='right') - 1
    rows[-1] = len(times) - 1
    return np.clip(rows, 0, len(times) - 1)


def motion_review_clips(model, qpos, timestamps, phases, output_dir):
    """A few short physical-time overview clips, using the final renderer policy."""
    from .finalize import _writer, _finish_writer
    from .stages import overview_camera
    from .camera_visibility import ExternalCameraSelector
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in phases if p.get('action') in ('operate_arm','carry_with_base','push_with_base')]
    selected = [candidates[i] for i in sorted(set((0,len(candidates)//2,len(candidates)-1)))] if candidates else []
    renderer = presentation_renderer(model, height=240, width=320)
    data = mujoco.MjData(model)
    result = []
    try:
        for index, phase in enumerate(selected):
            start,end = int(phase['start_row']),int(phase['end_row'])
            if end - start < 2: continue
            # Preserve true playback speed. Select the central 8 seconds if long.
            center = (timestamps[start] + timestamps[end-1]) / 2
            start = max(start, int(np.searchsorted(timestamps, center-4)))
            end = min(end, int(np.searchsorted(timestamps, center+4, side='right')))
            path=output_dir/f'motion_{index}.mp4'
            writer=_writer(path,'320x240')
            selector = ExternalCameraSelector()
            try:
                for row in start + sample_rows(timestamps[start:end]):
                    data.qpos[:]=qpos[row];mujoco.mj_forward(model,data)
                    view = selector.select(model, data, renderer, overview_camera(model, data, phase.get('object')),
                                           phase.get('object'), time=float(timestamps[row]), phase=phase['name'])
                    renderer.update_scene(data,camera=view,
                                          scene_option=presentation_options())
                    writer.stdin.write(cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR).tobytes())
                _finish_writer(writer)
            except Exception:
                writer.kill();writer.wait();raise
            result.append({'phase':phase['name'],'path':str(path),'physical_speed':True,
                           'start_sec':float(timestamps[start]),'end_sec':float(timestamps[end-1])})
    finally:
        renderer.close()
    return result
