"""Recorded-state evidence adapted from mobile's shared-panel sheet workflow.

One source RGB stream is temporal context. Five robot/external views establish
physics; source-camera alignment is deliberately not an acceptance condition.
"""
from pathlib import Path
import cv2
import mujoco
import numpy as np
from .inputs import load_reference
from .render_utils import _label, _Video
from .camera_views import load_camera_plan, camera_from_spec
from .utils import write_json, digest, sha256

SETTINGS = {'version': '2.0.0', 'width': 320, 'height': 180, 'clip_fps': 15,
            'views': ['source', 'left_wrist', 'right_wrist', 'external_global',
                      'external_close_a', 'external_close_b'], 'layout': [2, 3],
            'default_motion_clips': False}


def sample_plan(states, trajectory, diagnostics, probe_phase=None):
    times = states['time']; end = float(times[-1]); start = float(times[0])
    phases = trajectory['time_map']
    if probe_phase:
        phases = [p for p in phases if p['phase'] == probe_phase]
    samples = []
    def add(t, reason):
        i = int(np.argmin(abs(times - np.clip(t, start, end))))
        samples.append({'row': i, 'reason': reason})
    transitions = []
    for p in phases:
        for key in ('sim_start', 'sim_end'):
            t = float(p[key])
            if start <= t <= end + 1e-6:
                label = p['phase'] + ':' + key
                transitions.append({'sim_time': min(t, end), 'reason': label,
                                    'hand_objects': p.get('hand_objects', {})})
                add(t, label)
        middle = (p['sim_start'] + min(p['sim_end'], end)) / 2
        add(middle, p['phase'] + ':midpoint')
        transitions.append({'sim_time': middle, 'reason': p['phase'] + ':midpoint-motion'})
        for event in p.get('events', []):
            t = float(event['sim_time'])
            if start <= t <= end:
                reason = p['phase'] + ':event:' + str(event.get('name', event.get('kind', 'declared')))
                transitions.append({'sim_time': t, 'reason': reason}); add(t, reason)
    local_start = phases[0]['sim_start'] if probe_phase else start
    source_values, source_indices = np.unique(states['source_time'], return_index=True)
    for event in diagnostics.get('thresholds', {}).get('events', []):
        for key in ('source_start', 'source_end'):
            t = float(np.interp(event[key], source_values, times[source_indices]))
            if local_start <= t <= end:
                reason = 'success-event:' + event['name'] + ':' + key
                transitions.append({'sim_time': t, 'reason': reason}); add(t, reason)
    for event in diagnostics.get('events', []):
        failure = event.get('first_failure') or {}
        if 'sim_time' in failure and local_start <= float(failure['sim_time']) <= end:
            t = float(failure['sim_time']); reason = 'event-failure:' + event['name']
            transitions.append({'sim_time': t, 'reason': reason}); add(t, reason)
    if not probe_phase:
        add(start, 'initial'); add(end, 'terminal')
    failure = diagnostics.get('first_failure')
    if failure and (not probe_phase or float(failure.get('sim_time', end)) >= phases[0]['sim_start']):
        t = float(failure.get('sim_time', end))
        for dt in (-.25, 0, .25): add(t + dt, 'failure%+.2fs' % dt)
    for window in diagnostics.get('contact_force_summary', {}).get('evidence_windows', []):
        entries = [('start', window['start']), ('end', window['end'])]
        entries += [('peak', t) for t in window.get('peak_times', [window['peak_time']])]
        for key, value in entries:
            t = float(value)
            if local_start <= t <= end:
                add(t, 'contact-force:' + str(window.get('reason', 'peak')) + ':' + key)
    grouped = {}
    for sample in samples: grouped.setdefault(sample['row'], []).append(sample['reason'])
    return [{'row': i, 'reasons': reasons} for i, reasons in sorted(grouped.items())], transitions


class EvidenceRenderer:
    def __init__(self, ctx, placement, states_path, width=320, height=180):
        self.ctx = ctx
        self.model = mujoco.MjModel.from_xml_path(str(ctx.path(placement['scene_xml'])))
        self.data = mujoco.MjData(self.model)
        with np.load(states_path) as z: self.states = {k: z[k].copy() for k in z.files}
        _, self.reference = load_reference(ctx.input_dir)
        self.capture = cv2.VideoCapture(str(ctx.input_dir / 'source/rgb.mp4'))
        if not self.capture.isOpened(): raise ValueError('source RGB unavailable')
        self.width, self.height = int(width), int(height)
        self.camera_plan = load_camera_plan(ctx, placement, states_path)
        self.cameras = {name: camera_from_spec(spec) for name, spec in self.camera_plan['cameras'].items()}
        from .robot_runtime import runtime_contract
        self.wrist_cameras = runtime_contract(self.model)['cameras']
        for name in self.wrist_cameras.values():
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError('required physical-evidence camera missing: ' + name)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
        self.renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
        key = digest({'scene': sha256(ctx.path(placement['scene_xml'])), 'states': sha256(states_path),
                      'plan': self.camera_plan['plan_id'], 'settings': SETTINGS,
                      'width': self.width, 'height': self.height,
                      'source': sha256(ctx.input_dir / 'source/rgb.mp4'),
                      'input_signature': ctx._input_signature()})
        self.cache_dir = ctx.path(Path('evidence_cache') / key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}; self.source_cache = {}; self.next_source_frame = None

    def close(self):
        self.renderer.close(); self.capture.release()

    def mapping(self, i):
        source_time = float(self.states['source_time'][i])
        si = int(np.argmin(abs(self.reference['timestamps'] - source_time)))
        return {'row': int(i), 'sim_time': float(self.states['time'][i]),
                'source_time': source_time, 'exported_frame': si, 'source_frame': int(self.reference['source_frames'][si])}

    def views(self, i):
        i = int(i)
        if i in self.cache: return self.cache[i]
        path = self.cache_dir / ('views_%07d.npz' % i)
        if path.exists():
            with np.load(path) as z: result = {k: z[k].copy() for k in SETTINGS['views']}
        else:
            m = self.mapping(i); frame = m['exported_frame']
            if frame not in self.source_cache:
                if self.next_source_frame != frame: self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, bgr = self.capture.read(); self.next_source_frame = frame + 1
                if not ok: raise ValueError('cannot decode source frame %s' % frame)
                scale = min(self.width / bgr.shape[1], self.height / bgr.shape[0])
                w, h = max(1, round(bgr.shape[1] * scale)), max(1, round(bgr.shape[0] * scale))
                source = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                x, y = (self.width - w) // 2, (self.height - h) // 2
                source[y:y+h, x:x+w] = cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB)
                self.source_cache[frame] = source
                if len(self.source_cache) > 128: self.source_cache.pop(next(iter(self.source_cache)))
            result = {'source': self.source_cache[frame]}
            for key in ('qpos', 'qvel', 'ctrl'): getattr(self.data, key)[:] = self.states[key][i]
            self.data.time = m['sim_time']; mujoco.mj_forward(self.model, self.data)
            cameras = {**self.wrist_cameras, **self.cameras}
            for name, camera in cameras.items():
                self.renderer.update_scene(self.data, camera=camera)
                result[name] = self.renderer.render().copy()
            np.savez_compressed(path, **result)
        if len(self.cache) >= 24: self.cache.pop(next(iter(self.cache)))
        self.cache[i] = result
        return result

    def panel(self, i):
        m = self.mapping(i); views = self.views(i)
        panels = [_label(views[name].copy(),
                         ('source f%d %.2fs' % (m['source_frame'], m['source_time'])) if name == 'source'
                         else '%s %.2fs' % (name, m['sim_time'])) for name in SETTINGS['views']]
        return np.concatenate([np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)], axis=0)

    def sheets(self, directory, samples):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        sheets = []
        for n, sample in enumerate(samples):
            panel = self.panel(sample['row'])
            reason = ' | '.join(sample.get('reasons', ['requested']))
            # Separate header preserves all six camera labels and records temporal context.
            header = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
            cv2.putText(header, reason[:150], (7, 19), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            path = directory / ('keyframes_%03d.jpg' % n)
            cv2.imwrite(str(path), cv2.cvtColor(np.concatenate([header, panel]), cv2.COLOR_RGB2BGR))
            sheets.append(str(path.relative_to(self.ctx.root)))
        return sheets


def build_rollout_bundle(ctx, placement, trajectory, states_path, diagnostics, directory, probe_phase=None):
    directory = ctx.path(directory); directory.mkdir(parents=True, exist_ok=True)
    renderer = EvidenceRenderer(ctx, placement, states_path)
    try:
        samples, transitions = sample_plan(renderer.states, trajectory, diagnostics, probe_phase)
        sheets = renderer.sheets(directory, samples)
        mapped = [{**renderer.mapping(s['row']), 'reasons': s['reasons'], 'sheet': sheet}
                  for s, sheet in zip(samples, sheets)]
        camera_plan = renderer.camera_plan
    finally: renderer.close()
    packet = {'schema': 'human-rollout-evidence-v2.0', 'version': '2.0.0', 'settings': SETTINGS,
              'states_npz': str(Path(states_path).relative_to(ctx.root)), 'keyframes': sheets,
              'samples': mapped, 'camera_plan': camera_plan, 'transitions': transitions, 'probe_phase': probe_phase,
              'summary': {'mobility': diagnostics.get('mobility'), 'status': diagnostics['status'], 'first_failure': diagnostics.get('first_failure'),
                          'collision_classes': diagnostics.get('collision_classes', {}),
                          'contact_force_summary': diagnostics.get('contact_force_summary', {}),
                          'hand_objects': diagnostics.get('hand_object_monitoring', {}),
                          'events': diagnostics.get('events', []), 'terminal_checks': diagnostics.get('terminal_checks', []),
                          'thresholds': diagnostics.get('thresholds', {}),
                          'maximum_penetration_m': diagnostics.get('maximum_penetration_m'),
                          'max_tracking_error_rad': diagnostics.get('max_tracking_error_rad')},
              'source_policy': 'One RGB temporal reference; source-camera alignment and robot occlusion are nonblocking.'}
    path = directory / 'bundle.json'; write_json(path, packet)
    return {**packet, 'bundle_json': str(path.relative_to(ctx.root))}
