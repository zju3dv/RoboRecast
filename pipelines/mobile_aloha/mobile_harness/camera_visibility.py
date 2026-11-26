"""Presentation-only external camera selection; no changes to simulation state."""
import copy

import mujoco
import numpy as np

from .rendering import presentation_options


class ExternalCameraSelector:
    """Keep a visible angle; reconsider every 0.5 s with a one-second dwell."""

    def __init__(self):
        self.choice = None
        self.checked_at = -np.inf
        self.switched_at = -np.inf
        self.phase = None
        self.events = []

    @staticmethod
    def targets(model, data, focus):
        targets = []
        for name in ('base', 'tracer/base'):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                targets.append((data.xpos[bid].copy(), bid, 1.0))
                break
        for sid in range(model.nsite):
            name = model.site(sid).name or ''
            if name.endswith('/pinch'):
                targets.append((data.site_xpos[sid].copy(), int(model.site_bodyid[sid]), 1.0))
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, focus) if focus else -1
        if bid >= 0:
            targets.append((data.xpos[bid].copy(), bid, 2.0))
        return targets

    @staticmethod
    def score(model, data, renderer, camera, targets):
        renderer.update_scene(data, camera=camera, scene_option=presentation_options())
        gl = renderer.scene.camera[0]
        eye = np.mean([c.pos for c in renderer.scene.camera], axis=0).astype(float)
        forward, up = np.asarray(gl.forward), np.asarray(gl.up)
        right = np.cross(forward, up)
        half_y = float((gl.frustum_top - gl.frustum_bottom) / (2 * gl.frustum_near))
        half_x = half_y * renderer.width / renderer.height
        score = 0.0
        groups = presentation_options().geomgroup
        for point, body, weight in targets:
            delta = point - eye
            depth = float(delta @ forward)
            if depth <= 0 or abs(delta @ up) > .92 * depth * half_y or abs(delta @ right) > .92 * depth * half_x:
                continue
            distance = float(np.linalg.norm(delta))
            gid = np.array([-1], dtype=np.int32)
            hit = mujoco.mj_ray(model, data, eye, delta / distance, groups, 1, -1, gid)
            hit_body = int(model.geom_bodyid[gid[0]]) if gid[0] >= 0 else -1
            # Accept the target body's own visible surface (including child geometry).
            while hit_body > 0 and hit_body != body:
                hit_body = int(model.body_parentid[hit_body])
            if hit < 0 or hit >= distance - .08 or hit_body == body:
                score += weight
        return score / max(sum(t[2] for t in targets), 1.0)

    def select(self, model, data, renderer, camera, focus=None, *, time=0.0, phase=None):
        camera = copy.copy(camera)
        if self.choice is not None:
            camera.azimuth, camera.elevation, camera.distance = self.choice
        changed_phase = phase != self.phase
        if not changed_phase and time - self.checked_at < .5:
            return camera
        self.checked_at, self.phase = time, phase
        targets = self.targets(model, data, focus)
        if not targets:
            return camera
        current = self.score(model, data, renderer, camera, targets)
        if current >= .99:
            self.choice = (camera.azimuth, camera.elevation, camera.distance)
            return camera
        if self.choice is not None and time - self.switched_at < 1.0:
            return camera
        best, best_score = camera, current
        for elevation in (camera.elevation, -55.0):
            for offset in (0, 45, -45, 90, -90, 135, -135, 180):
                candidate = copy.copy(camera)
                candidate.azimuth = (camera.azimuth + offset) % 360
                candidate.elevation = elevation
                # Slightly wider fallback also protects framing around large objects.
                candidate.distance = max(camera.distance, 2.25) if elevation == -55 else camera.distance
                score = self.score(model, data, renderer, candidate, targets)
                if score > best_score + .01:
                    best, best_score = candidate, score
        if best_score > current + .15:
            self.switched_at = time
            self.events.append(dict(time_sec=float(time), phase=phase, before=current,
                                    after=best_score, azimuth=float(best.azimuth)))
            camera = best
        self.choice = (camera.azimuth, camera.elevation, camera.distance)
        return camera
