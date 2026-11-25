"""Deterministic fixed external views; render-only artifacts, never physics state.

Visibility is sampled with MuJoCo geometry rays, including robot occluders.
Transparent objects are conservatively treated as opaque. Sparse samples are a
selection heuristic, not a claim that every frame is free of occlusion.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .utils import digest, read_json, sha256, write_json
from .robot_runtime import runtime_contract,base_pose

VERSION = 'human-external-views-v2.0.0'
VIEW_NAMES = ('external_global', 'external_close_a', 'external_close_b')


def camera_from_spec(spec):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = spec['lookat']
    camera.distance = spec['distance']
    camera.azimuth = spec['azimuth']
    camera.elevation = spec['elevation']
    return camera


def _position(spec):
    az, el = np.deg2rad([spec['azimuth'], spec['elevation']])
    return np.asarray(spec['lookat']) + spec['distance'] * np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), -np.sin(el)])


def _robot(model, body):
    while body:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
        if name==runtime_contract(model)['base']['body']:
            return True
        body = int(model.body_parentid[body])
    return False


def _asset_hashes(xml):
    # Generated placements are flattened; include mesh/texture contents in cache.
    root = ET.parse(xml).getroot()
    compiler = root.find('compiler')
    result = {}
    for element in root.iter():
        if element.get('file'):
            base = xml.parent
            if compiler is not None:
                base /= compiler.get('meshdir' if element.tag == 'mesh' else 'texturedir', '')
            path = (base / element.get('file')).resolve()
            if path.is_file():
                result[str(path)] = sha256(path)
    return result


def _bounds(model, data, ids):
    # Local axis-aligned geom boxes provide tighter bounds than mesh spheres.
    centers = data.geom_xpos[ids]
    half = np.asarray([abs(data.geom_xmat[g].reshape(3, 3)) @ model.geom_aabb[g, 3:] for g in ids])
    centers = centers + np.asarray([data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g, :3] for g in ids])
    return np.min(centers-half, axis=0), np.max(centers+half, axis=0)


def _visible(model, data, spec, targets):
    origin = _position(spec)
    hit = np.zeros(1, dtype=np.int32)
    values = []
    for g in targets:
        center = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g, :3]
        offsets = [np.zeros(3)]
        for axis in range(3):
            delta = data.geom_xmat[g].reshape(3, 3)[:, axis] * model.geom_aabb[g, axis+3] * .65
            offsets.extend([delta, -delta])
        seen = []
        for offset in offsets:
            direction = center + offset - origin
            distance = float(np.linalg.norm(direction))
            if distance < 1e-8:
                seen.append(0.)
                continue
            direction /= distance
            ray = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, hit)
            seen.append(float(ray >= 0 and int(hit[0]) == g))
        values.append(float(np.mean(seen)))
    return values


def load_camera_plan(ctx, placement, states_path):
    """Return persisted {plan_id, plan_path, cameras, ...}; no ctx.put calls.

    `placement` is the placement payload; `states_path` is an absolute NPZ path
    with time/qpos. ctx needs root and path(relative). Cameras use 16:9 framing.
    """
    xml = Path(ctx.path(placement['scene_xml']))
    states_path = Path(states_path)
    signature = {'version': VERSION, 'scene': sha256(xml), 'assets': _asset_hashes(xml),
                 'states': sha256(states_path), 'mujoco': mujoco.__version__}
    key = digest(signature)
    relative = f'artifacts/camera_plans/{key[:24]}/plan.json'
    path = ctx.path(relative)
    if Path(path).is_file():
        cached = read_json(path)
        if cached.get('signature') == signature:
            return cached
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    with np.load(states_path) as archive:
        qpos = archive['qpos']
        times = archive['time']
    if not len(qpos):
        raise ValueError('camera selection requires recorded states')
    indices = np.unique(np.linspace(0, len(qpos)-1, min(8, len(qpos))).round().astype(int))
    finite = [g for g in range(model.ngeom) if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE and model.geom_rgba[g, 3] > 0]
    task = [g for g in finite if not _robot(model, int(model.geom_bodyid[g])) and np.max(model.geom_aabb[g, 3:]) < .5]
    if not task:
        task = [g for g in finite if not _robot(model, int(model.geom_bodyid[g]))]
    if not task or not finite:
        raise ValueError('no task geometry available for external camera framing')
    close_bounds, global_bounds = [], []
    for index in indices:
        data.qpos[:] = qpos[index]; mujoco.mj_forward(model, data)
        close_bounds.append(_bounds(model, data, task))
        global_bounds.append(_bounds(model, data, finite))
    def candidates(bounds, elevations):
        lower = np.min([b[0] for b in bounds], axis=0)
        upper = np.max([b[1] for b in bounds], axis=0)
        center = (lower+upper)/2
        radius = max(float(np.linalg.norm(upper-lower)/2), .12)
        # Fit enclosing sphere into vertical FOV with 12% margin, independent of direction.
        distance = radius / np.sin(np.deg2rad(float(model.vis.global_.fovy)/2)) * 1.12
        return [{'lookat': center.tolist(), 'distance': distance, 'azimuth': float(az), 'elevation': float(el)} for el in elevations for az in range(0, 360, 45)]
    close = candidates(close_bounds, (-25, -45, -60))
    global_ = candidates(global_bounds, (-30, -45))
    def score(spec):
        samples = []
        for index in indices:
            data.qpos[:] = qpos[index]; mujoco.mj_forward(model, data)
            samples.append(_visible(model, data, spec, task))
        a = np.asarray(samples)
        return float(.65*a.mean()+.35*np.min(a.mean(axis=1))), a
    ranked = [(score(spec), spec) for spec in close]
    ranked.sort(key=lambda item: item[0][0], reverse=True)
    first_score, first = ranked[0]
    alternatives = [item for item in ranked if 90 <= abs((item[1]['azimuth']-first['azimuth']+180)%360-180) <= 180]
    second_score, second = max(alternatives, key=lambda item: float(np.maximum(first_score[1], item[0][1]).mean()) + .2*item[0][0])
    globals_ranked = [(score(spec), spec) for spec in global_]
    global_score, global_spec = max(globals_ranked, key=lambda item: item[0][0])
    station_views=[]
    if runtime_contract(model)['motion_mode']=='mobile':
        # Fixed close views at the two farthest observed stopping regions. Use
        # actual recorded base states, not the human head path as robot motion.
        poses=[]
        for index in range(len(qpos)):
            data.qpos[:]=qpos[index];poses.append(base_pose(model,data))
        poses=np.asarray(poses)
        speed=np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1)/np.maximum(np.diff(times),1e-8)
        stops=np.flatnonzero(np.r_[speed<.005,True])
        if len(stops):
            anchor=stops[0];other=stops[np.argmax(np.linalg.norm(poses[stops,:2]-poses[anchor,:2],axis=1))]
            for anchor in (anchor,other):
                nearby=stops[np.linalg.norm(poses[stops,:2]-poses[anchor,:2],axis=1)<.2]
                sample=nearby[np.linspace(0,len(nearby)-1,min(4,len(nearby))).round().astype(int)]
                bounds=[];target_sets=[]
                for index in sample:
                    data.qpos[:]=qpos[index];mujoco.mj_forward(model,data)
                    near=[g for g in task if np.linalg.norm(data.geom_xpos[g,:2]-poses[anchor,:2])<1.25]
                    if not near:near=task
                    bounds.append(_bounds(model,data,near));target_sets.append(near)
                options=candidates(bounds,(-25,-45,-60))
                def station_score(spec):
                    values=[]
                    for index,targets in zip(sample,target_sets):
                        data.qpos[:]=qpos[index];mujoco.mj_forward(model,data)
                        values.extend(_visible(model,data,spec,targets))
                    return float(np.mean(values))
                chosen=max(options,key=station_score)
                station_views.append((chosen,station_score(chosen)))
            first,second=station_views[0][0],station_views[1][0]
            first_score=(station_views[0][1],None);second_score=(station_views[1][1],None)
    result = {'plan_id': key[:24], 'plan_path': relative, 'signature': signature,
              'cameras': dict(zip(VIEW_NAMES, [global_spec, first, second])),
              'motion_mode':runtime_contract(model)['motion_mode'], 'close_view_policy':'two observed station regions' if station_views else 'complementary task views',
              'sample_times_s': [float(times[i]) for i in indices],
              'target_geoms': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g) for g in task],
              'visibility_scores': dict(zip(VIEW_NAMES, [global_score[0], first_score[0], second_score[0]])),
              'limitations': 'Geometry-ray visibility at up to eight recorded states; transparent geometry conservatively opaque; sparse visibility cannot guarantee every frame. Close scope excludes nonrobot geometry with half-extent >= 0.5 m; global includes all finite visible geometry. Cameras are fixed for the entire recording.'}
    write_json(path, result)
    return result
