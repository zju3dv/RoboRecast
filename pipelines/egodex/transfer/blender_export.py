"""Export evaluated MuJoCo visual geometry and poses; never re-simulate in Blender."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from PIL import Image
def effective_intrinsics(model, camera_id, width, height):
    """Actual rendered K at requested resolution, including legacy cameras."""
    size = np.asarray(model.cam_sensorsize[camera_id], float)
    if np.all(size > 0):
        intrinsic = np.asarray(model.cam_intrinsic[camera_id], float)
        fx, fy = intrinsic[:2] / size * [width, height]
        cx, cy = (np.asarray([width, height])-1)/2 - intrinsic[2:] / size * [width, height]
    else:
        fy = height / (2*np.tan(np.deg2rad(model.cam_fovy[camera_id])/2))
        fx = fy; cx = (width-1)/2; cy = (height-1)/2
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])


def export(scene: Path, rollout: Path, output: Path, fps: float = 24, max_frames: int = 0,
           requested_times: list[float] | None = None):
    started = time.monotonic()
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    with np.load(rollout, allow_pickle=False) as source:
        ts, qpos = source['timestamp_sec'], source['qpos']
    if fps <= 0 or qpos.shape != (len(ts), model.nq) or np.any(np.diff(ts) <= 0):
        raise ValueError('Expected positive fps and monotonic rollout with model.nq positions')
    if requested_times is not None:
        target = np.asarray(requested_times, dtype=float)
        if max_frames or not len(target) or not np.isfinite(target).all() or np.any(target < ts[0]-1e-6) or np.any(target > ts[-1]+1e-6):
            raise ValueError('Requested times must be within the rollout; do not combine with max_frames')
        right = np.minimum(np.searchsorted(ts, target), len(ts)-1)
        left = np.maximum(right-1, 0)
        ids = np.unique(np.where(abs(ts[left]-target) <= abs(ts[right]-target), left, right))
    else:
        target = np.arange(ts[0], ts[-1] + 1e-8, 1 / fps)
        ids = np.minimum(np.searchsorted(ts, target), len(ts) - 1)
        if max_frames:
            ids = ids[np.unique(np.linspace(0, len(ids)-1, max_frames).astype(int))]
    output.mkdir(parents=True, exist_ok=True)
    arrays, geoms = {}, []
    for i in range(model.ngeom):
        # Existing models distinguish collision proxies (group 3+) from visible surfaces.
        if model.geom_group[i] >= 3 or model.geom_rgba[i, 3] == 0:
            continue
        material = int(model.geom_matid[i])
        rgba = (model.mat_rgba[material] if material >= 0 else model.geom_rgba[i]).tolist()
        item = dict(id=i, name=model.geom(i).name or f'geom_{i}', type=int(model.geom_type[i]),
                    size=model.geom_size[i].tolist(), rgba=rgba, material_id=material)
        if material >= 0:
            roughness = float(model.mat_roughness[material])
            item['roughness'] = roughness if roughness >= 0 else max(.18, 1 - float(model.mat_shininess[material]))
            item['metallic'] = max(0., float(model.mat_metallic[material])) if hasattr(model, 'mat_metallic') else 0.
            texture = int(model.mat_texid[material, 1])
            if texture >= 0 and model.tex_type[texture] == mujoco.mjtTexture.mjTEXTURE_2D:
                w, h, channels = int(model.tex_width[texture]), int(model.tex_height[texture]), int(model.tex_nchannel[texture])
                adr = int(model.tex_adr[texture])
                pixels = model.tex_data[adr:adr+w*h*channels].reshape(h,w,channels)
                filename = f'texture_{texture}.png'
                Image.fromarray(np.flipud(pixels)).save(output / filename)
                item['texture'] = filename
                item['texrepeat'] = model.mat_texrepeat[material].tolist()
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[i])
            v, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            f, nf = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            arrays[f'vertices_{i}'] = model.mesh_vert[v:v+nv].copy()
            arrays[f'faces_{i}'] = model.mesh_face[f:f+nf].copy()
            uv = int(model.mesh_texcoordadr[mesh])
            if uv >= 0:
                uv_ids = model.mesh_facetexcoord[f:f+nf]
                if np.all(uv_ids >= 0):
                    arrays[f'face_uv_{i}'] = model.mesh_texcoord[uv+uv_ids].copy()
        geoms.append(item)
    positions, rotations, camera_positions, camera_rotations = [], [], [], []
    for idx in ids:
        data.qpos[:] = qpos[idx]
        mujoco.mj_forward(model, data)
        positions.append(data.geom_xpos.copy())
        rotations.append(data.geom_xmat.copy())
        camera_positions.append(data.cam_xpos.copy())
        camera_rotations.append(data.cam_xmat.copy())
    arrays.update(time=ts[ids], source_indices=ids, geom_pos=positions, geom_rot=rotations,
                  camera_pos=camera_positions, camera_rot=camera_rotations)
    np.savez_compressed(output / 'states.npz', **arrays)
    cameras = []
    for i in range(model.ncam):
        camera = dict(id=i, name=model.camera(i).name or f'camera_{i}', fovy=float(model.cam_fovy[i]))
        sensor = model.cam_sensorsize[i]
        resolution = model.cam_resolution[i]
        if np.all(sensor > 0) and np.all(resolution > 0):
            width, height = map(int, resolution)
            K = effective_intrinsics(model, i, width, height)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            camera['projection'] = dict(type='calibrated_pinhole', resolution=[width, height],
                                        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                                        convention='OpenCV pixel centers; resize with (c+0.5)*scale-0.5')
            camera['fovy'] = float(np.rad2deg(2*np.arctan(height/2/fy)))
        cameras.append(camera)
    metadata = dict(schema='roborecast-blender-v1', geoms=geoms, cameras=cameras, fps=fps,
                    sampling_mode='requested_times' if requested_times is not None else 'sparse_preview' if max_frames else 'video_fps',
                    requested_times_sec=requested_times, actual_times_sec=ts[ids].tolist(),
                    frame_count=len(ids), scene=str(scene.resolve()), rollout=str(rollout.resolve()),
                    scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                    rollout_sha256=hashlib.sha256(rollout.read_bytes()).hexdigest(),
                    export_seconds=time.monotonic()-started, excluded_from_task_time=True,
                    appearance='Compiled visual geometry, base colors, metallic/roughness factors and compatible 2D RGB textures. Cube textures are omitted.',
                    motion='Evaluated rollout poses without Blender physics or object motion edits.')
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=float, default=24)
    p.add_argument('--max-frames', type=int, default=0, help='Sparse preview only; zero exports full video timing')
    p.add_argument('--times', type=float, nargs='+', help='Sparse event times in rollout seconds; select nearest actual states')
    a = p.parse_args()
    print(json.dumps(export(a.scene, a.rollout, a.output, a.fps, a.max_frames, a.times), indent=2))


if __name__ == '__main__':
    main()
