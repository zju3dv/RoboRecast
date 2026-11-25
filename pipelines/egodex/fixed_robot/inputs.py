"""Offline EgoDex export. Runner consumes this format and never converts raw data."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import os
import h5py
import numpy as np
from .utils import read_json, write_json, sha256, digest, rigid_transform

CONTRACT_VERSION = '1.0.0'
# ARKit origin is right-handed Y-up; simulation is right-handed Z-up.
DEFAULT_ALIGNMENT = np.array([[1.,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
CAVEAT = ('Vision Pro RGB is synthesized; calibrated reprojections may differ from visible joints. '
          'Hand centers/openings are geometric references, not contact or object ground truth.')


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _rigid_mask(transforms):
    t = np.asarray(transforms, dtype=float)
    finite = np.isfinite(t).all(axis=(-2, -1))
    safe = np.where(np.isfinite(t), t, 0.)
    r = safe[:, :3, :3]
    orthogonal = np.max(np.abs(np.swapaxes(r, 1, 2) @ r - np.eye(3)), axis=(1, 2)) < 2e-3
    return finite & orthogonal & (np.abs(np.linalg.det(r) - 1.) < 2e-3) & (np.max(np.abs(safe[:, 3, :] - [0, 0, 0, 1]), axis=1) < 1e-5)


def _video_info(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                             '-show_streams', '-show_entries',
                             'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    probe = json.loads(result.stdout)
    stream = probe['streams'][0]
    numerator, denominator = map(float, stream['avg_frame_rate'].split('/'))
    fps = numerator / denominator
    times = np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']], dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Video requires at least two frames with strictly increasing presentation timestamps')
    if abs(fps - 30.) > 0.02 or not np.allclose(np.diff(times), 1 / 30., atol=1e-4):
        raise ValueError('EgoDex v1 requires the original constant 30 Hz source video')
    return stream, times, fps


def convert_episode(mp4, hdf5, output, T_sim_source_world=None):
    """Prepare an official, same-task MP4/HDF5 pair at original timestamps."""
    mp4, hdf5, output = Path(mp4).resolve(), Path(hdf5).resolve(), Path(output).resolve()
    episode_id = f"{mp4.parent.name}/{mp4.stem}"
    if mp4.stem != hdf5.stem or mp4.parent.name != hdf5.parent.name:
        raise ValueError('EgoDex RGB and pose files must have matching task/index names')
    if output.exists():
        raise FileExistsError(f'Export already exists; create a new input revision: {output}')
    alignment = np.asarray(rigid_transform(DEFAULT_ALIGNMENT if T_sim_source_world is None else T_sim_source_world), dtype=float)
    config = {'thumb_tip': 'ThumbTip', 'opposing_tips': ['IndexFingerTip'], 'min_confidence': 0.0}
    if not isinstance(config['opposing_tips'], list) or not config['opposing_tips'] or not all(isinstance(x, str) for x in config['opposing_tips']):
        raise ValueError('opposing_tips must be a nonempty list of joint suffixes')
    if not 0 <= float(config['min_confidence']) <= 1:
        raise ValueError('min_confidence must be in [0,1]')
    stream, pts, fps = _video_info(mp4)
    n = len(pts)
    arrays = {'timestamps': pts - pts[0], 'video_pts': pts, 'source_frames': np.arange(n, dtype=np.int64)}
    raw = {}
    with h5py.File(hdf5, 'r') as f:
        attrs = {k: _json_value(v) for k, v in f.attrs.items()}
        K = np.asarray(f['camera/intrinsic'], dtype=float)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError('Invalid source camera intrinsic')
        arrays['K'] = K
        for name, dataset in f['transforms'].items():
            value = np.asarray(dataset, dtype=np.float64)
            if value.shape != (n, 4, 4):
                raise ValueError(f'RGB/HDF5 count or shape mismatch: transforms/{name} {value.shape}, RGB {n}')
            raw[f'transform__{name}'] = value
            raw[f'valid__{name}'] = _rigid_mask(value)
            conf = np.full(n, np.nan)
            if 'confidences' in f and name in f['confidences']:
                conf = np.asarray(f['confidences'][name], dtype=float)
                if conf.shape != (n,) or np.any(np.isfinite(conf) & ((conf < 0) | (conf > 1))) or np.isinf(conf).any():
                    raise ValueError(f'Invalid confidence array for {name}')
            raw[f'confidence__{name}'] = conf
        arrays['T_world_camera'] = raw['transform__camera']
        if not raw['valid__camera'].all():
            raise ValueError('Camera poses must be finite rigid transforms at every source frame')
        for side in ('left', 'right'):
            names = [side + config['thumb_tip']] + [side + tip for tip in config['opposing_tips']] + [side + 'Hand']
            for name in names:
                if f'transform__{name}' not in raw:
                    raise ValueError(f'Missing required reference joint {name}')
            thumb = raw[f'transform__{names[0]}'][:, :3, 3]
            opposing = np.mean([raw[f'transform__{name}'][:, :3, 3] for name in names[1:-1]], axis=0)
            wrist = raw[f'transform__{side}Hand'][:, :3, 3]
            center = (thumb + opposing) / 2
            opening = np.linalg.norm(opposing - thumb, axis=1)
            direction = center - wrist
            length = np.linalg.norm(direction, axis=1)
            confs = np.stack([raw[f'confidence__{name}'] for name in names])
            # Unknown constituent confidence makes the derived aggregate unknown.
            confidence = np.min(confs, axis=0)
            known_bad = np.any(np.isfinite(confs) & (confs <= float(config['min_confidence'])), axis=0)
            valid = np.all([raw[f'valid__{name}'] for name in names], axis=0) & ~known_bad
            direction_valid = valid & (length > 1e-6)
            direction = np.divide(direction, length[:, None], out=np.full_like(direction, np.nan), where=direction_valid[:, None])
            for key, value in [('center', center), ('wrist', wrist), ('opening', opening)]:
                value = value.copy()
                value[~valid] = np.nan
                arrays[f'{side}_{key}'] = value
            arrays[f'{side}_direction'] = direction
            arrays[f'{side}_direction_valid'] = direction_valid
            arrays[f'{side}_confidence'] = confidence
            arrays[f'{side}_valid'] = valid
            arrays[f'{side}_confidence_known'] = np.isfinite(confidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    try:
        (stage / 'source').mkdir()
        shutil.copyfile(mp4, stage / 'source/rgb.mp4')
        shutil.copyfile(hdf5, stage / 'source/poses.hdf5')
        np.savez_compressed(stage / 'reference.npz', **arrays)
        np.savez_compressed(stage / 'raw_joints.npz', **raw)
        which = attrs.get('which_llm_description', 1)
        selected = 'llm_description2' if str(which) == '2' else 'llm_description'
        meta = {
            'contract_version': CONTRACT_VERSION, 'episode_id': episode_id,
            'fps': fps, 'frame_count': n, 'width': int(stream['width']), 'height': int(stream['height']),
            'duration': n / fps, 'last_timestamp': float(arrays['timestamps'][-1]),
            'raw_file_hashes': {'mp4': sha256(mp4), 'hdf5': sha256(hdf5)},
            'source_paths': {'mp4': str(mp4), 'hdf5': str(hdf5)},
            'T_sim_source_world': alignment.tolist(), 'base_mode': 'fixed', 'route_reference': None,
            'source_conventions': {'world': 'episode-local ARKit origin, right-handed Y-up, metres',
                'camera': 'T_world_camera maps camera to source world; +X right, +Y down, +Z forward',
                'projection': 'pixel homogeneous ~ K @ inverse(T_world_camera) @ point_world',
                'timestamps': 'original video presentation timestamps minus first PTS; no temporal resampling',
                'joint_transform': 'T_world_joint; raw transforms and optional confidence preserved',
                'authority': 'https://github.com/apple-aiml-research/ml-egodex', 'caveat': CAVEAT},
            'task_text': {'selected_attribute': selected, 'text': attrs.get(selected, attrs.get('description')), 'status': 'weak_prior_only'},
            'source_attributes': attrs, 'reference_extraction': config,
            'processing': {'cropped': False, 'filtered': False, 'interpolated': False, 'position_frame': 'source_world',
                'center': 'mean(thumb_tip, mean(opposing_tips))', 'direction': 'normalized(center - wrist), not full grasp orientation',
                'opening': 'distance(thumb_tip, mean(opposing_tips)), not robot gripper command',
                'confidence': 'minimum constituent confidence; unknown if any missing; zero is invalid'},
            'quality': {side: {'valid_frames': int(arrays[f'{side}_valid'].sum()),
                        'unknown_confidence_frames': int((~arrays[f'{side}_confidence_known']).sum())} for side in ('left', 'right')},
            'files': {name: sha256(stage / name) for name in ['source/rgb.mp4', 'source/poses.hdf5', 'reference.npz', 'raw_joints.npz']},
        }
        meta['input_fingerprint'] = digest(meta)
        write_json(stage / 'meta.json', meta)
        os.rename(stage, output)
    except Exception:
        shutil.rmtree(stage)
        raise
    return meta


def load_reference(input_dir):
    """Load and verify a completed offline export; never convert or repair in place."""
    root = Path(input_dir)
    meta = read_json(root / 'meta.json')
    if meta.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Expected a prepared EgoDex episode")
    with np.load(root / 'reference.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if len(arrays['timestamps']) != meta['frame_count']:
        raise ValueError('Input frame count mismatch')
    meta = dict(meta, motion_mode='not_mobile')
    return meta, arrays
