"""Bounded human reference queries with source-frame overlays and trajectory plots."""
from pathlib import Path
import uuid
import cv2
import numpy as np
from .inputs import load_reference, CAVEAT
from .geometry import project_points


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _indices(meta, arrays, request):
    count = request.get('max_samples', 12)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
        raise ValueError('max_samples must be an integer from 1 to 64')
    if 'frames' in request and 'times' in request:
        raise ValueError('Use frames or times, not both')
    if 'frames' in request:
        indices = request['frames']
        if not indices or len(indices) > 64 or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= meta['frame_count'] for i in indices):
            raise ValueError('frames must contain 1..64 valid original frame indices')
        return np.array(indices, dtype=int)
    if 'times' in request:
        times = np.asarray(request['times'], float)
        if times.ndim != 1 or not 1 <= len(times) <= 64 or not np.isfinite(times).all() or np.any(times < 0) or np.any(times > arrays['timestamps'][-1]):
            raise ValueError('times must be within the observed source interval')
        return np.array([np.argmin(abs(arrays['timestamps'] - t)) for t in times])
    start, end = request.get('start', 0.), request.get('end', float(arrays['timestamps'][-1]))
    if not (0 <= start <= end <= arrays['timestamps'][-1]):
        raise ValueError('Invalid reference time interval')
    valid = np.flatnonzero((arrays['timestamps'] >= start) & (arrays['timestamps'] <= end))
    if not len(valid):
        raise ValueError('No observed frames in requested interval')
    return valid[np.unique(np.linspace(0, len(valid) - 1, min(count, len(valid))).round().astype(int))]


def _trajectory_image(arrays, indices, output):
    canvas = np.full((450, 1200, 3), 248, np.uint8)
    sides = [('left', (220, 120, 30)), ('right', (40, 100, 230))]
    all_points = np.concatenate([arrays[f'{s}_center'][indices] for s, _ in sides])
    finite = all_points[np.isfinite(all_points).all(axis=1)]
    if not len(finite):
        cv2.putText(canvas, 'No valid hand observations', (25, 100), 0, .8, (0, 0, 0), 2)
    else:
        for panel, (a, b, label) in enumerate([(0, 1, 'source XY'), (0, 2, 'source XZ'), (2, 1, 'source ZY')]):
            low, high = finite[:, [a, b]].min(axis=0), finite[:, [a, b]].max(axis=0)
            extent = max(float(np.max(high - low)), 0.1)
            center = (low + high) / 2
            cv2.putText(canvas, label + ' (metres)', (panel * 400 + 25, 35), 0, .65, (30, 30, 30), 1)
            cv2.putText(canvas, f'window {extent:.3f}m; L blue / R orange', (panel * 400 + 10, 425), 0, .45, (30, 30, 30), 1)
            for side, color in sides:
                points = arrays[f'{side}_center'][indices][:, [a, b]]
                previous = None
                for point in points:
                    if not np.isfinite(point).all():
                        previous = None
                        continue
                    p = (point - center) / extent * 310
                    xy = (int(panel * 400 + 200 + p[0]), int(225 - p[1]))
                    if previous is not None:
                        cv2.line(canvas, previous, xy, color, 2)
                    cv2.circle(canvas, xy, 3, color, -1)
                    previous = xy
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError('Could not save trajectory image')


def inspect_human_reference(ctx, request):
    """Query frames/times or start/end interval; overlays<=8, raw joints optional.

    Returned hand positions/directions are in source world. The explicit rigid
    alignment maps these to sim. Unknown confidences become JSON null.
    """
    meta, arrays = load_reference(ctx.input_dir)
    indices = _indices(meta, arrays, request)
    joints = request.get('joints', [])
    if not isinstance(joints, list) or len(joints) > 24:
        raise ValueError('Request at most 24 raw joint names')
    result = {'episode_id': meta['episode_id'], 'input_fingerprint': meta['input_fingerprint'],
              'exported_frames':indices, 'source_frames': arrays['source_frames'][indices], 'timestamps': arrays['timestamps'][indices],
              'source_timestamps':arrays.get('source_timestamps',arrays['timestamps'])[indices],
              'frame_query_convention':'frames requests use exported_frames, not original source_frames',
              'position_frame': 'source_world', 'T_sim_source_world': meta['T_sim_source_world'],
              'camera': {'K': arrays['K'][indices] if arrays['K'].ndim == 3 else arrays['K'], 'T_world_camera': arrays['T_world_camera'][indices]},
              'task_text': meta['task_text'], 'caveat': meta.get('source_conventions', {}).get('caveat', CAVEAT if meta['contract_version'] == '1.0.0' else 'Geometric human references are not contact or object ground truth.'), 'hands': {}}
    for side in ('left', 'right'):
        result['hands'][side] = {key: arrays[f'{side}_{key}'][indices] for key in ['center', 'wrist', 'direction', 'opening', 'confidence', 'confidence_known', 'valid', 'direction_valid']}
    if joints:
        with np.load(Path(ctx.input_dir) / 'raw_joints.npz', allow_pickle=False) as raw:
            result['joints'] = {}
            for name in joints:
                if f'transform__{name}' in raw:
                    fields=['transform','confidence','valid']
                elif f'position__{name}' in raw:
                    fields=['position','confidence','valid']
                else:
                    raise ValueError(f'Unknown source joint: {name}')
                result['joints'][name] = {field: raw[f'{field}__{name}'][indices] for field in fields}
    output = Path(ctx.root) / 'evidence' / ('reference_' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    files = []
    mappings = {}
    trajectory = output / 'trajectory.png'
    # Full interval, not only subsampled tool-return points; preserve invalid gaps.
    full_indices = np.arange(int(indices.min()), int(indices.max()) + 1)
    _trajectory_image(arrays, full_indices, trajectory)
    files.append(str(trajectory.relative_to(ctx.root)))
    overlay_count = request.get('overlay_count', 4)
    if isinstance(overlay_count, bool) or not isinstance(overlay_count, int) or not 0 <= overlay_count <= 8:
        raise ValueError('overlay_count must be 0..8')
    fingertips = {'left': [], 'right': []}
    if meta.get('reference_extraction'):
        with np.load(Path(ctx.input_dir) / 'raw_joints.npz', allow_pickle=False) as raw:
            for side in ('left', 'right'):
                names = [side + meta['reference_extraction']['thumb_tip']] + [side + name for name in meta['reference_extraction']['opposing_tips']]
                fingertips[side] = [raw[f'transform__{name}'][:, :3, 3] for name in names if f'transform__{name}' in raw]
    cap = cv2.VideoCapture(str(Path(ctx.input_dir) / 'source/rgb.mp4'))
    try:
        selected = indices[np.unique(np.linspace(0, len(indices) - 1, min(overlay_count, len(indices))).round().astype(int))] if overlay_count else []
        for i in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Could not read source frame {i}')
            for side, color in [('left', (220, 120, 30)), ('right', (40, 100, 230))]:
                points = np.stack([arrays[f'{side}_wrist'][i], arrays[f'{side}_center'][i]])
                pixels, depths = project_points(points, arrays['T_world_camera'][i], arrays['K'][i] if arrays['K'].ndim == 3 else arrays['K'])
                if not arrays[f'{side}_valid'][i] or not np.isfinite(pixels).all() or np.any(depths <= 0):
                    continue
                if np.all((pixels[:, 0] >= 0) & (pixels[:, 0] < meta['width']) & (pixels[:, 1] >= 0) & (pixels[:, 1] < meta['height'])):
                    p, q = [tuple(x.round().astype(int)) for x in pixels]
                    cv2.line(frame, p, q, color, 5)
                    cv2.circle(frame, q, 12, color, -1)
                    cv2.putText(frame, side + ' center', q, 0, 1., color, 2)
                tip_pixels, tip_depths = project_points([tip[i] for tip in fingertips[side]], arrays['T_world_camera'][i], arrays['K'][i] if arrays['K'].ndim == 3 else arrays['K'])
                for xy, depth in zip(tip_pixels, tip_depths):
                    if depth > 0 and np.isfinite(xy).all() and 0 <= xy[0] < meta['width'] and 0 <= xy[1] < meta['height']:
                        cv2.circle(frame, tuple(xy.round().astype(int)), 9, color, 3)
            frame = cv2.resize(frame, (960, round(meta['height'] * 960 / meta['width'])))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (20, 20, 20), -1)
            cv2.putText(frame, f'export f{i} / source f{arrays["source_frames"][i]} | {arrays["timestamps"][i]:.3f}s | refs, NOT contacts', (10, 22), 0, .55, (255, 255, 255), 1)
            cv2.putText(frame, meta.get('source_conventions', {}).get('caveat', 'Geometric references; see source quality metadata')[:105], (10, 44), 0, .55, (255, 255, 255), 1)
            path = output / f'frame_{i:06d}.jpg'
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError('Could not save reference overlay')
            relative = str(path.relative_to(ctx.root))
            files.append(relative)
            mappings[relative] = {'frame': int(i), 'width': frame.shape[1], 'height': frame.shape[0],
                'T_original_image': [[meta['width']/frame.shape[1], 0, 0], [0, meta['height']/frame.shape[0], 0], [0, 0, 1]],
                'occluded_header_height': 54}

    finally:
        cap.release()
    result['media'] = files
    result['image_mappings'] = mappings
    result = _clean(result)
    record = ctx.put('human_reference_evidence', result, files=files)
    return {'record_id': record['id'], **result}
