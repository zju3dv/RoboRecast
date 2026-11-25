"""Conditional, auditable geometry aids. These do not provide object ground truth."""
import itertools
import numpy as np
from .inputs import load_reference, CAVEAT


def project_points(points, T_world_camera, K):
    """Project source-world points through the published EgoDex +Z camera."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    pose = np.asarray(T_world_camera, dtype=float)
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    homogeneous = camera @ np.asarray(K).T
    pixels = homogeneous[:, :2] / np.where(np.abs(homogeneous[:, 2:3]) > 1e-12, homogeneous[:, 2:3], np.nan)
    return pixels, camera[:, 2]


def camera_ray(pixel, pose, K):
    direction = pose[:3, :3] @ np.linalg.solve(K, np.r_[pixel, 1.])
    return pose[:3, 3].copy(), direction / np.linalg.norm(direction)


def triangulate(pixels, poses, K, min_baseline_m=0.01, min_parallax_deg=1.0, max_reprojection_px=8.0):
    """DLT static-point estimate, with degeneracy and reprojection diagnostics."""
    pixels, poses, K = np.asarray(pixels, float), np.asarray(poses, float), np.asarray(K, float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 2 or poses.shape != (len(pixels), 4, 4):
        raise ValueError('At least two matching pixel/pose observations are required')
    if not np.isfinite(pixels).all() or not np.isfinite(poses).all():
        raise ValueError('Nonfinite observation')
    intrinsics = np.broadcast_to(K, (len(pixels), 3, 3))
    rays = [camera_ray(p, pose, k) for p, pose, k in zip(pixels, poses, intrinsics)]
    pairs = list(itertools.combinations(range(len(pixels)), 2))
    baselines = [float(np.linalg.norm(rays[i][0] - rays[j][0])) for i, j in pairs]
    angles = [float(np.degrees(np.arccos(np.clip(rays[i][1] @ rays[j][1], -1, 1)))) for i, j in pairs]
    diagnostics = {'baseline_max_m': max(baselines), 'parallax_max_deg': max(angles),
                   'thresholds': {'min_baseline_m': min_baseline_m, 'min_parallax_deg': min_parallax_deg, 'max_reprojection_px': max_reprojection_px}}
    if max(baselines) < min_baseline_m or max(angles) < min_parallax_deg:
        return {'valid': False, 'reason': 'insufficient_translation_or_parallax', 'diagnostics': diagnostics}
    def solve(indices):
        rows = []
        for i in indices:
            P = intrinsics[i] @ np.linalg.inv(poses[i])[:3]
            u, v = pixels[i]
            rows.extend([u * P[2] - P[0], v * P[2] - P[1]])
        _, singular, vh = np.linalg.svd(rows)
        q = vh[-1]
        return (q[:3] / q[3] if abs(q[3]) > 1e-12 else None), singular
    point, singular = solve(range(len(pixels)))
    if point is None or not np.isfinite(point).all():
        return {'valid': False, 'reason': 'point_at_infinity', 'diagnostics': diagnostics}
    errors, depths = [], []
    for pixel, pose, k in zip(pixels, poses, intrinsics):
        projected, depth = project_points(point, pose, k)
        errors.append(float(np.linalg.norm(projected[0] - pixel)))
        depths.append(float(depth[0]))
    pair_points = []
    for (i, j), baseline, angle in zip(pairs, baselines, angles):
        if baseline >= min_baseline_m and angle >= min_parallax_deg:
            pair, _ = solve([i, j])
            if pair is not None and np.isfinite(pair).all():
                pair_points.append(pair)
    spread = max((float(np.linalg.norm(p - point)) for p in pair_points), default=None)
    diagnostics.update({'reprojection_px': errors, 'depths_m': depths, 'pairwise_point_spread_m': spread,
                        'pairwise_estimate_count': len(pair_points), 'singular_values': singular.tolist(),
                        'uncertainty': 'Pair spread is a consistency diagnostic, not a calibrated metric error bound.'})
    reason = 'behind_camera' if min(depths) <= 0 else ('reprojection_error' if max(errors) > max_reprojection_px else None)
    return {'valid': reason is None, 'reason': reason, 'point_source_world': point.tolist(), 'diagnostics': diagnostics}


def intersect_ray_plane(pixel, pose, K, normal, offset):
    """Plane equation dot(normal, X) + offset = 0, in source-world metres."""
    normal = np.asarray(normal, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9 or not np.isfinite(offset):
        raise ValueError('A finite nonzero plane normal and offset are required')
    origin, direction = camera_ray(pixel, np.asarray(pose), np.asarray(K))
    denominator = float(normal @ direction)
    if abs(denominator) < 1e-8 * np.linalg.norm(normal):
        return {'valid': False, 'reason': 'ray_parallel_to_plane'}
    distance = -float(normal @ origin + offset) / denominator
    if distance <= 0:
        return {'valid': False, 'reason': 'intersection_behind_camera'}
    return {'valid': True, 'point_source_world': (origin + distance * direction).tolist(), 'ray_distance_m': distance,
            'incidence_cosine': abs(denominator) / float(np.linalg.norm(normal)),
            'assumption': {'normal_source_world': normal.tolist(), 'offset': float(offset)},
            'interpretation': 'Conditional on the supplied plane; not independent evidence of plane position.'}


def inspect_scene_evidence(ctx, request):
    """Request: operation=triangulate|ray_plane; observations=[{frame,pixel:[u,v]}].

    Triangulate also requires static_point=true, correspondence_note. Ray-plane
    requires plane={normal:[x,y,z],offset:number,evidence:string}, source world.
    """
    if request.get('operation') == 'query_point':
        from .point_query import query_point
        return query_point(ctx, request)
    meta, arrays = load_reference(ctx.input_dir)
    observations = request.get('observations', [])
    if not 1 <= len(observations) <= 32:
        raise ValueError('Provide 1..32 observations in original source pixels')
    indices = []
    pixels = []
    for item in observations:
        frame = item['frame']
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < meta['frame_count']:
            raise ValueError('Observation frame is outside the original episode')
        pixel = np.asarray(item['pixel'], float)
        if pixel.shape != (2,) or not np.isfinite(pixel).all() or not (0 <= pixel[0] < meta['width'] and 0 <= pixel[1] < meta['height']):
            raise ValueError('Observation pixel must be inside the original RGB image')
        indices.append(frame)
        pixels.append(pixel)
    poses = arrays['T_world_camera'][indices]
    operation = request.get('operation', 'triangulate')
    if operation == 'triangulate':
        if request.get('static_point') is not True or not request.get('correspondence_note'):
            raise ValueError('Agent must assert static_point=true and document correspondence evidence')
        if len(set(indices)) != len(indices):
            raise ValueError('Triangulation observations need distinct source frames')
        result = triangulate(pixels, poses, arrays['K'][indices] if arrays['K'].ndim == 3 else arrays['K'])
    elif operation == 'ray_plane':
        plane = request['plane']
        if len(observations) != 1 or not plane.get('evidence'):
            raise ValueError('Ray-plane requires one observation and explicit plane evidence/assumption')
        result = intersect_ray_plane(pixels[0], poses[0], arrays['K'][indices[0]] if arrays['K'].ndim == 3 else arrays['K'], plane['normal'], plane['offset'])
    else:
        raise ValueError('operation must be triangulate or ray_plane')
    if 'point_source_world' in result:
        point = np.r_[result['point_source_world'], 1.]
        result['point_sim_world'] = (np.asarray(meta['T_sim_source_world']) @ point)[:3].tolist()
    result.update({'operation': operation, 'source_frames': indices, 'caveat': CAVEAT,
                   'input_fingerprint': meta['input_fingerprint'], 'evidence_request': request})
    record = ctx.put('geometry_evidence', result)
    return {'record_id': record['id'], **result}
