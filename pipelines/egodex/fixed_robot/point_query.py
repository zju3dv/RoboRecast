"""Auditable source RGB point queries; unavailable is an explicit result."""
from pathlib import Path
import uuid
import cv2
import numpy as np
from .inputs import load_reference
from .geometry import project_points


def query_point(ctx, request):
    meta, arrays = load_reference(ctx.input_dir)
    result = {'operation': 'query_point', 'valid': False, 'status': 'unavailable',
              'input_fingerprint': meta['input_fingerprint'], 'request': request,
              'interpretation': 'A visible surface point, not object center, pose or contact ground truth.'}
    files = []
    def finish(reason=None):
        if reason:
            result['reason'] = reason
        record = ctx.put('geometry_evidence', result, files=files)
        return {'record_id': record['id'], **result}
    frame = request.get('frame')
    if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < meta['frame_count']:
        raise ValueError('frame must be an exported RGB frame index')
    pixel = np.asarray(request['pixel'], float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError('pixel must be finite [u,v]')
    if not request.get('semantic_label') or request.get('target_kind') not in ('static_environment', 'moving_object'):
        raise ValueError('Provide semantic_label and target_kind static_environment|moving_object')
    space = request.get('image_space', 'original')
    if space != 'original':
        mapping = meta.get('image_mappings', {}).get(request.get('evidence_id'))
        if request.get('evidence_id', '').startswith('human_reference_evidence-'):
            from .utils import read_json, sha256
            identifier = request['evidence_id']
            if '/' in identifier or '..' in identifier:
                return finish('invalid_evidence_id')
            record = read_json(ctx.path('.harness/records/' + identifier + '.json'))
            if record['input_signature'] != ctx._input_signature():
                return finish('stale_image_evidence')
            image_path = request.get('image_path')
            mapping = record['payload'].get('image_mappings', {}).get(image_path)
            if image_path not in record['files'] or sha256(ctx.path(image_path)) != record['files'][image_path]:
                return finish('image_evidence_changed')
        if space != 'registered' or not mapping or mapping.get('frame') != frame:
            return finish('unregistered_image_mapping')
        w, h = mapping['width'], mapping['height']
        if pixel[1] < mapping.get('occluded_header_height', 0):
            return finish('pixel_in_evidence_header')
        if not 0 <= pixel[0] < w or not 0 <= pixel[1] < h:
            return finish('pixel_outside_registered_image')
        transform = np.asarray(mapping['T_original_image'], float)
        q = transform @ np.r_[pixel, 1.]
        if q.shape != (3,) or not np.isfinite(q).all() or abs(q[2]) < 1e-12:
            return finish('invalid_image_mapping')
        pixel = q[:2] / q[2]
    if not 0 <= pixel[0] < meta['width'] or not 0 <= pixel[1] < meta['height']:
        return finish('pixel_outside_rgb')
    camera = meta.get('camera', {'model': 'pinhole', 'axes': 'x_right_y_down_z_forward', 'distortion_status': 'none'})
    if request.get('camera', camera.get('name', 'rgb')) != camera.get('name', 'rgb'):
        return finish('camera_not_registered')
    if camera['model'] != 'pinhole' or camera.get('distortion_status') not in ('none', 'undistorted'):
        return finish('unsupported_camera_model')
    K = arrays['K'][frame] if arrays['K'].ndim == 3 else arrays['K']
    pose = arrays['T_world_camera'][frame]
    result.update(frame=frame, source_frame=int(arrays['source_frames'][frame]), timestamp=float(arrays['timestamps'][frame]),
                  pixel_original=pixel.tolist(), camera=camera, K=K.tolist(), T_world_camera=pose.tolist(),
                  T_sim_source_world=meta['T_sim_source_world'], unit='m')
    mask_decl=camera.get('valid_mask')
    if mask_decl:
        if mask_decl.get('file') not in meta['files']:
            return finish('rgb_valid_mask_not_in_verified_input_manifest')
        with np.load(Path(ctx.input_dir)/mask_decl['file'],allow_pickle=False) as z:
            mask=z[mask_decl.get('key','mask')]
        if mask.shape!=(meta['height'],meta['width']):
            return finish('rgb_valid_mask_shape_mismatch')
        u,v=np.floor(pixel).astype(int)
        if not bool(mask[v,u]):return finish('pixel_outside_calibrated_rgb_valid_region')
    geometry = meta.get('geometry', {})
    declaration = geometry.get('depth') or geometry.get('pointcloud')
    if not declaration:
        return finish('no_depth_or_pointcloud; use explicit conditional ray_plane/static triangulation')
    if not declaration.get('provenance') or declaration.get('length_unit') != 'm':
        return finish('missing_geometry_units_or_provenance')
    path = Path(ctx.input_dir) / declaration['file']
    if declaration['file'] not in meta['files']:
        return finish('geometry_not_in_verified_input_manifest')
    with np.load(path, allow_pickle=False) as z:
        if geometry.get('depth'):
            result['method'] = 'aligned_depth'
            if declaration.get('aligned_to') != camera.get('name', 'rgb') or declaration.get('convention') not in ('axial_z', 'ray_distance') or declaration.get('source_type') not in ('measured', 'rendered', 'estimated'):
                return finish('undeclared_depth_alignment_convention_or_source')
            depths = z[declaration.get('key', 'depth')]
            if depths.shape != (meta['frame_count'], meta['height'], meta['width']):
                return finish('depth_rgb_shape_mismatch')
            u, v = np.rint(pixel).astype(int)
            u, v = min(u, meta['width'] - 1), min(v, meta['height'] - 1)
            depth = depths[frame]
            patch = depth[max(v-1, 0):v+2, max(u-1, 0):u+2]
            finite = patch[np.isfinite(patch) & (patch > 0)]
            if not np.isfinite(depth[v, u]) or depth[v, u] <= 0 or len(finite) < 3:
                return finish('invalid_depth_at_selected_pixel')
            spread = float(np.ptp(finite))
            quality = {'neighborhood_samples': len(finite), 'neighborhood_range_m': spread,
                       'max_range_m': max(.02, .02 * float(np.median(finite)))}
            result['quality'] = quality
            if spread > quality['max_range_m']:
                return finish('depth_discontinuity_edge_risk')
            used = np.array([u, v], float)
            ray = np.linalg.solve(K, np.r_[used, 1.])
            point_camera = ray * float(depth[v, u]) / (np.linalg.norm(ray) if declaration['convention'] == 'ray_distance' else ray[2])
            point = pose[:3, :3] @ point_camera + pose[:3, 3]
        else:
            result['method'] = 'visible_pointcloud_projection'
            if declaration.get('scope') != 'static_environment' or request['target_kind'] != 'static_environment':
                return finish('static_cloud_cannot_measure_moving_object')
            if declaration.get('frame') != 'source_world':
                return finish('pointcloud_frame_not_source_world')
            points = z[declaration.get('key', 'points')]
            pixels, depths = project_points(points, pose, K)
            distance = np.linalg.norm(pixels - pixel, axis=1)
            hits = np.flatnonzero(np.isfinite(distance) & (depths > 0) & (distance <= 1.5))
            if len(hits) < 3:
                return finish('insufficient_projected_cloud_support')
            front = hits[np.argmin(depths[hits])]
            spread = float(np.ptp(depths[hits]))
            result['quality'] = {'neighborhood_samples': len(hits), 'depth_range_m': spread,
                                 'visibility': 'nearest projected cloud layer only; dynamic occlusion requires RGB assertion'}
            if not request.get('visibility_note'):
                return finish('static_cloud_requires_RGB_visibility_note')
            if spread > max(.02, .02 * float(depths[front])):
                return finish('multiple_projected_depth_layers')
            used, point = pixels[front], points[front]
    result.update(valid=True, status='available', pixel_used=used.tolist(),
                  pixel_offset=float(np.linalg.norm(used-pixel)), point_source_world=point.tolist(),
                  point_sim_world=(np.asarray(meta['T_sim_source_world']) @ np.r_[point, 1.])[:3].tolist(),
                  provenance=declaration)
    cap = cv2.VideoCapture(str(Path(ctx.input_dir) / 'source/rgb.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if ok:
        cv2.drawMarker(image, tuple(np.rint(pixel).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.circle(image, tuple(np.rint(used).astype(int)), 5, (0, 0, 255), 2)
        output = Path(ctx.root) / 'evidence' / ('query_' + uuid.uuid4().hex[:12] + '.jpg')
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)
        files.append(str(output.relative_to(ctx.root)))
        result['visualization'] = files[-1]
    return finish()
