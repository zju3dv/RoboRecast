"""Advisory contact/support placement. Never changes registered geometry."""
import numpy as np
from mobile_input.loader import load_episode
from .operation_reference import source_poses
from .semantic_trajectory import _normalize_quaternion, _quat_to_matrix


def placement_candidate(hand_position, local_contact, quaternion, support_z=None, local_bottom=None):
    hand=np.asarray(hand_position,dtype=float);contact=np.asarray(local_contact,dtype=float)
    if hand.shape != (3,) or contact.shape != (3,) or not np.isfinite([hand,contact]).all():
        raise ValueError('hand position and contact point must be finite xyz')
    quat=_normalize_quaternion(quaternion);rotation=_quat_to_matrix(quat)
    position=hand-rotation@contact
    if support_z is not None:
        if local_bottom is None:raise ValueError('support_height_m requires local_support_point_m')
        bottom=np.asarray(local_bottom,dtype=float)
        if bottom.shape != (3,) or not np.isfinite(bottom).all() or not np.isfinite(support_z):
            raise ValueError('support point/height must be finite')
        position[2]=float(support_z)-(rotation@bottom)[2]
    residual=position+rotation@contact-hand
    return {'position_m':position.tolist(),'quaternion_wxyz':quat.tolist(),
            'contact_residual_xyz_m':residual.tolist(),'contact_residual_m':float(np.linalg.norm(residual))}


def suggest_placement(input_dir,store,request):
    episode=load_episode(input_dir);side=request['side']
    if side not in ('left','right'):raise ValueError('side must be left or right')
    pose=source_poses(episode,[request['row']])[0]
    candidates=[placement_candidate(pose['hands'][side]['position_m'],request['local_contact_point_m'],q,
                request.get('support_height_m'),request.get('local_support_point_m'))
                for q in request['object_quaternions_wxyz']]
    return {'status':'ADVISORY','coordinate_frame':'source route world; align with reconstruction before use',
            'reference':pose,'candidates':candidates,
            'limitations':'Orientation and local contact/support points are Agent estimates. No unique visual pose recovery, collision or reach certification; register/settle and probe to verify. Nonzero residual means contact and support assumptions disagree.'}
