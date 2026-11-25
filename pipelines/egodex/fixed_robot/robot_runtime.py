"""Model-embedded robot mappings, shared by planning and execution."""
from __future__ import annotations
import json
from functools import lru_cache
import mujoco
import numpy as np

CONTRACT_TEXT = 'human_robot_runtime'

def fixed_vx300s_contract(motion_mode='not_mobile', xyyaw=(0, 0, 0)):
    if motion_mode != 'not_mobile':
        raise ValueError('EgoDex transfer requires a fixed robot base')
    names = ('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate')
    arms = {}
    for side in ('left', 'right'):
        arms[side] = dict(joints=[f'{side}/{n}' for n in names], actuators=[f'{side}/{n}' for n in names],
            eef_site=f'{side}/pinch', eef_body=f'{side}/gripper_link',
            default_configuration=[0, -.7, .9, 0, -.2, 0, 1],
            gripper=dict(actuator=f'{side}/gripper', joints=[f'{side}/left_finger', f'{side}/right_finger'],
                joint_signs=[1, -1], finger_bodies=[f'{side}/left_finger_link', f'{side}/right_finger_link'],
                finger_geoms=[f'{side}/{finger}_{i}' for finger in ('left_finger', 'right_finger') for i in range(5)],
                control_range=[.021, .057]))
    return dict(schema_version='2.0', bundle_id='fixed_dual_vx300s', bundle_version='2.0.0', motion_mode=motion_mode,
        supported_modes=['not_mobile'], controller='joint_position', arms=arms,
        base=dict(body='tracer/base', joints=[],
            initial_pose=list(map(float, xyyaw)), drive_type='fixed',
            limits=dict(v_mps=.3, omega_radps=.6, accel_mps2=.3, angular_accel_radps2=.8),
            footprint_half_extents_m=[.36, .317], height_m=1.01,
            physics='fixed base pose'),
        cameras={'left_wrist':'wrist_cam_left', 'right_wrist':'wrist_cam_right'},
        robot_geom_prefixes=['left/', 'right/', 'tracer/'],
        collision_groups={'base_prefix':'tracer/', 'arms':{'left':'left/', 'right':'right/'}},
        self_collision_policy='MuJoCo asset contact exclusions; remaining robot contacts are checked',
        limitations=['The paired VX300S arms use joint position control.'])

@lru_cache(maxsize=16)
def runtime_contract(model):
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, CONTRACT_TEXT)
    if idx >= 0:
        start, size = int(model.text_adr[idx]), int(model.text_size[idx])
        raw = bytes(model.text_data[start:start + size]).split(b'\0', 1)[0]
        return json.loads(raw.decode())
    # Explicit v1 legacy adapter only; never guess mappings for unfamiliar robots.
    contract = fixed_vx300s_contract()
    for arm in contract['arms'].values():
        for name in arm['joints']:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
                raise ValueError('Robot has no runtime contract and is not a legacy Human Mobile ALOHA model')
    if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0 for n in ('tracer/x','tracer/y','tracer/yaw')):
        raise ValueError('Legacy fallback only supports structurally fixed Human ALOHA')
    contract['legacy_fallback'] = 'human_v1_fixed_aloha'
    return contract

def base_pose(model, data):
    contract = runtime_contract(model)
    joints = contract['base']['joints']
    if joints:
        return np.array([data.qpos[model.jnt_qposadr[model.joint(n).id]] for n in joints])
    body = model.body(contract['base']['body']).id
    quat = model.body_quat[body]
    return np.array([*model.body_pos[body, :2], 2*np.arctan2(quat[3], quat[0])])

def set_base_pose(model, data, pose, twist=None):
    pose = np.asarray(pose, float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError('base pose requires finite world x,y,yaw')
    joints = runtime_contract(model)['base']['joints']
    velocity = np.zeros(3)
    if twist is not None:
        twist = np.asarray(twist, float)
        if twist.shape == (2,):
            velocity = np.array([twist[0]*np.cos(pose[2]), twist[0]*np.sin(pose[2]), twist[1]])
        elif twist.shape == (3,):
            velocity = twist
        else:
            raise ValueError('twist requires [v,omega] or world [xdot,ydot,yawdot]')
        if not np.isfinite(velocity).all():
            raise ValueError('nonfinite base twist')
    if not joints:
        if not np.allclose(base_pose(model, data), pose, atol=1e-8) or np.any(velocity):
            raise ValueError('not_mobile robot cannot change base pose or velocity')
        return
    for index, name in enumerate(joints):
        joint = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint]] = pose[index]
        data.qvel[model.jnt_dofadr[joint]] = velocity[index]
