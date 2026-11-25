"""Numerical motion evidence, independent of agent verdicts."""
import numpy as np


def velocity_report(times, velocities, joint_names, limits):
    values = np.asarray(velocities, float)
    if limits is None:
        return {'status': 'UNKNOWN', 'reason': 'Provide named robot velocity limits from the robot description'}
    bound = np.asarray([limits[name] for name in joint_names], float)
    if not np.isfinite(bound).all() or np.any(bound <= 0):
        raise ValueError('Velocity limits must be finite positive values in rad/s')
    exceeded = ~np.isfinite(values) | (np.abs(values) > bound[None, :] + 1e-6)
    hits = np.argwhere(exceeded)
    result = {'status': 'FAIL' if len(hits) else 'PASS', 'limit_rad_s': bound.tolist(),
              'peak_rad_s': np.max(np.abs(values), axis=0).tolist()}
    if len(hits):
        i, j = map(int, hits[0])
        result.update(first_time_sec=float(times[i]), joint=joint_names[j], measured_rad_s=float(values[i, j]))
    return result


def build_contact_policy(model, base_model, requirements, intervals, allowances=()):
    import mujoco
    from .rollout import _resolve_contact_groups
    robot_names = {mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_BODY, i)
                   for i in range(1, base_model.nbody)} - {None}
    robot_bodies = {i for i in range(1, model.nbody)
                    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) in robot_names}
    robot_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in robot_bodies}
    allowed = {}
    def add(a, b, windows):
        groups = _resolve_contact_groups(model, a, b)
        if groups is None:
            raise ValueError('Unknown contact policy entities: ' + a + ', ' + b)
        for ga in groups[0]:
            for gb in groups[1]:
                allowed.setdefault(tuple(sorted((int(ga), int(gb)))), []).extend(windows)
    for event in requirements['I']:
        for pred in event['predicates']:
            parts = pred['key'].split(':')
            if pred['type'] == 'contact' and len(parts) == 3:
                add(parts[1], parts[2], intervals[event['id']])
    for row in allowances:
        add(row['a'], row['b'], row['intervals'])
    # Adjacent/welded robot links are mechanical assembly pairs, not task contacts.
    structural = set()
    for ga in robot_geoms:
        for gb in robot_geoms:
            a, b = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
            if (model.body_weldid[a] == model.body_weldid[b] or model.body_parentid[a] == b or model.body_parentid[b] == a):
                structural.add(tuple(sorted((ga, gb))))
    # The complete Robotiq pads meet at zero aperture. Their opposing contact is
    # the mechanical closing stop, including unloaded closure before a hook.
    # Keep the contact in MuJoCo and in penetration metrics; classify only this
    # known mating pair as structural, not arbitrary gripper or object contacts.
    pads = {}
    for side in ('left', 'right'):
        pads[side] = {g for g in robot_geoms
                      if model.geom(g).name in (side+'_finger_pad', side+'_finger_pad_lower')
                      and model.body(int(model.geom_bodyid[g])).name == 'rq_'+side+'_pad'}
    structural.update(tuple(sorted((a, b))) for a in pads['left'] for b in pads['right'])
    return robot_geoms, structural, allowed


def unexpected_contact(data, policy, time_sec):
    robot_geoms, structural, allowed = policy
    for contact in data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if getattr(contact, 'efc_address', 0 if contact.dist <= 0 else -1) < 0 or not ({a, b} & robot_geoms):
            continue
        pair = tuple(sorted((a, b)))
        if pair in structural or any(start <= time_sec <= end for start, end in allowed.get(pair, ())):
            continue
        return a, b
    return -1, -1


def default_motion_policy(runtime):
    names = runtime['arm_joint_names']
    limits = runtime.get('joint_velocity_limits')
    source = 'robot_runtime_contract.joint_velocity_limits'
    if limits is None and runtime.get('robot_family') == 'Panda' and names == ['panda_joint'+str(i) for i in range(1,8)]:
        limits = [2.175]*4 + [2.610]*3
        source = 'https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml (verified 2026-11-19)'
    if limits is None and runtime.get('robot_family') == 'FR3' and names in (
            ['panda_joint'+str(i) for i in range(1,8)],
            ['fr3_joint'+str(i) for i in range(1,8)]):
        # Prepared legacy FR3 assets retain panda_joint* identifiers.
        limits = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
        source = 'https://raw.githubusercontent.com/frankarobotics/franka_description/main/robots/fr3/joint_limits.yaml (verified 2026-11-20)'
    if limits is None:
        return {}
    return {'velocity_limits_rad_s': dict(zip(names, limits)), 'velocity_limits_source': source, 'allowed_contacts': []}


def dwell_report(source_path, dense, position_tolerance_m=.01, rotation_tolerance_rad=.1):
    """Check actual end-effector holding during explicitly authored dwell segments."""
    from scipy.spatial.transform import Rotation
    with np.load(source_path, allow_pickle=False) as source:
        if 'motion_mode' not in source:
            return {'status': 'UNKNOWN', 'reason': 'Semantic motion modes missing from compiled targets'}
        times, modes = source['timestamp_sec'], source['motion_mode']
        positions, quaternions = source['position_m'], source['quaternion_xyzw']
        rows=[]
        for i in range(1,len(times)):
            if str(modes[i]) != 'dwell' or times[i-1] >= dense['timestamp_sec'][-1]: continue
            mask=(dense['timestamp_sec'] >= times[i-1]) & (dense['timestamp_sec'] <= times[i])
            if not np.any(mask):
                rows.append({'segment': i, 'status': 'UNKNOWN'});continue
            measured_positions=dense['ee_position_m'][mask]
            delta=np.linalg.norm(measured_positions-positions[i],axis=1)
            actual=Rotation.from_quat(dense['ee_quaternion_wxyz'][mask][:,[1,2,3,0]])
            angle=(Rotation.from_quat(quaternions[i]).inv()*actual).magnitude()
            drift=np.linalg.norm(measured_positions-measured_positions[0],axis=1)
            rotation_drift=(actual[0].inv()*actual).magnitude()
            rows.append({'segment':i,'start_sec':float(times[i-1]),'end_sec':float(times[i]),
                'max_position_error_m':float(delta.max()),'max_rotation_error_rad':float(angle.max()),
                'max_position_drift_m':float(drift.max()),'max_rotation_drift_rad':float(rotation_drift.max()),
                'status':'PASS' if drift.max() <= position_tolerance_m and rotation_drift.max() <= rotation_tolerance_rad else 'FAIL'})
    return {'status':'PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL',
            'criterion':'Actual pose drift during a fixed-target dwell; tracking bias is diagnostic, task contacts and outcomes are checked separately',
            'position_tolerance_m':position_tolerance_m,'rotation_tolerance_rad':rotation_tolerance_rad,'segments':rows}
