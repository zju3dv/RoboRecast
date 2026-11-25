"""Read-only, bounded inspection of recorded execution and runtime conventions."""
from pathlib import Path
import re
import numpy as np
import mujoco
from .utils import read_json, sha256
from .robot_runtime import runtime_contract, fixed_vx300s_contract, base_pose
from .rollout import _descendant, _check, _contacts
from .checkpoints import STATE


def _path(ctx, relative):
    path = (ctx.root / relative).resolve()
    if not path.is_relative_to(ctx.root.resolve()):
        raise ValueError('record path leaves workspace')
    return path


def _record(ctx, identifier):
    if not re.fullmatch(r'[a-z_]+-[0-9a-f]{16}', identifier):
        raise ValueError('expected an existing harness record ID')
    record = read_json(_path(ctx, '.harness/records/' + identifier + '.json'))
    if record['id'] != identifier or record['input_signature'] != ctx._input_signature():
        raise ValueError('record identity/input mismatch')
    for path, signature in record.get('files', {}).items():
        if sha256(_path(ctx, path)) != signature:
            raise ValueError('recorded artifact changed: ' + path)
    return record


def _quat(matrix):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(matrix).reshape(9))
    return q.tolist()


def inspect_robot_runtime(ctx, request=None):
    """Return the installed robot's explicit conventions; no model generation."""
    from .inputs import load_reference
    meta, _ = load_reference(ctx.input_dir)
    contract = fixed_vx300s_contract(meta['motion_mode'])
    result = {'contract': contract, 'motion_mode': meta['motion_mode'],
              'position_frame': 'simulation world', 'units': 'metres, radians, seconds',
              'quaternion_order': 'wxyz', 'installed': False}
    try:
        placement = ctx.latest('placement')
    except ValueError:
        result['note'] = 'Fixed dual VX300S defaults. Commit a placement to inspect its installed model.'
        return result
    model = mujoco.MjModel.from_xml_path(str(_path(ctx, placement['payload']['scene_xml'])))
    contract = runtime_contract(model)
    result.update(contract=contract, installed=True, motion_mode=contract['motion_mode'], placement_id=placement['id'])
    result['actuator_order'] = [model.actuator(i).name for i in range(model.nu)]
    result['actuator_ctrlrange'] = model.actuator_ctrlrange.tolist()
    result['task_body_names'] = [model.body(i).name for i in range(1,model.nbody)
                                if not model.body(i).name.startswith(tuple(contract['robot_geom_prefixes']))]
    result['joint_qpos_addresses'] = {model.joint(i).name:int(model.jnt_qposadr[i]) for i in range(model.njnt)}
    result['eef_local'] = {side: {'site': arm['eef_site'], 'body': arm['eef_body'],
        'position_m': model.site_pos[model.site(arm['eef_site']).id].tolist(),
        'quaternion_wxyz': model.site_quat[model.site(arm['eef_site']).id].tolist()}
        for side,arm in contract['arms'].items()}
    result['finger_joint_axes_local'] = {side: {name:model.jnt_axis[model.joint(name).id].tolist()
        for name in arm['gripper']['joints']} for side,arm in contract['arms'].items()}
    return result


def inspect_execution_state(ctx, request):
    """Inspect <=12 sim/source times and <=8 named bodies from a recorded run.

    request: record_id OR kind='rollout'|'stage_probe', times, time_domain='sim'|
    'source', bodies, side='left'|'right'|'both', optional start/end (sim seconds).
    Historical IDs use their own placement/trajectory, never today's latest model.
    """
    kind = request.get('kind', 'rollout')
    if kind not in ('rollout','stage_probe'): raise ValueError('unsupported record kind')
    record = _record(ctx, request['record_id']) if request.get('record_id') else ctx.latest(kind)
    if record['kind'] not in ('rollout','stage_probe'): raise ValueError('not an execution record')
    placement = _record(ctx, record['dependencies']['placement'])
    trajectory = _record(ctx, record['dependencies']['trajectory'])
    payload = record['payload']; diag = read_json(_path(ctx,payload['diagnostics_json']))
    model = mujoco.MjModel.from_xml_path(str(_path(ctx,placement['payload']['scene_xml'])))
    data = mujoco.MjData(model)
    contract = runtime_contract(model)
    with np.load(_path(ctx,payload['states_npz'])) as z: states = {k:z[k].copy() for k in z.files}
    with np.load(_path(ctx,trajectory['payload']['trajectory_npz'])) as z:
        trajectory_time=z['time'].copy(); source_time=z['source_time'].copy()
    physics = None
    if payload.get('physics_steps_npz') and _path(ctx,payload['physics_steps_npz']).exists():
        with np.load(_path(ctx,payload['physics_steps_npz'])) as z: physics={k:z[k].copy() for k in z.files}
    archive = physics if physics is not None else states
    times = np.asarray(archive['time']); source = np.interp(times,trajectory_time,source_time)
    domain = request.get('time_domain','sim')
    if domain not in ('sim','source'): raise ValueError('time_domain must be sim/source')
    requested = request.get('times',[float((times if domain=='sim' else source)[-1])])
    bodies = request.get('bodies',[])
    if not 1 <= len(requested) <= 12 or len(bodies)>8: raise ValueError('limit: 1..12 times, <=8 bodies')
    if not np.isfinite(requested).all(): raise ValueError('times must be finite')
    side = request.get('side','both')
    if side not in ('left','right','both'): raise ValueError('invalid side')
    sides = ('left','right') if side=='both' else (side,)
    body_ids = {}
    for name in bodies:
        bid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if bid<0: raise ValueError('unknown body: '+name)
        body_ids[name]=bid
    minimum = float(diag['thresholds'].get('minimum_contact_force',.01))
    samples=[]
    for query in requested:
        axis=times if domain=='sim' else source
        if query < axis[0]-1e-8 or query > axis[-1]+1e-8: raise ValueError('time outside recorded execution')
        i=int(np.argmin(abs(axis-query)))
        if physics is not None:
            mujoco.mj_setState(model,data,physics['integration_state'][i],STATE)
        else:
            mujoco.mj_resetData(model,data)
            data.qpos[:]=states['qpos'][i];data.qvel[:]=states['qvel'][i];data.time=float(times[i])
            data.ctrl[:]=states['ctrl'][i]
        # Match rollout's order: restore state, forward, then assign this step's command.
        mujoco.mj_forward(model,data)
        data.ctrl[:]=archive['control' if physics is not None else 'ctrl'][i]
        hands={}
        for hand in sides:
            arm=contract['arms'][hand]; site=model.site(arm['eef_site']).id; R=data.site_xmat[site].reshape(3,3); p=data.site_xpos[site]
            actuator=model.actuator(arm['gripper']['actuator']).id
            low,high=arm['gripper']['control_range']
            hands[hand]={'pinch_position_m':p.tolist(),'pinch_quaternion_wxyz':_quat(R),
                'pinch_axes_world':{a:R[:,j].tolist() for j,a in enumerate(('x','y','z'))},
                'finger_qpos_m':{f:float(data.qpos[model.jnt_qposadr[model.joint(f).id]]) for f in arm['gripper']['joints']},
                'gripper_ctrl_m':float(data.ctrl[actuator]),'commanded_opening':float((data.ctrl[actuator]-low)/(high-low)),
                'objects_in_pinch_frame':{n:{'position_m':(R.T@(data.xpos[b]-p)).tolist(),
                    'quaternion_wxyz':_quat(R.T@data.xmat[b].reshape(3,3))} for n,b in body_ids.items()}}
        raw=[]
        for j in range(data.ncon):
            c=data.contact[j]; a,b=int(c.geom1),int(c.geom2)
            bids=[int(model.geom_bodyid[g]) for g in (a,b)]
            if body_ids and not any(_descendant(model,k,v) for k in bids for v in body_ids.values()): continue
            if not body_ids and not any((model.geom(g).name or '').startswith(tuple(contract['collision_groups']['arms'][s] for s in sides)) for g in (a,b)):continue
            force=np.zeros(6);mujoco.mj_contactForce(model,data,j,force)
            raw.append({'geoms':[model.geom(g).name or str(g) for g in (a,b)],
                'bodies':[model.body(k).name for k in bids], 'distance_m':float(c.dist),
                'force_contact_frame':force.tolist(),'normal_force_N':float(force[0]),
                'passes_distance':bool(c.dist<=.002),'passes_minimum_force':bool(force[0]>=minimum),
                'passes_contact_filter':bool(c.dist<=.002 and force[0]>=minimum)})
        filtered=_contacts(model,data,minimum)
        paired={hand:{name:_check(model,data,filtered,{'kind':'grasp','side':hand,'body':name}) for name in body_ids} for hand in sides}
        samples.append({'requested_time':float(query),'sim_time':float(times[i]),'source_time':float(source[i]),
            'sample_error_sec':float(axis[i]-query),'hands':hands,'base_xyyaw':base_pose(model,data).tolist(),
            'objects':{n:{'position_m':data.xpos[b].tolist(),'quaternion_wxyz':data.xquat[b].tolist()} for n,b in body_ids.items()},
            'raw_contacts':raw,'paired_fingers':paired})
    phases=trajectory['payload'].get('time_map',[])
    start=float(request.get('start',times[0]));end=float(request.get('end',times[-1]))
    if not np.isfinite([start,end]).all() or start>end:raise ValueError('invalid interval')
    summaries=[]
    for phase in phases:
        if phase['sim_end']<start or phase['sim_start']>end:continue
        metrics=diag.get('hand_object_monitoring',{}).get(phase['phase'],{})
        summaries.append({'phase':phase['phase'],'sim_start':phase['sim_start'],'sim_end':phase['sim_end'],
            'scope':'whole recorded phase; not recomputed for partial interval',
            'metrics':{s:m for s,m in metrics.items() if s in sides and (not bodies or m.get('object') in bodies)}})
    return {'record_id':record['id'],'status':payload['status'],'placement_id':placement['id'],
        'sampling':'nearest physics step' if physics is not None else 'nearest recorded video sample',
        'contact_evidence':'recomputed with mj_forward from recorded state; no simulation stepping',
        'contact_thresholds':{'minimum_force_N':minimum,'maximum_distance_m':.002},
        'samples':samples,'interval':{'start':start,'end':end,'phase_summaries':summaries},
        'runtime':contract,
        'available_task_bodies':[model.body(i).name for i in range(1,model.nbody) if not model.body(i).name.startswith(tuple(contract['robot_geom_prefixes']))],
        'limitations':['Reduced states lack integration warm-start state; prefer physics-step records.',
            'Source-time plateaus map to the earliest nearest sample; use sim time to inspect terminal settle.']}
