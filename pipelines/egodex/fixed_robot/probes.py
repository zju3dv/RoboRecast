"""Advisory contact probes in isolated simulator state, never task certification."""
import uuid
import mujoco
import numpy as np
from .robot import solve_ik
from .trajectory import _controls
from .utils import write_json


def probe_contact(ctx, request):
    """Move physically from committed state to IK target, close and hold.

    Request: side, object (body name), target_position (world metres), optional
    quaternion_wxyz, gripper_opening (normalized 0 closed..1 open), duration
    (hold seconds, default .5), approach_seconds (default 1), close_seconds (.3).
    Probe motion uses a smooth actuator ramp, not a certified collision planner.
    """
    from .rollout import _contacts, _check
    from .diagnostics import classify_collision, render_failure_evidence
    from .execution_reference import load_execution_state
    from .robot_runtime import runtime_contract
    from .base_control import step_base
    state = load_execution_state(ctx, request.get('reference_record_id'))
    model, data = state['model'], state['data']
    contract = runtime_contract(model)
    side=request['side']; body=request.get('object',request.get('body'))
    if side not in contract['arms']:raise ValueError('side must name a declared arm')
    if not body or mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)<0:raise ValueError('object must name a scene body')
    ancestor=model.body(body).id
    while ancestor:
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,ancestor) or ''
        if name.startswith(tuple(contract['robot_geom_prefixes'])):raise ValueError('object must be a task body, not robot')
        ancestor=int(model.body_parentid[ancestor])
    start_time = float(data.time)
    mujoco.mj_forward(model,data)
    target={'position':request['target_position']}
    if 'quaternion_wxyz' in request:target['quaternion_wxyz']=request['quaternion_wxyz']
    ikdata=mujoco.MjData(model);ikdata.qpos[:]=data.qpos;mujoco.mj_forward(model,ikdata)
    ik=solve_ik(model,ikdata,{side:target})
    directory=ctx.path('probes/contact_'+uuid.uuid4().hex[:10]);directory.mkdir(parents=True,exist_ok=True)
    if not ik['success']:
        report={'advisory_only':True,'task_certified':False,'probe_executed':False,'reason':'IK target failed; no physical probe executed','ik':ik,'request':request}
        path=directory/'probe.json';write_json(path,report)
        return {key:value for key,value in {**report,'probe_json':str(path.relative_to(ctx.root))}.items() if key != 'contact_samples'}
    opening=float(request.get('gripper_opening',0))
    approach=float(request.get('approach_seconds',1));close=float(request.get('close_seconds',.3));hold=float(request.get('duration',.5))
    if not np.isfinite([approach,close,hold,opening]).all() or min(approach,close,hold)<=0 or not 0<=opening<=1:
        raise ValueError('positive finite durations and normalized gripper_opening required')
    start=data.ctrl.copy();opened=_controls(model,ikdata.qpos,start,{side:1})
    closed=_controls(model,ikdata.qpos,start,{side:opening})
    # Leave the other arm and gripper exactly at its existing controller command.
    for i in range(model.nu):
        if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) or '') not in contract['arms'][side]['actuators'] + [contract['arms'][side]['gripper']['actuator']]:
            opened[i]=closed[i]=start[i]
    rows=[];samples=[];counts={};next_sample=0.;end=approach+close+hold
    instability=None
    while data.time-start_time<=end+model.opt.timestep/2:
        if not all(np.isfinite(x).all() for x in (data.qpos,data.qvel,data.ctrl)):
            instability={'sim_time':float(data.time),'kind':'nonfinite_physics_state'};break
        t=float(data.time)-start_time
        if t<approach:
            u=t/approach;u=u*u*u*(10+u*(-15+6*u));data.ctrl[:]=start+u*(opened-start)
        else:
            u=min(1.,(t-approach)/close);data.ctrl[:]=opened+u*(closed-opened)
        if t>=next_sample-1e-9:
            contacts=_contacts(model,data,float(request.get('minimum_contact_force',.01)))
            result=_check(model,data,contacts,{'kind':'grasp','side':side,'body':body})
            samples.append({'sim_time':t,'holding':t>=approach+close,'two_finger_contact':result['passed'],**result['measured']})
            for a,b,d,f in contacts:
                category=classify_collision(model,a,b,[body])['category'];counts[category]=counts.get(category,0)+1
            rows.append((t,data.qpos.copy(),data.qvel.copy(),data.ctrl.copy()));next_sample+=1/60
        step_base(model,data,[0.,0.])
        mujoco.mj_forward(model,data)
    held=[x for x in samples if x['holding']]
    failing=next((x for x in held if not x['two_finger_contact']),None)
    states=directory/'states.npz'
    np.savez_compressed(states,time=[r[0] for r in rows],qpos=[r[1] for r in rows],qvel=[r[2] for r in rows],ctrl=[r[3] for r in rows],source_time=np.full(len(rows),float(request.get('source_time', state['reference']['payload']['source_time'] if state['reference'] else 0))))
    report={'advisory_only':True,'task_certified':False,'ik':ik,'request':request,
            'two_finger_hold_fraction':sum(x['two_finger_contact'] for x in held)/max(1,len(held)),
            'continuous_two_finger_hold':bool(held) and failing is None and instability is None,'contact_samples':samples,
            'collision_sample_counts':counts,'states_npz':str(states.relative_to(ctx.root)),
            'first_failure':instability or (dict(failing,kind='missing_two_finger_contact') if failing else None),
            'reference_record_id':request.get('reference_record_id'), 'reference_sim_time':start_time,
            'note':'Isolated physical approach/close/hold with parked prescribed base. Does not change placement, trajectory, rollout or review records.'}
    report['evidence']=render_failure_evidence(ctx,model,states,report,directory)
    path=directory/'probe.json';write_json(path,report)
    return {key:value for key,value in {**report,'probe_json':str(path.relative_to(ctx.root))}.items() if key != 'contact_samples'}
