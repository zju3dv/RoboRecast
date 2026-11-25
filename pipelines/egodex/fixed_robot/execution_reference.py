"""Immutable actual stage boundaries for suffix planning; never reset scene objects."""
from pathlib import Path
import uuid
import numpy as np
import mujoco
from . import checkpoints
from .scene_compatibility import historical_record
from .utils import read_json
from .robot_runtime import base_pose


def _arrays(ctx, record):
    with np.load(ctx.path(record['payload']['trajectory_npz'])) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _same_prefix(ctx, left, right, end, dt):
    """Compare commands actually consumed before the boundary, plus endpoint ctrl."""
    a, b = _arrays(ctx, left), _arrays(ctx, right)
    if any(float(x['time'][-1]) < end-1e-8 or abs(float(x['time'][0])) > 1e-8 for x in (a,b)):
        raise ValueError('current trajectory does not cover the selected reference prefix')
    samples = np.arange(round(end/dt)+1, dtype=float)*dt
    samples[-1] = end
    for key in ('ctrl',):
        if a[key].shape[1:] != b[key].shape[1:]:
            raise ValueError('reference control layout changed')
        for j in range(a[key].shape[1]):
            if not np.allclose(np.interp(samples,a['time'],a[key][:,j]), np.interp(samples,b['time'],b[key][:,j]), atol=1e-10,rtol=0):
                raise ValueError('reference invalidated: arm control prefix changed')
    for time in samples[:-1]:
        commands = [checkpoints.base_command_at(time,x['time'],x.get('base_commands',np.zeros((len(x['time']),2)))) for x in (a,b)]
        if not np.allclose(*commands, atol=1e-12, rtol=0):
            raise ValueError('reference invalidated: base command prefix changed')
    def phase_semantics(record):
        return [{'phase': p['phase'], 'kind': p.get('kind','parked_manipulation'),
                 'hand_objects': p.get('hand_objects',{}),
                 'sim_start': float(p['sim_start']), 'sim_end': min(float(p['sim_end']),end),
                 'source_start': float(p['source_start']), 'source_end': float(p['source_end'])}
                for p in record['payload']['time_map'] if float(p['sim_start'])<end-1e-9]
    if phase_semantics(left) != phase_semantics(right):
        raise ValueError('reference invalidated: frozen phase/binding semantics changed')
    # The physical prefix may match while semantic phase/source timing changed.
    if not np.allclose(np.interp(samples,a['time'],a['source_time']), np.interp(samples,b['time'],b['source_time']),atol=1e-8,rtol=0):
        raise ValueError('reference invalidated: source time prefix changed')


def validate_trajectory_lineage(ctx, trajectory_record):
    """Check immutable prefix/ref/route graph against current physical identity.

    Ancestors are historical, not latest dependencies: adding a valid suffix does
    not stale its parent. Only selection of an old planning reference additionally
    compares against the latest active trajectory in load_execution_state.
    """
    placement = ctx.latest('placement')
    model = mujoco.MjModel.from_xml_path(str(ctx.path(placement['payload']['scene_xml'])))
    identity = checkpoints.identity(ctx, placement, model)
    seen, visiting = set(), set()
    def trajectory(identifier):
        if identifier in seen:
            return historical_record(ctx, identifier, 'trajectory')
        if identifier in visiting:
            raise ValueError('cyclic trajectory lineage')
        visiting.add(identifier)
        record = historical_record(ctx, identifier, 'trajectory')
        if record['dependencies'].get('placement') != placement['id']:
            raise ValueError('trajectory lineage placement changed')
        payload = record['payload']
        if payload.get('physics_identity') and payload['physics_identity'] != identity:
            raise ValueError('trajectory lineage physics identity changed')
        rid = payload.get('reference_record_id')
        if rid:
            ref = historical_record(ctx, rid, 'execution_reference')
            r = ref['payload']
            if r['placement_id'] != placement['id'] or r['physics_identity'] != identity:
                raise ValueError('execution reference physics/placement changed')
            parent = trajectory(r['trajectory_id'])
            if payload.get('frozen_prefix_trajectory_id') != parent['id'] or abs(float(payload['frozen_prefix_end'])-float(r['sim_time']))>1e-8:
                raise ValueError('trajectory frozen prefix does not match reference')
            executed = historical_record(ctx,r['execution_id'],r['execution_id'].rsplit('-',1)[0])
            if executed['dependencies'].get('trajectory') != parent['id'] or executed['dependencies'].get('placement') != placement['id'] or executed['payload']['status'] not in ('PASS','PROBE_OK'):
                raise ValueError('reference execution does not certify its parent prefix')
            _same_prefix(ctx, parent, record, float(r['sim_time']), float(model.opt.timestep))
        elif payload.get('frozen_prefix_trajectory_id'):
            raise ValueError('frozen prefix requires an execution reference')
        route_id = payload.get('route_id')
        if route_id:
            route = historical_record(ctx, route_id, 'base_route')['payload']
            if route.get('placement_id') != placement['id'] or route.get('reference_record_id') != rid:
                raise ValueError('route does not match trajectory placement/reference')
            if route.get('physics_identity') and route['physics_identity'] != identity:
                raise ValueError('route physics identity changed')
        visiting.remove(identifier);seen.add(identifier)
        return record
    trajectory(trajectory_record['id'])
    return {'valid': True, 'trajectory_id': trajectory_record['id'], 'validated_trajectories': len(seen), 'physics_identity': identity}


def load_execution_state(ctx, reference_record_id=None):
    placement=ctx.latest('placement')
    model=mujoco.MjModel.from_xml_path(str(ctx.path(placement['payload']['scene_xml'])))
    data=mujoco.MjData(model); reference=None; prefix=None
    if reference_record_id:
        reference=historical_record(ctx,reference_record_id,'execution_reference')
        p=reference['payload']
        if p['placement_id']!=placement['id'] or p['physics_identity']!=checkpoints.identity(ctx,placement,model):
            raise ValueError('reference invalidated by placement/physics change')
        prefix=historical_record(ctx,p['trajectory_id'],'trajectory')
        validate_trajectory_lineage(ctx,prefix)
        latest=ctx.latest('trajectory')
        validate_trajectory_lineage(ctx,latest)
        _same_prefix(ctx,prefix,latest,float(p['sim_time']),float(model.opt.timestep))
        execution_kind=p['execution_id'].rsplit('-',1)[0]
        executed=historical_record(ctx,p['execution_id'],execution_kind)
        if executed['dependencies'].get('trajectory') != prefix['id'] or executed['payload']['status'] not in ('PASS','PROBE_OK'):
            raise ValueError('reference execution does not certify the selected prefix')
        with np.load(ctx.path(p['state_npz'])) as a:
            mujoco.mj_setState(model,data,a['integration_state'],checkpoints.STATE)
    else:
        with np.load(ctx.path(placement['payload']['initial_state'])) as a:
            data.qpos[:]=a['qpos'];data.qvel[:]=a['qvel'];data.ctrl[:]=a['ctrl']
    warm=data.qacc_warmstart.copy();mujoco.mj_forward(model,data);data.qacc_warmstart[:]=warm
    return dict(model=model,data=data,placement=placement,reference=reference,time=float(data.time),prefix_trajectory=prefix)


def create_execution_reference(ctx,request):
    kind=request.get('execution_kind','stage_probe')
    if kind not in ('stage_probe','rollout'):raise ValueError('execution_kind must be stage_probe or rollout')
    execution=ctx.latest(kind)
    if execution['payload']['status'] not in ('PROBE_OK','PASS'):
        raise ValueError('only a passing executed prefix can become a planning reference')
    trajectory=ctx.latest('trajectory');placement=ctx.latest('placement')
    validate_trajectory_lineage(ctx,trajectory)
    phase=next((p for p in trajectory['payload']['time_map'] if p['phase']==request['phase_name']),None)
    if phase is None:raise ValueError('unknown executed phase')
    end=float(phase['sim_end'])
    with np.load(ctx.path(execution['payload']['physics_steps_npz'])) as a:
        index=int(np.argmin(abs(a['time']-end)))
        state=a['integration_state'][index].copy();actual=float(a['time'][index]);ctrl=a['control'][index].copy()
    model=mujoco.MjModel.from_xml_path(str(ctx.path(placement['payload']['scene_xml'])))
    if abs(actual-end)>model.opt.timestep/2+1e-8:raise ValueError('execution did not reach exact stage boundary')
    # Mobile compilation snaps boundaries to physics ticks; legacy fixed stages may not.
    if abs(actual-end)>1e-7:raise ValueError('stage boundary must align to a physics timestep')
    data=mujoco.MjData(model);mujoco.mj_setState(model,data,state,checkpoints.STATE);data.ctrl[:]=ctrl
    warm=data.qacc_warmstart.copy();mujoco.mj_forward(model,data);data.qacc_warmstart[:]=warm
    mujoco.mj_getState(model,data,state,checkpoints.STATE)
    from .robot_runtime import runtime_contract
    contract=runtime_contract(model)
    if contract['base']['joints']:
        speed=np.array([data.qvel[model.jnt_dofadr[model.joint(n).id]] for n in contract['base']['joints']])
        if np.max(abs(speed))>1e-6:raise ValueError('reference requires a stopped base')
    path=ctx.path(Path('execution_references')/uuid.uuid4().hex/'state.npz')
    np.savez_compressed(path,integration_state=state,qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl)
    payload=dict(state_npz=str(path.relative_to(ctx.root)),placement_id=placement['id'],trajectory_id=trajectory['id'],execution_id=execution['id'],phase_name=phase['phase'],sim_time=actual,source_time=phase['source_end'],base_xyyaw=base_pose(model,data).tolist(),hand_objects=phase['hand_objects'],physics_identity=checkpoints.identity(ctx,placement,model))
    # Upstream records are immutable historical links: latest trajectory legitimately changes
    # when a suffix is compiled. Validate their full graph explicitly on every load.
    return ctx.put('execution_reference',payload,dependencies=('placement',),files=(path,))
