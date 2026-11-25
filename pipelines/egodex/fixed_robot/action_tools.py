"""Sparse authoring and bounded contact diagnosis; never step physics or change lineage."""
from uuid import uuid4
import copy
import numpy as np
import mujoco
from .utils import read_json, write_json
from .inspection import _path, _record
from .checkpoints import STATE
from .rollout import _descendant
from .targets import vector, rotation


def edit_trajectory(ctx, request):
    """Edit explicitly selected knots into an authoring copy, requiring recompilation."""
    source = _path(ctx, request['trajectory_json_path'])
    original = read_json(source)
    spec = copy.deepcopy(original)
    phases = spec.get('phases', [])
    names = [p['name'] for p in phases]
    if not phases or len(set(names)) != len(names):
        raise ValueError('trajectory requires uniquely named phases')
    changed = []
    edits = request.get('edits', [])
    if not isinstance(edits, list) or len(edits) > 100:
        raise ValueError('edits must be a list of at most 100 sparse operations')
    for edit in edits:
        allowed = {'phase', 'side', 'waypoint_indices', 'world_offset_m', 'quaternion_wxyz', 'gripper_opening'}
        if set(edit) - allowed: raise ValueError('unsupported edit fields: '+str(sorted(set(edit)-allowed)))
        if edit.get('phase') not in names: raise ValueError('unknown phase')
        side = edit.get('side')
        if side not in ('left', 'right'): raise ValueError('side must be left/right')
        indices = edit.get('waypoint_indices')
        pi = names.index(edit['phase'])
        if phases[pi].get('kind')=='transit':raise ValueError('transit has no arm waypoints; generate a new base route and recompile from its reference')
        waypoints = phases[pi]['waypoints']
        if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < len(waypoints) for i in indices):
            raise ValueError('waypoint_indices must explicitly select existing integer indices')
        if len(set(indices)) != len(indices): raise ValueError('duplicate waypoint index')
        if not set(edit) & {'world_offset_m', 'quaternion_wxyz', 'gripper_opening'}:
            raise ValueError('edit requires offset, quaternion, or gripper opening')
        for index in indices:
            knot = waypoints[index]; before = copy.deepcopy(knot)
            if 'world_offset_m' in edit or 'quaternion_wxyz' in edit:
                target = knot.get('targets', {}).get(side)
                if target is None: raise ValueError('selected waypoint has no target for '+side)
                if target.get('type', 'world_pose') != 'world_pose':
                    raise ValueError('world pose edits require a world_pose target; relative targets retain their explicit frame')
                if 'world_offset_m' in edit:
                    key = 'position_m' if 'position_m' in target else 'position'
                    target[key] = (vector(target.get(key), 'position') + vector(edit['world_offset_m'], 'world_offset_m')).tolist()
                if 'quaternion_wxyz' in edit:
                    rotation(edit['quaternion_wxyz'])
                    q = np.asarray(edit['quaternion_wxyz'], float)
                    target['quaternion_wxyz'] = (q/np.linalg.norm(q)).tolist()
            if 'gripper_opening' in edit:
                opening = float(edit['gripper_opening'])
                if not np.isfinite(opening) or not 0 <= opening <= 1: raise ValueError('gripper_opening must be in [0,1]')
                knot.setdefault('grippers', {})[side] = opening
            if knot != before:
                changed.append({'phase': names[pi], 'phase_index': pi, 'side': side, 'waypoint_index': index,
                                'fields': sorted(set(edit)-{'phase','side','waypoint_indices'})})
    if 'hold_committed_arms' in request:
        sides = request['hold_committed_arms']
        if not isinstance(sides, list) or len(set(sides)) != len(sides) or not set(sides) <= {'left','right'}:
            raise ValueError('hold_committed_arms must be unique left/right names')
        for side in sides:
            if any(p.get('hand_objects', {}).get(side, {}).get('mode') != 'idle'
                   or p.get('hand_objects', {}).get(side, {}).get('object') is not None for p in phases):
                raise ValueError('hold_committed_arms requires idle binding with null object in every phase')
        # Explicitly opting into committed hold replaces redundant Cartesian/gripper commands.
        for pi, phase in enumerate(phases):
            for index, knot in enumerate(phase.get('waypoints',[])):
                for side in sides:
                    removed = {}
                    for field in ('targets', 'grippers'):
                        if side in knot.get(field, {}):
                            removed[field] = knot[field].pop(side)
                    if removed:
                        changed.append({'phase': names[pi], 'phase_index': pi, 'side': side,
                                        'waypoint_index': index, 'fields': list(removed),
                                        'operation': 'remove_commands_for_committed_hold', 'removed': removed})
        if sides != spec.get('hold_committed_arms', []):
            spec['hold_committed_arms'] = sides
            changed.append({'phase': names[0], 'phase_index': 0, 'fields': ['hold_committed_arms']})
    # Restrict writes to authoring files: immutable compiled records remain usable until compile.
    destination = _path(ctx, request.get('output_json_path', 'authoring/trajectory-'+uuid4().hex+'.json'))
    if not destination.is_relative_to((ctx.root/'authoring').resolve()) or destination.suffix != '.json':
        raise ValueError('output_json_path must be a .json file under authoring/')
    write_json(destination, spec)
    earliest = min((x['phase_index'] for x in changed), default=None)
    compile_request={'trajectory_json_path':str(destination.relative_to(ctx.root))}
    try:
        current=ctx.latest('trajectory')['payload']
        if source==ctx.path(current['spec_json']) and current.get('reference_record_id'):
            compile_request['reference_record_id']=current['reference_record_id']
    except (ValueError,KeyError):pass
    return {'trajectory_json_path': str(destination.relative_to(ctx.root)), 'changes': changed,
            'earliest_changed_phase': names[earliest] if earliest is not None else None,
            'unchanged_prefix_phases': names[:earliest] if earliest is not None else names,
            'compile_request':compile_request,'requires_compile': bool(changed), 'compiled_lineage_unchanged': True,
            'checkpoint_reuse': 'Actual reusable prefix is measured after compile from exact controls; the phase boundary is only a conservative authoring hint.',
            'next_tool': 'compile_human_guided_trajectory'}


def _gap(times, missing, timestep):
    """Longest missing-pair run; includes one timestep for the final sample."""
    longest = 0.; began = None
    for t, absent in zip(times, missing):
        if absent and began is None: began = float(t)
        if not absent and began is not None:
            longest = max(longest, float(t)-began); began = None
    if began is not None: longest = max(longest, float(times[-1])-began+timestep)
    return longest


def inspect_contact_interval(ctx, request):
    """Inspect all physics samples in <=10 sim seconds, <=5000 states, without stepping."""
    kind = request.get('kind', 'rollout')
    if kind not in ('rollout', 'stage_probe'): raise ValueError('kind must be rollout/stage_probe')
    record = _record(ctx, request['record_id']) if request.get('record_id') else ctx.latest(kind)
    if record['kind'] not in ('rollout', 'stage_probe'): raise ValueError('not an execution record')
    placement = _record(ctx, record['dependencies']['placement'])
    trajectory = _record(ctx, record['dependencies']['trajectory'])
    payload = record['payload']
    if not payload.get('physics_steps_npz'): raise ValueError('physics-step archive required; sparse video states cannot establish continuity')
    with np.load(_path(ctx, payload['physics_steps_npz'])) as z:
        times = z['time'].copy(); states = z['integration_state'].copy(); controls = z['control'].copy()
    with np.load(_path(ctx, trajectory['payload']['trajectory_npz'])) as z:
        source = np.interp(times, z['time'], z['source_time'])
    start, end = float(request['start']), float(request['end'])
    domain = request.get('time_domain', 'sim')
    if domain not in ('sim','source'): raise ValueError('time_domain must be sim/source')
    axis = times if domain == 'sim' else source
    if not np.isfinite([start,end]).all() or not axis[0]-1e-8 <= start <= end <= axis[-1]+1e-8:
        raise ValueError('interval must lie within recorded execution')
    indices = np.flatnonzero((axis >= start-1e-9) & (axis <= end+1e-9))
    if not len(indices): raise ValueError('interval contains no physics samples')
    if len(indices)>5000 or times[indices[-1]]-times[indices[0]]>10.000001:
        raise ValueError('interval exceeds 10 sim seconds or 5000 physics samples; narrow the request')
    side = request['side']; body = request['body']
    if side not in ('left','right'): raise ValueError('side must be left/right')
    diag = read_json(_path(ctx, payload['diagnostics_json']))
    minimum = float(diag['thresholds'].get('minimum_contact_force', .01))
    model = mujoco.MjModel.from_xml_path(str(_path(ctx,placement['payload']['scene_xml'])))
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)
    from .robot_runtime import runtime_contract
    contract = runtime_contract(model)
    arm = contract['arms'][side]
    robot_prefixes = tuple(contract['robot_geom_prefixes'])
    hand = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,arm['eef_body'])
    if bid <= 0 or hand < 0: raise ValueError('unknown body/hand')
    if body.startswith(robot_prefixes): raise ValueError('body must be a task object')
    fingers = tuple(arm['gripper']['finger_bodies'])
    finger_ids = [model.body(name).id for name in fingers]
    if len(fingers) != 2: raise ValueError('Contact interval adapter requires a two-finger gripper')
    raw_presence=[]; filtered_presence=[]; raw_forces=[]; filtered_forces=[]; positions=[]; relative=[]; rotations=[]
    supports = {}; examples=[]; hand_environment = {}
    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, contract['base']['body'])
    for i in indices:
        mujoco.mj_setState(model,data,states[i],STATE)
        mujoco.mj_forward(model,data); data.ctrl[:] = controls[i]
        raw=np.zeros(2,bool); filtered=np.zeros(2,bool); forces=np.zeros(2); accepted=np.zeros(2)
        touching = set(); hand_raw_seen = set(); hand_filtered_seen = set()
        for j in range(data.ncon):
            contact=data.contact[j]; a,b=int(contact.geom1),int(contact.geom2)
            ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
            force=np.zeros(6);mujoco.mj_contactForce(model,data,j,force)
            close=bool(contact.dist<=.002); passed=bool(close and force[0]>=minimum)
            # Track gripper/finger versus environment independently of object contacts.
            for hand_geom, environment_geom, hand_body, environment_body in ((a,b,ba,bb),(b,a,bb,ba)):
                environment_name = model.body(environment_body).name or 'world'
                robot_environment = ((robot_root >= 0 and _descendant(model,environment_body,robot_root))
                                     or environment_name.startswith(robot_prefixes))
                if (not _descendant(model,hand_body,hand) or robot_environment
                        or _descendant(model,environment_body,bid) or not close):
                    continue
                key = (hand_geom, environment_geom)
                row = hand_environment.setdefault(key, {
                    'hand_geom': model.geom(hand_geom).name or str(hand_geom),
                    'environment_geom': model.geom(environment_geom).name or str(environment_geom),
                    'environment_body': environment_name, 'raw_sample_count': 0, 'filtered_sample_count': 0,
                    'maximum_normal_force_N': 0., 'maximum_penetration_m': 0.})
                row['maximum_normal_force_N'] = max(row['maximum_normal_force_N'], float(force[0]))
                row['maximum_penetration_m'] = max(row['maximum_penetration_m'], max(0., -float(contact.dist)))
                hand_raw_seen.add(key)
                if passed: hand_filtered_seen.add(key)
            if _descendant(model,ba,bid): other=b
            elif _descendant(model,bb,bid): other=a
            else: continue
            name=model.geom(other).name or str(other)
            for fi,finger in enumerate(fingers):
                if _descendant(model, int(model.geom_bodyid[other]), finger_ids[fi]):
                    forces[fi]+=float(force[0]); raw[fi] |= close
                    if passed: accepted[fi]+=float(force[0]);filtered[fi]=True
            other_body=model.body(int(model.geom_bodyid[other])).name or 'world'
            if passed and not other_body.startswith(robot_prefixes) and not _descendant(model,int(model.geom_bodyid[other]),bid):
                touching.add(other_body)
        for key in hand_raw_seen: hand_environment[key]['raw_sample_count'] += 1
        for key in hand_filtered_seen: hand_environment[key]['filtered_sample_count'] += 1
        for name in touching: supports[name]=supports.get(name,0)+1
        raw_presence.append(raw);filtered_presence.append(filtered);raw_forces.append(forces);filtered_forces.append(accepted)
        positions.append(data.xpos[bid].copy())
        R=data.xmat[hand].reshape(3,3)
        relative.append(R.T@(data.xpos[bid]-data.xpos[hand])); rotations.append(R.T@data.xmat[bid].reshape(3,3))
        if not filtered.all() and len(examples)<8:
            examples.append({'sim_time':float(times[i]),'source_time':float(source[i]),'raw_fingers':[fingers[k] for k in range(2) if raw[k]],
                             'filtered_fingers':[fingers[k] for k in range(2) if filtered[k]],'raw_normal_force_N':dict(zip(fingers,forces.tolist())),
                             'other_body_contacts':sorted(touching)})
    selected=times[indices]; raw=np.asarray(raw_presence); filtered=np.asarray(filtered_presence)
    pair=filtered.all(axis=1); pos=np.asarray(positions); rel=np.asarray(relative); rotations=np.asarray(rotations)
    anchor=int(np.flatnonzero(pair)[0]) if pair.any() else 0
    slip=np.linalg.norm(rel[anchor:]-rel[anchor],axis=1)
    angles=np.degrees(np.arccos(np.clip((np.einsum('ij,nij->n',rotations[anchor],rotations[anchor:])-1)/2,-1,1)))
    def metrics(presence, force):
        f=np.asarray(force)
        return {'pair_fraction':float(presence.all(axis=1).mean()),
                'finger_fraction':dict(zip(fingers,presence.mean(axis=0).tolist())),
                'max_pair_gap_sec':_gap(selected,~presence.all(axis=1),model.opt.timestep),
                'normal_force_N':{finger:{'mean':float(f[:,k].mean()),'max':float(f[:,k].max()),'min':float(f[:,k].min())} for k,finger in enumerate(fingers)}}
    phase_summaries=[]
    for phase in trajectory['payload'].get('time_map',[]):
        if phase['sim_end'] < selected[0] or phase['sim_start'] > selected[-1]: continue
        phase_summaries.append({'phase':phase['phase'],'binding':phase.get('hand_objects',{}).get(side),
            'recorded_whole_phase_monitor':diag.get('hand_object_monitoring',{}).get(phase['phase'],{}).get(side),
            'scope':'recorded full phase; interval metrics above are separately recomputed'})
    return {'record_id':record['id'],'side':side,'body':body,'physics_samples':len(indices),
            'interval':{'requested_domain':domain,'start':start,'end':end,'sim_start':float(selected[0]),'sim_end':float(selected[-1]),
                        'source_start':float(source[indices[0]]),'source_end':float(source[indices[-1]])},
            'contact_thresholds':{'maximum_distance_m':.002,'minimum_normal_force_N':minimum},
            'raw':metrics(raw,raw_forces),'filtered':metrics(filtered,filtered_forces),
            'strict_every_step_grasp_passed':bool(pair.all()),
            'motion':{'slip_reference_sim_time':float(selected[anchor]),'slip_reference':'first filtered pair contact, otherwise interval start',
                      'max_relative_slip_m':float(slip.max()),'max_relative_rotation_deg':float(angles.max()),
                      'object_displacement_m':(pos[-1]-pos[0]).tolist(),'object_height_range_m':[float(pos[:,2].min()),float(pos[:,2].max())]},
            'other_body_contact_fraction':{k:v/len(indices) for k,v in supports.items()},
            'hand_environment_contacts':[{**row, 'raw_sample_fraction': row['raw_sample_count']/len(indices),
                                          'filtered_sample_fraction': row['filtered_sample_count']/len(indices)}
                                         for _,row in sorted(hand_environment.items())],
            'first_missing_pair_samples':examples,'phase_summaries':phase_summaries,
            'limitations':['Read-only mj_forward recomputation from every recorded integration state; no simulation stepping.',
                'Raw occupancy uses distance only; raw force includes all MuJoCo contact candidates. Filtered occupancy applies the recorded force threshold.',
                'other_body_contact_fraction measures object-support contact; hand_environment_contacts measures selected gripper/fingers versus environment, excluding the selected object and robot bodies. These are interference candidates, not causal or invalid-collision verdicts.',
                'Strict every-step grasp and phase hold gap/slip policies are distinct; this diagnostic changes neither acceptance policy nor recorded verdict.',
                'Source-time intervals include every matching sample, including terminal plateaus. Use sim time to narrow them.']}
