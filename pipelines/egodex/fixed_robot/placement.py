"""Revision-bound fixed-base candidate inspection and exact-pose commitment."""
import copy
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .robot import initial_state, solve_ik, collision_report
from .robot_bundles import install_robot_bundle
from .robot_runtime import runtime_contract
from .base_control import step_base
from .scene import _directory, _relative, merge_xml, render_image
from .placement_visuals import bounds_points, camera_spec, camera, sheet, detail_views
from .utils import digest


def _combined(ctx,scene,out,xyyaw,arm_qpos):
    from .inputs import load_reference
    meta, _ = load_reference(ctx.input_dir)
    context_path = ctx.root/'.agent_run/run.json'
    context = json.loads(context_path.read_text()) if context_path.exists() else {}
    bundle_id = context.get('robot_bundle_id', context.get('robot_bundle', 'fixed_dual_vx300s'))
    if isinstance(bundle_id, dict): bundle_id = bundle_id.get('bundle_id', 'fixed_dual_vx300s')
    bundle = install_robot_bundle(out, meta.get('motion_mode', 'not_mobile'), xyyaw, bundle_id)
    robot = bundle['scene_xml']
    root=ET.parse(robot).getroot(); taskroot=ET.parse(ctx.path(scene['scene_xml'])).getroot()
    # Task owns its floor; avoid a duplicate contact plane.
    world=root.find('worldbody')
    for geom in list(world.findall('geom')):
        if geom.get('name')=='floor': world.remove(geom)
    merge_xml(root,taskroot)
    xml=out/'combined.xml';ET.ElementTree(root).write(xml,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    data.qpos[:],data.ctrl[:]=initial_state(model)
    taskmodel=mujoco.MjModel.from_xml_path(str(ctx.path(scene['scene_xml'])))
    state=np.load(ctx.path(scene['settled_state']))
    for j in range(taskmodel.njnt):
        name=mujoco.mj_id2name(taskmodel,mujoco.mjtObj.mjOBJ_JOINT,j)
        if not name: raise ValueError('task joints must be named for settled state transfer')
        nj=model.joint(name).id; old=taskmodel.jnt_qposadr[j];new=model.jnt_qposadr[nj]
        width=7 if taskmodel.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 4 if taskmodel.jnt_type[j]==mujoco.mjtJoint.mjJNT_BALL else 1
        data.qpos[new:new+width]=state['qpos'][old:old+width]
    if arm_qpos is not None:
        values=np.asarray(arm_qpos,float)
        arms=runtime_contract(model)['arms']
        if values.shape!=(sum(len(a['joints']) for a in arms.values()),) or not np.isfinite(values).all():raise ValueError('initial_arm_qpos must match bundle arm joints')
        offset=0
        for arm in arms.values():
            for i,(name,actuator) in enumerate(zip(arm['joints'],arm['actuators'])):
                joint=model.joint(name)
                if not model.jnt_range[joint.id,0]<=values[offset+i]<=model.jnt_range[joint.id,1]:raise ValueError('initial arm joint exceeds limits')
                data.qpos[joint.qposadr]=values[offset+i];data.ctrl[model.actuator(actuator).id]=values[offset+i]
            offset+=len(arm['joints'])
    mujoco.mj_forward(model,data)
    return xml,model,data


def inspect_robot_placement(ctx,request):
    scene=ctx.latest('scene')['payload'];out=_directory(ctx,'placement_preview')
    prior=np.asarray(request['prior_xyyaw'],float)
    if prior.shape!=(3,) or not np.isfinite(prior).all():raise ValueError('prior_xyyaw requires finite x,y,yaw')
    poses=request.get('candidates_xyyaw',[prior.tolist(),(prior+[.1,0,0]).tolist(),(prior+[-.1,0,0]).tolist(),(prior+[0,.1,.15]).tolist(),(prior+[0,-.1,-.15]).tolist()])
    events=request.get('events',[])
    if not events:raise ValueError('placement requires synchronized task events')
    from .inputs import load_reference
    from scipy.spatial.transform import Rotation
    meta, reference = load_reference(ctx.input_dir)
    alignment = np.asarray(meta['T_sim_source_world'])
    for event in events:
        t = float(event['source_time'])
        if not np.isfinite(t) or not reference['timestamps'][0] <= t <= reference['timestamps'][-1]:
            raise ValueError('placement event time must lie within the source episode')
        row = int(np.argmin(abs(reference['timestamps'] - t)))
        for side in event['targets']:
            if side not in ('left', 'right') or not reference[f'{side}_valid'][row]:
                raise ValueError('placement key event needs a valid source hand observation')
    if not 1<=len(poses)<=12:raise ValueError('provide 1 to 12 candidates')
    candidates=[]; files=[]; states=[]; shared_points=[]
    for i,pose in enumerate(poses):
        pose=np.asarray(pose,float)
        if pose.shape!=(3,) or not np.isfinite(pose).all():raise ValueError('invalid candidate pose')
        folder=out/f'candidate_{i}';folder.mkdir()
        xml,model,data=_combined(ctx,scene,folder,pose,request.get('initial_arm_qpos'))
        init_q=data.qpos.copy(); init_ctrl=data.ctrl.copy(); initial_collisions=collision_report(model,data)
        state=folder/'initial_state.npz';np.savez_compressed(state,qpos=init_q,qvel=np.zeros(model.nv),ctrl=init_ctrl)
        init_image=folder/'initial.png'
        shared_points.extend(bounds_points(model,data)); states.append((model,init_q,init_ctrl,init_image))
        initial_end_effectors = {}
        for side in ('left', 'right'):
            site = model.site(runtime_contract(model)['arms'][side]['eef_site']).id
            q = Rotation.from_matrix(data.site_xmat[site].reshape(3,3)).as_quat()
            initial_end_effectors[side] = {'position': data.site_xpos[site].tolist(), 'quaternion_wxyz': q[[3,0,1,2]].tolist()}
        checks=[];hardest=None;worst=-1
        for event in events:
            check=solve_ik(model,data,event['targets']);check['source_time']=event['source_time']
            row = int(np.argmin(abs(reference['timestamps'] - float(event['source_time']))))
            source_references = {}
            for side, target in event['targets'].items():
                center = (alignment @ np.r_[reference[f'{side}_center'][row], 1])[:3]
                confidence = float(reference[f'{side}_confidence'][row])
                source_references[side] = {'center_sim_m': center.tolist(), 'confidence': confidence if np.isfinite(confidence) else None,
                    'target_offset_m': float(np.linalg.norm(center - np.asarray(target['position'])))}
            check['source_reference'] = source_references
            checks.append(check)
            severity=max(e['position_m']+.2*e['orientation_rad'] for e in check['errors'].values())+len(check['collisions'])
            if severity>worst:worst=severity;hardest=data.qpos.copy()
        data.qpos[:]=hardest;mujoco.mj_forward(model,data)
        hard_image=folder/'hardest_event.png';render_image(model,data,hard_image)
        candidate={'candidate_id':f'candidate_{i}','base_xyyaw':pose.tolist(),'scene_xml':_relative(ctx,xml),'initial_state':_relative(ctx,state),'initial_collisions':initial_collisions,'initial_end_effectors':initial_end_effectors,'assets':[_relative(ctx,f) for f in [*sorted((folder/'robot_assets').iterdir()),folder/'manifest.json',folder/'runtime_contract.json',folder/'robot.xml',folder/'robot_initial_state.npz']], 'bundle_id':runtime_contract(model)['bundle_id'], 'motion_mode':runtime_contract(model)['motion_mode'],'events':checks,'feasible':not initial_collisions and all(c['success'] for c in checks),'renders':{'initial':_relative(ctx,init_image),'hardest_event':_relative(ctx,hard_image)}}
        candidates.append(candidate);files.extend([candidate['scene_xml'],candidate['initial_state'],*candidate['renders'].values(),*candidate['assets']])
    spec=camera_spec(shared_points)
    for model,qpos,ctrl,path in states:
        data=mujoco.MjData(model);data.qpos[:]=qpos;data.ctrl[:]=ctrl;mujoco.mj_forward(model,data)
        render_image(model,data,path,camera=camera(spec))
    comparison=out/'candidate_comparison.png'
    sheet([state[3] for state in states],[f"{c['candidate_id']} | feasible: {'yes' if c['feasible'] else 'no'}" for c in candidates],comparison,len(candidates))
    files.append(_relative(ctx,comparison))
    return ctx.put('placement_preview',{'comparison_sheet':_relative(ctx,comparison),'shared_camera':spec,'initial_arm_qpos':request.get('initial_arm_qpos'),'candidates':candidates,'events':events,'prior_xyyaw':prior.tolist(),'camera':'one shared camera fitted to union of all candidate robot and task geometry; candidate order is not a preference','note':'Discrete-event IK and penetration screen, not trajectory or grasp certification.'},dependencies=['scene'],files=files)


def preview_robot_placement(ctx,request):
    """Create evidence only: no placement commit, hold simulation, or rollout mutation."""
    preview=ctx.latest('placement_preview')
    if preview['id']!=request['preview_id']:raise ValueError('preview is not current')
    matches=[c for c in preview['payload']['candidates'] if c['candidate_id']==request['candidate_id']]
    if len(matches)!=1:raise ValueError('unknown candidate_id')
    candidate=copy.deepcopy(matches[0]);out=_directory(ctx,'selected_placement_preview')
    if request.get('initial_arm_qpos') is not None:
        # Reuse the complete synchronized-event screen in an isolated artifact store.
        # This never replaces the live candidate set or previously committed placement.
        from .store import Store
        scratch=Store(out/'adjustment',ctx.input_dir)
        scene=copy.deepcopy(ctx.latest('scene')['payload'])
        for key in ('scene_xml','settled_state'): scene[key]=str(ctx.path(scene[key]))
        class AdjustmentContext:
            input_dir=ctx.input_dir
            root=scratch.root
            def path(self,path):
                from pathlib import Path
                return Path(path) if Path(path).is_absolute() else scratch.path(path)
            def latest(self,kind):
                if kind=='scene':return {'id':ctx.latest('scene')['id'],'payload':scene}
                return scratch.latest(kind)
            def put(self,kind,payload,dependencies=(),files=()):return {'payload':payload}
        adjusted=inspect_robot_placement(AdjustmentContext(),{'prior_xyyaw':candidate['base_xyyaw'],
            'candidates_xyyaw':[candidate['base_xyyaw']],'events':preview['payload']['events'],
            'initial_arm_qpos':request['initial_arm_qpos']})['payload']['candidates'][0]
        for key in ('scene_xml','initial_state'):adjusted[key]=_relative(ctx,scratch.root/adjusted[key])
        adjusted['assets']=[_relative(ctx,scratch.root/p) for p in adjusted['assets']]
        adjusted['renders']={k:_relative(ctx,scratch.root/p) for k,p in adjusted['renders'].items()}
        adjusted['candidate_id']=candidate['candidate_id'];candidate=adjusted
    model=mujoco.MjModel.from_xml_path(str(ctx.path(candidate['scene_xml'])));data=mujoco.MjData(model)
    init=np.load(ctx.path(candidate['initial_state']));data.qpos[:]=init['qpos'];data.ctrl[:]=init['ctrl'];mujoco.mj_forward(model,data)
    renders,comparison,spec=detail_views(model,data,out)
    pose_id=digest({'scene_xml':candidate['scene_xml'],'qpos':data.qpos.tolist(),'ctrl':data.ctrl.tolist()})
    candidate.update(preview_id=preview['id'],pose_id=pose_id,
        initial_arm_qpos=[float(data.qpos[model.joint(name).qposadr[0]]) for arm in runtime_contract(model)['arms'].values() for name in arm['joints']])
    payload={'candidate':candidate,'preview_id':preview['id'],'candidate_id':candidate['candidate_id'],
        'pose_id':pose_id,'comparison_sheet':_relative(ctx,comparison),
        'renders':{k:_relative(ctx,p) for k,p in renders.items()},'global_camera':spec,
        'canonical_pinch_poses':candidate['initial_end_effectors'],'feasible':candidate['feasible'],
        'note':'Exact initial-state FK; visual preview does not simulate or commit. Review naturalness visually; no numeric ranking.'}
    return ctx.put('selected_placement_preview',payload,dependencies=['scene','placement_preview'],
        files=[candidate['scene_xml'],candidate['initial_state'],*candidate['assets'],_relative(ctx,comparison),*payload['renders'].values()])


def commit_robot_placement(ctx,request):
    preview=ctx.latest('placement_preview')
    if preview['id']!=request['preview_id']:raise ValueError('preview is not current')
    candidates=[c for c in preview['payload']['candidates'] if c['candidate_id']==request['candidate_id']]
    if len(candidates)!=1:raise ValueError('unknown candidate_id')
    selected=ctx.latest('selected_placement_preview')
    if selected['payload']['preview_id']!=preview['id'] or selected['payload']['candidate_id']!=request['candidate_id']:
        raise ValueError('preview selected candidate before committing; exact latest detailed preview is required')
    if request.get('selected_preview_id') and request['selected_preview_id']!=selected['id']:
        raise ValueError('selected_preview_id is not current')
    candidate=copy.deepcopy(selected['payload']['candidate'])
    candidate['selected_preview_id']=selected['id']
    if not candidate['feasible']:raise ValueError('candidate failed synchronized IK/collision checks')
    model=mujoco.MjModel.from_xml_path(str(ctx.path(candidate['scene_xml'])));data=mujoco.MjData(model)
    init=np.load(ctx.path(candidate['initial_state']));data.qpos[:]=init['qpos'];data.ctrl[:]=init['ctrl']
    initial=data.qpos.copy();mujoco.mj_forward(model,data)
    if collision_report(model,data):raise ValueError('initial state collides')
    for _ in range(round(.5/model.opt.timestep)):step_base(model,data,[0,0])
    drift=float(np.max(np.abs(data.qpos-initial)));speed=float(np.max(np.abs(data.qvel)))
    collisions=collision_report(model,data)
    if not np.isfinite(data.qpos).all() or drift>.08 or speed>.2 or collisions:
        raise ValueError(f'combined initial hold unstable: drift={drift}, speed={speed}, collisions={collisions}')
    # Save/reference the exact preview init, never substitute the held state.
    candidate.update(base_mode=runtime_contract(model)['motion_mode'],events=preview['payload']['events'],hold_diagnostics={'duration_s':.5,'max_qpos_drift':drift,'max_speed':speed},preview_id=preview['id'])
    return ctx.put('placement',candidate,dependencies=['scene','placement_preview'],files=[candidate['scene_xml'],candidate['initial_state'],*candidate['assets']])


def revalidate_robot_placement(ctx,request):
    """Keep a selected pose after a bounded physics-only revision, without rendering."""
    from .scene_compatibility import historical_record, require_same_layout
    previous=historical_record(ctx,request['previous_placement_id'],'placement')
    old_scene=historical_record(ctx,previous['dependencies']['scene'],'scene')
    current=ctx.latest('scene')
    require_same_layout(ctx,old_scene['payload'],current['payload'])
    if current['payload'].get('reused_scene_id')!=old_scene['id']:
        raise ValueError('first merge_and_settle_scene with reuse_scene_id equal to the previous placement scene')
    task_model=mujoco.MjModel.from_xml_path(str(ctx.path(current['payload']['scene_xml'])))
    old_q=np.load(ctx.path(old_scene['payload']['settled_state']))['qpos']
    new_q=np.load(ctx.path(current['payload']['settled_state']))['qpos']
    displacement=np.zeros(task_model.nv)
    mujoco.mj_differentiatePos(task_model,displacement,1.,old_q,new_q)
    # Conservative bound applies in metres to translations and radians to rotations.
    max_delta=float(np.max(np.abs(displacement))) if displacement.size else 0.
    if max_delta>.01:
        raise ValueError(f're-settled layout moved beyond 0.01 m/rad: {max_delta}; select placement anew')
    old=previous['payload'];out=_directory(ctx,'placement_revalidation')
    old_model=mujoco.MjModel.from_xml_path(str(ctx.path(old['scene_xml'])))
    old_init=np.load(ctx.path(old['initial_state']))
    arms=[float(old_init['qpos'][old_model.joint(name).qposadr[0]]) for arm in runtime_contract(old_model)['arms'].values() for name in arm['joints']]
    xml,model,data=_combined(ctx,current['payload'],out,old['base_xyyaw'],arms)
    mapped_joints={name for arm in runtime_contract(old_model)['arms'].values() for name in [*arm['joints'],*arm['gripper']['joints']]}
    for j in range(old_model.njnt):
        name=mujoco.mj_id2name(old_model,mujoco.mjtObj.mjOBJ_JOINT,j)
        if name in mapped_joints:
            data.qpos[model.joint(name).qposadr]=old_init['qpos'][old_model.joint(name).qposadr]
    for a in range(old_model.nu):
        name=mujoco.mj_id2name(old_model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)
        data.ctrl[model.actuator(name).id]=old_init['ctrl'][a]
    mujoco.mj_forward(model,data)
    initial=data.qpos.copy();ctrl=data.ctrl.copy()
    initial_collisions=collision_report(model,data)
    if initial_collisions: raise ValueError(f'revalidated initial state collides: {initial_collisions}')
    checks=[]
    for event in old['events']:
        check=solve_ik(model,data,event['targets']);check['source_time']=event['source_time'];checks.append(check)
    if not all(c['success'] for c in checks): raise ValueError(f'revalidated synchronized IK/collision checks failed: {checks}')
    data.qpos[:]=initial;data.qvel[:]=0;data.ctrl[:]=ctrl;mujoco.mj_forward(model,data)
    for _ in range(round(.5/model.opt.timestep)):step_base(model,data,[0,0])
    drift=float(np.max(np.abs(data.qpos-initial)));speed=float(np.max(np.abs(data.qvel)))
    collisions=collision_report(model,data)
    if not np.isfinite(data.qpos).all() or drift>.08 or speed>.2 or collisions:
        raise ValueError(f'revalidated initial hold unstable: drift={drift}, speed={speed}, collisions={collisions}')
    state=out/'initial_state.npz';np.savez_compressed(state,qpos=initial,qvel=np.zeros(model.nv),ctrl=ctrl)
    candidate={'base_mode':runtime_contract(model)['motion_mode'],'base_xyyaw':old['base_xyyaw'],'initial_arm_qpos':arms,
        'scene_xml':_relative(ctx,xml),'initial_state':_relative(ctx,state),
        'initial_end_effectors':copy.deepcopy(old.get('initial_end_effectors',{})),
        'assets':[_relative(ctx,f) for f in [*sorted((out/'robot_assets').iterdir()),out/'manifest.json',out/'runtime_contract.json',out/'robot.xml',out/'robot_initial_state.npz']],
        'bundle_id':runtime_contract(model)['bundle_id'], 'motion_mode':runtime_contract(model)['motion_mode'],
        'events':old['events'],'event_checks':checks,'feasible':True,'initial_collisions':initial_collisions,
        'revalidated_from':previous['id'],'visual_selection_origin':old.get('visual_selection_origin',old.get('selected_preview_id')),
        'hold_diagnostics':{'duration_s':.5,'max_qpos_drift':drift,'max_speed':speed},
        'layout_revalidation':{'max_settled_delta_m_or_rad':max_delta,'limit':.01},
        'note':'Same selected base/joints; physics-only revision revalidated without a new visual decision. Prior rollout/checkpoints are invalid.'}
    return ctx.put('placement',candidate,dependencies=['scene'],files=[candidate['scene_xml'],candidate['initial_state'],*candidate['assets']])
