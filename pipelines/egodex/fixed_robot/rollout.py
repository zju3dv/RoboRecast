"""Frozen-control rollout with exact physics-prefix reuse and measured checks.

success_json_path schema: {events:[{name,source_start,source_end,checks:[...]}],
terminal_checks:[...], settle_seconds:1, terminal_window_seconds:.5}. Check kinds: grasp (side,body),
release (side,body), support (body,support), position (body,target,tolerance),
relative_position (body,relative_to,target,tolerance), height (body,min,max),
speed (body,max). Events require every physics-step state in their source interval
to pass, except grasp_acquired (side, body, min_dwell_sec), which requires a
contiguous bilateral-contact dwell within its declared event window. A grasp means measured
contact with BOTH named fingers, not a closed command or object proximity.
"""
from pathlib import Path
import uuid
import time
import numpy as np
import mujoco
from .utils import read_json, write_json
from .robot_runtime import runtime_contract, base_pose


def _body(model, name):
    i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
    if i<0: raise ValueError(f"unknown body: {name}")
    return i


def _descendant(model, body, root):
    while body>0:
        if body==root:return True
        body=int(model.body_parentid[body])
    return body==root


def _contacts(model,data,minimum_force):
    result=[]
    for c in range(data.ncon):
        contact=data.contact[c]; force=np.zeros(6); mujoco.mj_contactForce(model,data,c,force)
        if contact.dist>0.002 or force[0]<minimum_force:continue
        a,b=int(contact.geom1),int(contact.geom2)
        result.append((a,b,float(contact.dist),float(force[0])))
    return result


def _check(model,data,contacts,check):
    kind=check["kind"]; body=_body(model,check["body"])
    if kind == "joint_range":
        joint = model.joint(check["joint"])
        if int(joint.bodyid[0]) != body or int(joint.type[0]) not in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("joint_range needs a slide or hinge on the named body")
        value = float(data.qpos[int(joint.qposadr[0])])
        return {"check":check, "passed": float(check["min"]) <= value <= float(check["max"]), "measured":{"qpos":value}}
    if kind == "inside_tray":
        from roborecast.rollout import _geom_bounds_in_frame
        tray = model.geom(check["floor"]).id
        if model.geom_type[tray] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("inside_tray needs a rectangular floor geom")
        orientation = data.geom_xmat[tray].reshape(3, 3).T
        bounds = [_geom_bounds_in_frame(model, data, g, orientation, data.geom_xpos[tray])
                  for g in range(model.ngeom) if _descendant(model, int(model.geom_bodyid[g]), body)
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
        if not bounds:
            raise ValueError("The tracked body needs collision geometry")
        limit = model.geom_size[tray,:2] - float(check.get("inset_m", 0))
        lo = np.min([b[0][:2] for b in bounds], axis=0)
        hi = np.max([b[1][:2] for b in bounds], axis=0)
        clearance = float(np.min(np.minimum(lo + limit, limit - hi)))
        return {"check":check,"passed":clearance >= 0,"measured":{"edge_clearance_m":clearance}}
    own_geom = model.geom(check["geom"]).id if check.get("geom") else None
    support_geom = model.geom(check["support_geom"]).id if check.get("support_geom") else None
    touching=[]
    for a,b,d,f in contacts:
        ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
        if _descendant(model,ba,body) and own_geom in (None,a):touching.append((b,bb))
        if _descendant(model,bb,body) and own_geom in (None,b):touching.append((a,ba))
    measured={}
    if kind in ("grasp","release"):
        side=check["side"]
        if side not in ("left","right"):raise ValueError("contact side must be left/right")
        fingers=set()
        finger_bodies=runtime_contract(model)['arms'][side]['gripper']['finger_bodies']
        for geom,body_id in touching:
            for name in finger_bodies:
                if _descendant(model,body_id,model.body(name).id):fingers.add(name)
        measured={"fingers":sorted(fingers)}
        passed=len(fingers)==2 if kind=="grasp" else not fingers
    elif kind=="support":
        support=_body(model,check["support"])
        passed=any(_descendant(model,b,support) and support_geom in (None,g) for g,b in touching)
        measured={"support_contact":passed}
    elif kind in ("position","relative_position"):
        position=data.xpos[body].copy()
        if kind=="relative_position":position-=data.xpos[_body(model,check["relative_to"])]
        error=float(np.linalg.norm(position-np.asarray(check["target"],float)))
        passed=error<=float(check.get("tolerance",.03)); measured={"error_m":error,"position":position.tolist()}
    elif kind=="height":
        z=float(data.xpos[body,2]);passed=float(check.get("min",-np.inf))<=z<=float(check.get("max",np.inf));measured={"height_m":z}
    elif kind=="speed":
        velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,velocity,0)
        speed=float(np.linalg.norm(velocity[3:]));passed=speed<=float(check["max"]);measured={"speed_m_s":speed}
    else:raise ValueError(f"unsupported task check: {kind}")
    return {"check":check,"passed":bool(passed),"measured":measured}


class EventMonitor:
    """Legacy checks are universal; grasp_acquired is contiguous measured dwell.

    Dwell is last_sample_time - first_sample_time, so a single observation has
    zero duration. Source-window retiming never changes the declared SI seconds.
    """
    def __init__(self,event):
        self.event=event
        self.result={"name":event["name"],"samples":0,"passed":True,"first_failure":None,"checks":[]}
        self.starts={}
        for check in event["checks"]:
            acquired=check.get("kind")=="grasp_acquired"
            if acquired:
                dwell=float(check.get("min_dwell_sec",0))
                if not np.isfinite(dwell) or dwell<=0:
                    raise ValueError("grasp_acquired requires finite positive min_dwell_sec")
            self.result["checks"].append({"check":check,"passed":not acquired,"max_contiguous_dwell_sec":0.} if acquired else {"check":check,"passed":True})

    def step(self,model,data,contacts,source_time):
        self.result["samples"]+=1
        self.result["last_sample_time"]=float(data.time)
        for index,(check,result) in enumerate(zip(self.event["checks"],self.result["checks"])):
            acquired=check["kind"]=="grasp_acquired"
            measured=_check(model,data,contacts,{**check,"kind":"grasp"} if acquired else check)
            result["last_measured"]=measured["measured"]
            if acquired:
                if measured["passed"]:
                    self.starts.setdefault(index,float(data.time))
                    dwell=float(data.time)-self.starts[index]
                    result["max_contiguous_dwell_sec"]=max(result["max_contiguous_dwell_sec"],dwell)
                    result["passed"] |= dwell+1e-9>=float(check["min_dwell_sec"])
                else:self.starts.pop(index,None)
            elif not measured["passed"]:
                result["passed"]=False
                if self.result["first_failure"] is None:
                    self.result["first_failure"]={"sim_time":float(data.time),"source_time":source_time,"checks":[measured]}
        self.result["passed"]=all(c["passed"] for c in self.result["checks"])

    def finish(self):
        if not self.result["samples"]:
            self.result["passed"]=False
            self.result["first_failure"]={"reason":"no samples in declared event interval"}
        elif not self.result["passed"] and self.result["first_failure"] is None:
            self.result["first_failure"]={"reason":"required contiguous grasp acquisition dwell not reached","sim_time":self.result["last_sample_time"],"checks":self.result["checks"]}
        return self.result


def _execute(ctx,request,probe=False):
    started=time.perf_counter()
    trajectory=ctx.latest("trajectory")["payload"]; placement_record=ctx.latest("placement"); placement=placement_record["payload"]
    from .execution_reference import validate_trajectory_lineage
    validate_trajectory_lineage(ctx,ctx.latest("trajectory"))
    spec=read_json(ctx.path(request["success_json_path"]))
    if not spec.get("terminal_checks"):raise ValueError("explicit nonempty terminal_checks required; empty checks cannot pass")
    model=mujoco.MjModel.from_xml_path(str(ctx.path(placement["scene_xml"])))
    data=mujoco.MjData(model)
    contract=runtime_contract(model);mobile=contract['motion_mode']=='mobile'
    from .inputs import load_reference
    meta,reference=load_reference(ctx.input_dir)
    if mobile != (meta.get('motion_mode','not_mobile')=='mobile'):raise ValueError('input/bundle motion mode mismatch')
    if model.neq and np.any(model.eq_type==mujoco.mjtEq.mjEQ_WELD):raise ValueError("weld constraints cannot certify a grasp")
    with np.load(ctx.path(placement["initial_state"])) as initial:
        data.qpos[:]=initial["qpos"]; data.qvel[:]=initial["qvel"];data.ctrl[:]=initial["ctrl"]
    mujoco.mj_forward(model,data)
    with np.load(ctx.path(trajectory["trajectory_npz"])) as archive: traj={k:archive[k].copy() for k in archive.files}
    t=traj["time"];source=traj["source_time"];ctrl=traj["ctrl"]
    base_commands=traj.get('base_commands',np.zeros((len(t),2)))
    if base_commands.shape!=(len(t),2) or not np.isfinite(base_commands).all():raise ValueError('invalid base commands')
    if not mobile and np.any(base_commands):raise ValueError('not_mobile cannot execute base commands')
    if mobile and not probe and (abs(source[0]-reference['timestamps'][0])>1e-6 or abs(source[-1]-reference['timestamps'][-1])>1e-6):
        raise ValueError('authoritative mobile rollout must cover the complete source; use stage probe for a prefix')
    settle=float(spec.get("settle_seconds",1.)); fps=float(spec.get("record_hz",30))
    if not np.isfinite([settle,fps]).all() or settle<0 or fps<=0:raise ValueError("invalid settle_seconds/record_hz")
    end=float(t[-1]+settle)
    if probe:
        chosen=next((p for p in trajectory["time_map"] if p["phase"]==request["phase_name"]),None)
        if chosen is None: raise ValueError("unknown phase_name")
        end=float(chosen["sim_end"])
    window=float(spec.get("terminal_window_seconds",.5))
    if not np.isfinite(window) or window<=0 or (not probe and window>end):raise ValueError("terminal_window_seconds must be positive and fit execution")
    next_sample=0.; rows=[]; failures=[];event_results=[]; contacts_log=[]
    terminal_results=[]; terminal_samples=0
    from .force_diagnostics import ContactForceMonitor
    force_monitor=ContactForceMonitor(model)
    from .monitoring import HandMonitor
    from . import checkpoints
    active_phases=[p for p in trajectory["time_map"] if not probe or p["sim_start"]<end-1e-9]
    key=checkpoints.identity(ctx,placement_record,model)
    cached,reuse_count,reuse_reason=checkpoints.load_prefix(ctx,key,model,t,ctrl,end,request.get("reuse_checkpoints",True),base_commands)
    physics_times=[]; physics_states=[]; physics_controls=[];physics_base=[]; step_index=0
    from .contact_policy import FixedContactMonitor
    fixed_contacts = FixedContactMonitor(model.opt.timestep)
    coverage = None
    if not mobile and not probe:
        from transfer.coverage import check_coverage, check_idle_commands
        requirements = ctx.latest("source_requirements")["payload"]
        events = requirements.get("events")
        coverage = check_coverage(events, spec, active_phases)
        actuator_indices = {side: [model.actuator(name).id for name in
                                   [*arm["actuators"], arm["gripper"]["actuator"]]]
                            for side, arm in contract["arms"].items()}
        check_idle_commands(events, active_phases, source, ctrl, actuator_indices)
    collision_counts={}; collision_examples={}
    from .diagnostics import classify_collision
    task_objects={item["object"] for p in trajectory["time_map"] for item in p.get("hand_objects",{}).values() if item.get("object")}
    if any(c.get("kind")=="grasp_acquired" for c in spec["terminal_checks"]):
        raise ValueError("grasp_acquired is an event-window check, not a terminal check")
    terminal_results=[{"check":c,"passed":True,"first_failure":None} for c in spec["terminal_checks"]]
    events=spec.get("events",[])
    if not events or not any(c.get("kind") in ("grasp","grasp_acquired") for event in events for c in event.get("checks",[])):
        raise ValueError("manipulation requires explicit grasp event contact checks; terminal-only success is insufficient")
    event_monitors=[]
    for ev in events:
        if not ev.get("checks"):raise ValueError("event requires checks")
        a,b=float(ev["source_start"]),float(ev["source_end"])
        if not reference["timestamps"][0]<=a<=b<=reference["timestamps"][-1]:raise ValueError("event outside input source interval")
        monitor=EventMonitor(ev);event_monitors.append(monitor);event_results.append(monitor.result)
    monitors={p["phase"]:HandMonitor(model,p,events if end+1e-9>=p["sim_end"] else ()) for p in active_phases}
    base_ids=[] if mobile else [model.body(contract["base"]["body"]).id]
    base_poses={i:data.xpos[i].copy() for i in base_ids}
    max_pen=float(spec.get("max_penetration_m",.008)); min_force=float(spec.get("minimum_contact_force",.01))
    arm_actuators=[model.actuator(n).id for arm in contract["arms"].values() for n in arm["actuators"]]
    max_tracking=0.; first_tracking=None; maximum_penetration=0.; final_checks=[]
    continuous_hold={};previous_phase=None;parked_poses={};max_parked_drift=0.;arrivals=[];checked_arrivals=set();previous_command=np.zeros(2)
    while data.time<=end+model.opt.timestep/2:
        if step_index<reuse_count:
            mujoco.mj_setState(model,data,cached["integration_state"][step_index],checkpoints.STATE)
        simtime=float(data.time)
        state=np.empty(mujoco.mj_stateSize(model,checkpoints.STATE));mujoco.mj_getState(model,data,state,checkpoints.STATE)
        warmstart=data.qacc_warmstart.copy()
        mujoco.mj_forward(model,data)
        data.qacc_warmstart[:]=warmstart
        for i in range(model.nu):data.ctrl[i]=np.interp(simtime,t,ctrl[:,i])
        command=checkpoints.base_command_at(simtime,t,base_commands)
        if mobile:
            from .base_control import validate_command
            validate_command(contract,command,previous_command,float(model.opt.timestep))
            previous_command=command.copy()
        physics_base.append(command.copy())
        st=float(np.interp(simtime,t,source))
        physics_times.append(simtime);physics_states.append(state);physics_controls.append(data.ctrl.copy())
        phase=next((p for p in active_phases if p["sim_start"]-1e-9<=simtime<p["sim_end"]-1e-9),None)
        if phase is None and abs(simtime-active_phases[-1]["sim_end"])<1e-8:phase=active_phases[-1]
        if mobile and simtime>=active_phases[-1]["sim_end"]:phase=active_phases[-1]
        force_monitor.step(data,st,phase)
        force_contacts = _contacts(model, data, 0.)
        contacts = [c for c in force_contacts if c[3] >= min_force]
        if phase:
            monitor=monitors[phase['phase']]
            if mobile and previous_phase!=phase['phase']:
                for side in ('left','right'):
                    item=phase['hand_objects'].get(side,{})
                    prior=continuous_hold.get(side)
                    if item.get('mode') not in ('grasp','hold') or (prior and prior['object']!=item.get('object')):continuous_hold.pop(side,None)
                    elif prior:
                        monitor.reference[side]=prior['reference'];monitor.gap[side]=prior['gap']
                previous_phase=phase['phase']
            monitor.step(model,data,contacts,_check)
            if mobile:
                for side,item in phase['hand_objects'].items():
                    if item['mode'] in ('grasp','hold'):
                        continuous_hold[side]={'object':item['object'],'reference':monitor.reference[side],'gap':monitor.gap[side]}
                if phase.get('kind')!='transit':
                    anchor=parked_poses.setdefault(phase['phase'],base_pose(model,data).copy())
                    drift=float(np.max(abs(base_pose(model,data)-anchor)));max_parked_drift=max(max_parked_drift,drift)
                    if (np.any(abs(command)>1e-10) or drift>1e-6) and not any(f['kind']=='parked_base_motion' for f in failures):failures.append({'kind':'parked_base_motion','sim_time':simtime})

        collision_contacts = [c for c in force_contacts if c[3] > 0.]
        if mobile:
            collision_contacts=[(int(c.geom1),int(c.geom2),float(c.dist),0.) for c in data.contact[:data.ncon] if c.dist<=0]
        unintended_forces = {}
        for a,b,d,f in collision_contacts:
            classification=classify_collision(model,a,b,task_objects)
            names=classification["geoms"];category=classification["category"]
            if phase:
                intended=False
                for side,item in phase.get("hand_objects",{}).items():
                    if item.get("object"):
                        object_id=_body(model,item["object"])
                        intended |= item["mode"] in (("grasp","hold","contact","release") if mobile else ("grasp","hold","contact")) and any(_descendant(model,int(model.geom_bodyid[(a,b)[k]]),model.body(contract["arms"][side]["eef_body"]).id) and _descendant(model,int(model.geom_bodyid[(a,b)[1-k]]),object_id) for k in (0,1))
                if intended and (mobile or classification["category"] != "payload_robot_structure"
                                 or tuple(sorted((a,b))) in monitor.permitted_pairs):
                    category="intended_hand_object_contact"
            if not mobile and classification["category"] in ("robot_self_collision", "robot_environment_contact", "payload_robot_structure"):
                if not phase or tuple(sorted((a,b))) not in monitor.permitted_pairs:
                    pair = tuple(sorted(model.body(i).name or f"body_{i}" for i in model.geom_bodyid[[a,b]]))
                    unintended_forces[pair] = unintended_forces.get(pair, 0.) + f
            if mobile:
                held_names={item['object'] for item in (phase or {}).get('hand_objects',{}).values() if item.get('mode')=='hold'}
                transit_category=classify_collision(model,a,b,held_names)['category']
                forbidden=category in ('robot_self_collision','robot_environment_contact','payload_robot_structure','payload_finger_contact') or (phase and phase.get('kind')=='transit' and transit_category=='payload_environment_contact')
                if forbidden and not any(f.get('kind')=='mobile_collision' and f.get('category')==category for f in failures):
                    failures.append({'kind':'mobile_collision','category':category,'sim_time':simtime,'geoms':names})
            collision_counts[category]=collision_counts.get(category,0)+1
            collision_examples.setdefault(category,{"sim_time":simtime,"geoms":names,"distance":d})
        if not mobile:
            fixed_contacts.step(simtime, unintended_forces)
        if mobile:
            for arrival in active_phases:
                if arrival.get('kind')!='transit' or arrival['phase'] in checked_arrivals or simtime<arrival['sim_end']-1e-8:continue
                checked_arrivals.add(arrival['phase'])
                error=base_pose(model,data)-np.asarray(arrival['goal_xyyaw']);error[2]=(error[2]+np.pi)%(2*np.pi)-np.pi
                velocity=np.array([data.qvel[model.jnt_dofadr[model.joint(n).id]] for n in contract['base']['joints']])
                tail=[c for time,c in zip(physics_times,physics_base) if arrival['sim_end']-.1-1e-8<=time<=arrival['sim_end']+1e-8]
                stopped=bool(tail) and np.max(np.abs(tail))<1e-8 and np.max(abs(velocity))<1e-6
                passed=np.linalg.norm(error[:2])<=.02 and abs(error[2])<=.03 and stopped
                arrivals.append({'phase':arrival['phase'],'sim_time':simtime,'error_xyyaw':error.tolist(),'stopped_last_0_1s':stopped,'passed':bool(passed)})
                if not passed:failures.append({'kind':'arrival_error','sim_time':simtime,'arrival':arrivals[-1]})
        if not probe and simtime>=end-window-1e-9:
            terminal_samples+=1
            for result,c in zip(terminal_results,spec["terminal_checks"]):
                measured=_check(model,data,contacts,c)
                result["last_measured"]=measured["measured"]
                result["last_sample_time"]=simtime
                if not measured["passed"]:
                    result["passed"]=False
                    if result["first_failure"] is None:result["first_failure"]={"sim_time":simtime,**measured}
        for ei,event in enumerate(events):
            if event["source_start"]-1e-9<=st<=event["source_end"]+1e-9 and simtime<=t[-1]+1e-8:
                event_monitors[ei].step(model,data,contacts,st)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            failures.append({"kind":"nonfinite_state","sim_time":simtime});break
        penetration=max([max(0.,-float(data.contact[c].dist)) for c in range(data.ncon)]+[0.])
        maximum_penetration=max(maximum_penetration,penetration)
        if penetration>max_pen and not any(f["kind"]=="penetration" for f in failures):failures.append({"kind":"penetration","sim_time":simtime,"source_time":st,"depth_m":penetration})
        for i,position in base_poses.items():
            if np.linalg.norm(data.xpos[i]-position)>1e-8:failures.append({"kind":"base_motion","sim_time":simtime})
        if simtime+1e-9>=next_sample:
            error=[]
            for i in arm_actuators:
                j=int(model.actuator_trnid[i,0])
                if j>=0:error.append(abs(data.qpos[model.jnt_qposadr[j]]-data.ctrl[i]))
            current_tracking=max(error,default=0.)
            max_tracking=max(max_tracking,current_tracking)
            if current_tracking>float(spec.get("max_tracking_error_rad",.5)) and first_tracking is None:first_tracking=simtime
            rows.append((simtime,data.qpos.copy(),data.qvel.copy(),data.ctrl.copy(),st))
            contacts_log.append({"sim_time":simtime,"contacts":[{"geoms":[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,a) or str(a),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,b) or str(b)],"distance":d,"normal_force":f} for a,b,d,f in contacts]})
            next_sample+=1/fps
        if simtime>=end:break
        step_index+=1
        if step_index>=reuse_count:
            if mobile:
                from .base_control import step_base
                step_base(model,data,command)
            else:mujoco.mj_step(model,data)
    failures.extend(fixed_contacts.failures)
    for monitor in monitors.values():
        if any(m["steps"] for m in monitor.metrics.values()):failures.extend(monitor.failures())
    for monitor in event_monitors:
        result=monitor.finish()
        if probe:
            event=next(e for e in events if e["name"]==result["name"])
            executed_source_end=float(np.interp(end,t,source))
            if event["source_start"]>executed_source_end:continue
            if event["source_end"]>executed_source_end+1e-9 and any(c.get("kind")=="grasp_acquired" for c in event["checks"]):
                result["passed"]=False
                result["first_failure"]={"reason":"acquisition event window only partially executed by probe","sim_time":end}
        if not result["samples"]:result["passed"]=False;result["first_failure"]={"reason":"no samples in declared event interval"}
        if not result["passed"]:failures.append({"kind":"event","event":result["name"],"detail":result["first_failure"]})
    final_checks=terminal_results
    if not probe and (not terminal_samples or not all(x["passed"] for x in final_checks)):
        first=min([r["first_failure"]["sim_time"] for r in final_checks if r["first_failure"]]+[end])
        failures.append({"kind":"terminal_checks","sim_time":first,"checks":final_checks})
    if max_tracking>float(spec.get("max_tracking_error_rad",.5)):failures.append({"kind":"tracking","sim_time":first_tracking,"max_error_rad":max_tracking})
    for failure in failures:
        ft=failure.get("sim_time",(failure.get("detail") or {}).get("sim_time",end))
        failure["sim_time"]=ft
        failure["phase"]=next((p["phase"] for p in trajectory["time_map"] if p["sim_start"]<=ft<=p["sim_end"]),"terminal_settle")
    failures.sort(key=lambda f:f["sim_time"])
    directory=Path("probes" if probe else "rollouts")/uuid.uuid4().hex; statepath=ctx.path(directory/"states.npz")
    np.savez_compressed(statepath,time=np.asarray([r[0] for r in rows]),qpos=np.asarray([r[1] for r in rows]),qvel=np.asarray([r[2] for r in rows]),ctrl=np.asarray([r[3] for r in rows]),source_time=np.asarray([r[4] for r in rows]))
    diag={"status":("PROBE_OK" if not failures else "PROBE_FAIL") if probe else ("PASS" if not failures else "FAIL"),"failures":failures,"first_failure":failures[0] if failures else None,"events":event_results,"terminal_checks":final_checks,"maximum_penetration_m":maximum_penetration,"max_tracking_error_rad":max_tracking,"thresholds":spec,"contact_history":contacts_log,"hand_object_monitoring":{name:m.metrics for name,m in monitors.items()},"terminal_window_seconds":window,"terminal_window_samples":terminal_samples,"collision_classes":collision_counts,"collision_examples":collision_examples,"checkpoint":{"reused_steps":max(0,reuse_count-1),"restored_time":float(cached["time"][reuse_count-1]) if reuse_count else 0.,"reason":reuse_reason,"checks_reevaluated":True},"authoritative":not probe,"limitations":["Support check measures contact; visual reviewer must verify intended load-bearing relation.","Contact/ownership/events evaluated every physics step; solver contacts remain model-dependent."]}
    diag["interaction_coverage"] = coverage
    diag["mobility"]={"motion_mode":contract["motion_mode"],"base_physics":contract["base"].get("physics"),"max_parked_drift":max_parked_drift,"final_xyyaw":base_pose(model,data).tolist(),"arrivals":arrivals}
    diag["contact_force_summary"]=force_monitor.finish()
    diagpath=ctx.path(directory/"diagnostics.json");write_json(diagpath,diag)
    specpath=ctx.path(directory/"success_spec.json");write_json(specpath,spec)
    checkpoint_path=ctx.path(directory/"physics_steps.npz")
    checkpoints.save(ctx,key,checkpoint_path,physics_times,physics_states,physics_controls,physics_base)
    physics_evaluation_seconds=time.perf_counter()-started
    render_started=time.perf_counter(); evidence={}; media_error=None
    try:
        from .evidence import build_rollout_bundle
        evidence=build_rollout_bundle(ctx,placement,trajectory,statepath,diag,directory/"evidence",request.get("phase_name") if probe else None)
    except Exception as exc:
        media_error=f"{type(exc).__name__}: {exc}";diag["status"]="PROBE_FAIL" if probe else "FAIL"
        diag["failures"].append({"kind":"evidence","error":media_error});diag["first_failure"]=diag["failures"][0]
    diag["timing"]={"physics_and_evaluation_seconds":physics_evaluation_seconds,"evidence_render_seconds":time.perf_counter()-render_started}
    write_json(diagpath,diag)
    compact_evidence={k:evidence[k] for k in ("bundle_json","keyframes","settings","summary") if k in evidence}
    payload={"failure_evidence":compact_evidence if failures else {},"rollout_bundle":compact_evidence,"status":diag["status"],"states_npz":str(statepath.relative_to(ctx.root)),"diagnostics_json":str(diagpath.relative_to(ctx.root)),"spec_json":str(specpath.relative_to(ctx.root)),"videos":{},"media_error":media_error,"authoritative":not probe,"checkpoint":diag["checkpoint"],"physics_steps_npz":str(checkpoint_path.relative_to(ctx.root)),"timing":diag["timing"],"compact_diagnostics":{"first_failure":diag["first_failure"],"collision_classes":collision_counts,"hand_objects":diag["hand_object_monitoring"],"contact_force_summary":diag["contact_force_summary"]}}
    files=[statepath,diagpath,specpath,checkpoint_path]
    if evidence:files.extend([ctx.path(evidence["bundle_json"])]+[ctx.path(p) for p in evidence["keyframes"]])
    return ctx.put("stage_probe" if probe else "rollout",payload,dependencies=("trajectory","placement") if probe or mobile else ("trajectory","placement","source_requirements"),files=files)


def simulate_task_rollout(ctx,request):
    return _execute(ctx,request)


def probe_human_stage(ctx,request):
    """Execute the prefix ending at phase_name; never produces a rollout PASS."""
    return _execute(ctx,request,probe=True)
