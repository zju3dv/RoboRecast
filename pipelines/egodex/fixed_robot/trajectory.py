"""Compile sparse world-space human-guided phases into frozen robot controls.

Grippers are normalized: 0 closed, 1 open. Source times always refer to the
original recording; retiming changes execution time only. This is waypoint IK,
not a global collision-free path planner; physical rollout is required.
"""
from pathlib import Path
import uuid
import copy
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from .utils import read_json, write_json
from .inputs import load_reference
from .robot import solve_ik
from .robot_runtime import runtime_contract
from .targets import resolve_target, validate_hand_objects


def _controls(model, qpos, previous, grippers):
    ctrl = previous.copy()
    arms = runtime_contract(model)['arms']
    if not set(grippers).issubset(arms):
        raise ValueError('gripper command names must match runtime arm roles')
    for side, arm in arms.items():
        for joint_name, actuator_name in zip(arm['joints'], arm['actuators']):
            ctrl[model.actuator(actuator_name).id] = qpos[model.jnt_qposadr[model.joint(joint_name).id]]
        if side in grippers:
            value = float(grippers[side])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('gripper commands must be finite and in [0,1]')
            gripper = arm['gripper']
            lo, hi = gripper['control_range']
            ctrl[model.actuator(gripper['actuator']).id] = lo + value * (hi-lo)
    if not np.isfinite(ctrl).all():
        raise ValueError("nonfinite actuator targets")
    limited = model.actuator_ctrllimited.astype(bool)
    if np.any(ctrl[limited] < model.actuator_ctrlrange[limited,0]-1e-8) or np.any(ctrl[limited] > model.actuator_ctrlrange[limited,1]+1e-8):
        raise ValueError("actuator target outside control range")
    return ctrl


def _target_mix(a, b, u):
    result = {}
    for side in set(a) | set(b):
        x, y = a.get(side, b.get(side)), b.get(side, a.get(side))
        p = (1-u)*np.asarray(x["position"],float) + u*np.asarray(y["position"],float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Cartesian position must contain three finite metres")
        result[side] = {"position":p.tolist()}
        if "quaternion_wxyz" in x or "quaternion_wxyz" in y:
            qa=np.asarray(x.get("quaternion_wxyz",y.get("quaternion_wxyz")),float)
            qb=np.asarray(y.get("quaternion_wxyz",x.get("quaternion_wxyz")),float)
            r=Rotation.from_quat(np.stack([qa[[1,2,3,0]],qb[[1,2,3,0]]]))
            q=Slerp([0,1],r)([u]).as_quat()[0]
            result[side]["quaternion_wxyz"]=q[[3,0,1,2]].tolist()
    if not set(result) <= {"left","right"}:
        raise ValueError("targets may only name left/right arms; base motion unsupported")
    return result


def _compile_arm_trajectory(ctx, request):
    placement=ctx.latest("placement")["payload"]
    spec=read_json(ctx.path(request["trajectory_json_path"]))
    if spec.get("base_mode","fixed") != "fixed" or any(k in spec for k in ("base_trajectory","base_commands","base_targets")):
        raise ValueError("v1.2 requires a fixed base")
    held=spec.get("hold_committed_arms",[])
    if not isinstance(held,list) or any(s not in ("left","right") for s in held) or len(set(held))!=len(held):
        raise ValueError("hold_committed_arms must be a unique list of left/right")
    meta,ref=load_reference(ctx.input_dir)
    ts=np.asarray(ref["timestamps"])
    phases=spec.get("phases",[])
    if not phases: raise ValueError("at least one sparse phase required")
    names=[phase.get("name") for phase in phases]
    if any(not isinstance(name,str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("phase names must be nonempty and unique for unambiguous probes")
    model=mujoco.MjModel.from_xml_path(str(ctx.path(placement["scene_xml"])))
    data=mujoco.MjData(model)
    with np.load(ctx.path(placement["initial_state"])) as init:
        data.qpos[:]=init["qpos"]; data.ctrl[:]=init["ctrl"]
    committed_ctrl=data.ctrl.copy()
    arms=runtime_contract(model)['arms']
    held_actuators=[model.actuator(name).id for side in held for name in [*arms[side]['actuators'],arms[side]['gripper']['actuator']]]
    gripper_ids={model.actuator(arm['gripper']['actuator']).id for arm in arms.values()}
    mujoco.mj_forward(model,data)
    # Resolve all frames before any IK mutates data, retaining original references.
    resolved_spec=copy.deepcopy(spec)
    for phase in resolved_spec["phases"]:
        validate_hand_objects(model,phase)
        if any(phase["hand_objects"][side]["mode"]!="idle" for side in held):
            raise ValueError("hold_committed_arms requires idle hand_objects bindings")
        for waypoint in phase.get("waypoints",[]):
            targets=waypoint.get("targets",{})
            if not isinstance(targets,dict) or not set(targets)<= {"left","right"}:
                raise ValueError("targets must map left/right to explicit Cartesian targets")
            if set(held)&(set(targets)|set(waypoint.get("grippers",{}))):
                raise ValueError("held arms cannot have Cartesian targets or gripper commands")
            waypoint["targets"]={side:resolve_target(model,data,value) for side,value in targets.items()}
    phases=resolved_spec["phases"]
    hz=float(spec.get("sample_hz",30)); vmax=float(spec.get("joint_velocity_limit",1.5)); amax=float(spec.get("joint_acceleration_limit",5.0))
    if not all(np.isfinite(x) and x>0 for x in (hz,vmax,amax)): raise ValueError("positive sampling and retiming limits required")
    times=[]; controls=[]; qs=[]; sources=[]; indices=[]; maps=[]; diagnostics=[]
    elapsed=0.; last_source=float(ts[0]); previous_ctrl=data.ctrl.copy()
    arm_q=[int(model.jnt_qposadr[model.joint(name).id]) for arm in arms.values() for name in arm['joints']]
    for pi,phase in enumerate(phases):
        start,end=float(phase["source_start"]),float(phase["source_end"])
        duration=float(phase["duration"])
        if not np.isfinite([start,end,duration]).all() or start < last_source-1e-7 or end < start or start < ts[0]-1e-7 or end > ts[-1]+1e-7 or duration<=0:
            raise ValueError("phase times must be finite, ordered and within original source range")
        if not isinstance(phase.get("roles"),dict) or set(phase["roles"]) != {"left","right"}:
            raise ValueError("each phase must explicitly describe left and right roles, including idle")
        wp=phase.get("waypoints",[])
        wt=np.asarray([w["t"] for w in wp],float)
        if len(wp)<2 or not np.isfinite(wt).all() or abs(wt[0])>1e-8 or abs(wt[-1]-duration)>1e-8 or np.any(np.diff(wt)<=0):
            raise ValueError("waypoints require strictly increasing times from 0 through duration")
        if any(any(k.startswith("base") for k in w) for w in wp): raise ValueError("base waypoint commands unsupported")
        # Include every knot so a sparse contact event cannot disappear between samples.
        grid=np.linspace(0,duration,max(2,int(np.ceil(duration*hz))+1))
        # Keep exact knots, dropping coincident grid samples to avoid spurious accelerations.
        interior=grid[np.min(np.abs(grid[:,None]-wt[None,:]),axis=1)>1e-9]
        local=np.sort(np.r_[wt,interior])
        pq=[]; pc=[]; boundary_ctrl=previous_ctrl.copy()
        for t in local:
            j=min(max(int(np.searchsorted(wt,t,side="right")-1),0),len(wp)-2)
            u=float((t-wt[j])/(wt[j+1]-wt[j])); u=float(np.clip(u*u*u*(10+u*(-15+6*u)),0.,1.))
            targets=_target_mix(wp[j].get("targets",{}),wp[j+1].get("targets",{}),u)
            if not targets and not held: raise ValueError("waypoints must contain an arm target or explicit committed hold")
            report=solve_ik(model,data,targets) if targets else {"success":True}
            if not report.get("success",False): raise ValueError(f"IK failed in phase {phase['name']} at {t:.3f}s: {report}")
            ga,gb=wp[j].get("grippers",{}),wp[j+1].get("grippers",{})
            grippers={s:(1-u)*float(ga.get(s,gb[s]))+u*float(gb.get(s,ga[s])) for s in set(ga)&set(gb)}
            for s in set(ga)^set(gb): grippers[s]=ga.get(s,gb.get(s))
            previous_ctrl=_controls(model,data.qpos,previous_ctrl,grippers)
            previous_ctrl[held_actuators]=committed_ctrl[held_actuators]
            if t==0:
                tolerances=np.array([.003 if i in gripper_ids else .03 for i in range(model.nu)])
                if np.any(np.abs(previous_ctrl-boundary_ctrl)>tolerances):
                    raise ValueError(f"phase {phase['name']} starts with a control discontinuity; include an explicit timed approach from the committed/preceding pose")
                previous_ctrl=boundary_ctrl.copy()
            pq.append(data.qpos.copy()); pc.append(previous_ctrl.copy())
        pq=np.asarray(pq); pc=np.asarray(pc)
        vel=np.gradient(pq[:,arm_q],local,axis=0); acc=np.gradient(vel,local,axis=0)
        factor=max(1.,float(np.max(np.abs(vel)))/vmax,float(np.sqrt(np.max(np.abs(acc))/amax)))
        # Explicit stationary endpoint smoothing is provided by quintic knot interpolation.
        actual=duration*factor
        for k,t in enumerate(local):
            if pi and k==0: continue
            times.append(elapsed+t*factor); controls.append(pc[k]); qs.append(pq[k]); sources.append(start+(end-start)*t/duration); indices.append(pi)
        ev=[]
        for event in phase.get("events",[]):
            st=float(event["source_time"])
            if not start<=st<=end: raise ValueError("phase event outside source interval")
            ev.append({**event,"sim_time":elapsed+(0 if end==start else (st-start)/(end-start))*actual})
        maps.append({"phase":phase["name"],"roles":phase["roles"],"hand_objects":phase["hand_objects"],"source_start":start,"source_end":end,"sim_start":elapsed,"sim_end":elapsed+actual,"retime_factor":factor,"events":ev})
        diagnostics.append({"phase":phase["name"],"retime_factor":factor,"waypoint_ik_only":True})
        elapsed+=actual; last_source=end
    run=Path("trajectories")/uuid.uuid4().hex
    out=ctx.path(run/"trajectory.npz")
    np.savez_compressed(out,time=times,ctrl=controls,qpos_target=qs,source_time=sources,phase_index=indices)
    specpath=ctx.path(run/"spec.json"); write_json(specpath,spec)
    resolvedpath=ctx.path(run/"resolved_spec.json"); write_json(resolvedpath,resolved_spec)
    payload={"hold_committed_arms":held,"target_reference_state":"committed_initial_state","resolved_spec_json":str(resolvedpath.relative_to(ctx.root)),"trajectory_npz":str(out.relative_to(ctx.root)),"spec_json":str(specpath.relative_to(ctx.root)),"time_map":maps,"diagnostics":diagnostics,"base_mode":"fixed"}
    return ctx.put("trajectory",payload,dependencies=("placement",),files=(out,specpath,resolvedpath))


def compile_human_guided_trajectory(ctx, request):
    meta,_=load_reference(ctx.input_dir)
    if meta.get("motion_mode","not_mobile")=="mobile":
        raise ValueError("EgoDex requires a fixed-base target")
    if request.get("reference_record_id"):
        raise ValueError("checkpoint-based suffix compilation is a mobile v2 operation")
    return _compile_arm_trajectory(ctx,request)
