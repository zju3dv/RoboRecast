"""Replay a DROID candidate and evaluate contact, task state and passive motion."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from roborecast import checkpoints, requirements
from roborecast.harness import compile_motion, load_candidate, preflight, read
from roborecast.motion_validity import default_motion_policy, velocity_report
from roborecast.rollout import _execute


def passive(model, roots):
    bodies = {model.body(name).id for name in roots}
    for body, parent in enumerate(model.body_parentid):
        if parent in bodies:
            bodies.add(body)
    # The only allowed actuated joints are on the robot. Also exclude mocap
    # and constraints linking a manipulated body to a different body.
    joints = {j for j, body in enumerate(model.jnt_bodyid) if body in bodies}
    for i in range(model.nu):
        kind = model.actuator_trntype[i]
        target = model.actuator_trnid[i, 0]
        if kind in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT) and target in joints:
            return False
        if kind in (mujoco.mjtTrn.mjTRN_SITE, mujoco.mjtTrn.mjTRN_SLIDERCRANK) and model.site_bodyid[target] in bodies:
            return False
        if kind == mujoco.mjtTrn.mjTRN_BODY and target in bodies:
            return False
        if kind == mujoco.mjtTrn.mjTRN_TENDON:
            for k in range(model.tendon_adr[target], model.tendon_adr[target] + model.tendon_num[target]):
                obj = model.wrap_objid[k]
                if model.wrap_type[k] == mujoco.mjtWrap.mjWRAP_JOINT and obj in joints:
                    return False
                if model.wrap_type[k] == mujoco.mjtWrap.mjWRAP_SITE and model.site_bodyid[obj] in bodies:
                    return False
    for i in range(model.neq):
        if model.eq_type[i] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
            if model.eq_obj1id[i] in bodies or model.eq_obj2id[i] in bodies:
                return False
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and (model.eq_obj1id[i] in joints or model.eq_obj2id[i] in joints):
            return False
    return all(model.body_mocapid[b] < 0 for b in bodies)


def replay(bundle, out, previous=None, save_prefixes=False):
    out.mkdir(parents=True, exist_ok=True)
    model, points = load_candidate(bundle, {})
    runtime = read(bundle / "runtime.json")
    contract = requirements.validate(read(bundle / "requirements.json"))
    physics = read(bundle / "success.json")
    physics["checks"] = physics["required_checks"] + physics["diagnostic_checks"]
    observations = read(bundle / "observables.json")
    if not observations["passive_roots"]:
        raise ValueError("Declare the manipulated passive body or articulated unit")
    trajectory = compile_motion(model, runtime, points)
    windows = requirements.associate(points, contract)
    validity = preflight(model, runtime, trajectory)
    result = {"status": "FAIL", "motion": validity, "windows": windows, "rollout_executed": False}
    if validity["status"] == "PASS":
        prefix = None
        if previous:
            prior = read(previous / "report.json")
            if prior.get("physics_spec") == physics and prior.get("observables") == observations:
                prefix = checkpoints.select_prefix(previous, prior.get("checkpoints", []), model, runtime, trajectory, contract, windows)
        speeds = observations.get("tail_speeds", {})
        dense, physical, _ = _execute(model, runtime, trajectory, physics, requirements=contract,
                                      resume=prefix, additional_body_positions=tuple(set(speeds.values())))
        t = dense["timestamp_sec"]
        measurements = {"physical:finite": float(all(np.isfinite(dense[k]).all() for k in ("qpos", "qvel", "ctrl"))),
                        "physical:max_penetration_m": float(np.max(dense["penetration_m"])),
                        "physical:no_object_drivers": float(passive(model, observations["passive_roots"]))}
        for key, body in speeds.items():
            velocity = np.gradient(dense[f"body:{body}:position"], t, axis=0)
            measurements[key] = float(np.linalg.norm(velocity[t >= t[-1]-.25], axis=1).max())
        for key, value in measurements.items():
            dense[key] = np.full(len(t), value)
        events = requirements.evaluate(contract, windows, dense)
        limits = default_motion_policy(runtime).get("velocity_limits_rad_s")
        measured_velocity = velocity_report(t, dense["arm_joint_velocity"], runtime["arm_joint_names"], limits)
        passed = all(report["status"] == "PASS" for report in (events, physical, measured_velocity)) and measurements["physical:no_object_drivers"] == 1
        result.update(status="PASS" if passed else "FAIL", rollout_executed=True, physics=physical,
                      interactions=events, observables=observations, physics_spec=physics, physical=measurements,
                      velocity=measured_velocity, steps=len(t), reused_events=prefix["event_ids"] if prefix else [])
        if save_prefixes:
            result["checkpoints"] = checkpoints.save_prefixes(out, out/"checkpoints", 1, model, runtime, trajectory, contract, windows, dense, physics)
        np.savez_compressed(out / "rollout.npz", **dense)
    else:
        (out / "rollout.npz").unlink(missing_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--save-prefixes", action="store_true")
    a = p.parse_args()
    report = replay(a.bundle, a.out, a.resume, a.save_prefixes)
    print(report["status"], a.out / "report.json")
    raise SystemExit(0 if report["status"] == "PASS" else 1)
