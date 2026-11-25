"""Compile a semantic program, execute it in MuJoCo and evaluate interaction requirements."""
import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import checkpoints, motion, requirements
from .motion_validity import default_motion_policy, velocity_report
from .robot_runtime_view import RobotRuntimeView
from .rollout import _execute


def read(path):
    return json.loads(Path(path).read_text())


def initial_state(model):
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "settled_initial")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return data


def load_scene(path, edit):
    if "world_collision_box" not in edit:
        return mujoco.MjModel.from_xml_path(str(path))
    spec = mujoco.MjSpec.from_file(str(path))
    box = edit["world_collision_box"]
    spec.worldbody.add_geom(name=box["name"], type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=box["center_m"], size=box["half_extents_m"],
                            group=3, rgba=[1,.2,.1,.65])
    return spec.compile()


def load_candidate(bundle, edit):
    model = load_scene(bundle / "scene.xml", edit)
    points = read(bundle / "program.json")["request"]["waypoints"]
    for patch in edit.get("waypoint_offsets", []):
        names = set(patch["names"])
        if names - {point["name"] for point in points}:
            raise ValueError("Unknown waypoint in edit")
        for point in points:
            if point["name"] in names:
                point["position_m"] = (np.array(point["position_m"]) + patch["offset_m"]).tolist()
    return model, points


@dataclass
class Compiled:
    timestamp_sec: np.ndarray
    joint_position: np.ndarray
    gripper_width_m: np.ndarray
    ik_error_m_rad: np.ndarray
    desired_xyz: np.ndarray
    desired_quaternions: np.ndarray


def compile_motion(model, runtime, points):
    view = RobotRuntimeView.from_contract(model, runtime)
    data = initial_state(model)
    times, positions, orientations, widths = [], [], [], []
    for i, point in enumerate(points):
        pos = np.array(point["position_m"], dtype=float)
        rot = Rotation.from_quat(point["quaternion_xyzw"])
        frame = point.get("reference_frame", {"type": "world"})
        if frame["type"] != "world":
            if frame["type"] == "body":
                target = data.body(frame["name"])
            elif frame["type"] == "site":
                target = data.site(frame["name"])
            else:
                raise ValueError("Unknown reference frame")
            rotation = Rotation.from_matrix(target.xmat.reshape(3, 3))
            pos = target.xpos + rotation.apply(pos)
            rot = rotation * rot
        if point["motion_mode"] not in ("dwell", "transit", "sustained_contact"):
            raise ValueError("Unknown motion mode")
        if i and point["motion_mode"] == "dwell" and (
                np.linalg.norm(pos - positions[-1]) > 1e-6 or
                (rot.inv() * Rotation.from_quat(orientations[-1])).magnitude() > 1e-6):
            raise ValueError("A dwell must preserve the previous pose")
        width = float(point["gripper_width_m"])
        if not 0 <= width <= runtime["gripper"]["maximum_width_m"]:
            raise ValueError("Width outside the gripper range")
        duration = float(point["duration_sec"])
        if i and (not np.isfinite(duration) or duration <= 0):
            raise ValueError("Segment duration must be positive")
        times.append(times[-1] + duration if i else 0.)
        positions.append(pos)
        orientations.append(rot.as_quat())
        widths.append(width)
    grid = motion._target_timestamps(0., times[-1])
    times = np.asarray(times)
    xyz, quat = motion.sample_cartesian(times, np.array(positions), np.array(orientations), grid, "quintic")
    q, error = motion._solve_ik(model, data.qpos.copy(), xyz, quat, view)
    width = motion._minimum_jerk(np.array(widths)[:, None], times, grid)[:, 0]
    return Compiled(grid, q, width, error, xyz, quat)


def preflight(model, runtime, trajectory):
    view = RobotRuntimeView.from_contract(model, runtime)
    data = initial_state(model)
    residuals = []
    for q, xyz, quat in zip(trajectory.joint_position, trajectory.desired_xyz, trajectory.desired_quaternions):
        data.qpos[list(view.arm_qpos_addresses)] = q
        mujoco.mj_forward(model, data)
        ee = data.body(view.end_effector_body_id)
        rot = Rotation.from_matrix(ee.xmat.reshape(3, 3))
        residuals.append([np.linalg.norm(ee.xpos-xyz), (rot.inv()*Rotation.from_quat(quat)).magnitude()])
    pos, angle = np.array(residuals).T
    joints = np.asarray(view.arm_joint_ids)
    bounded = np.asarray(model.jnt_limited[joints], dtype=bool)
    limits = model.jnt_range[joints[bounded]]
    q = trajectory.joint_position[:,bounded]
    inside = bool(np.all((q >= limits[:,0]) & (q <= limits[:,1])))
    velocity = velocity_report(trajectory.timestamp_sec, np.gradient(trajectory.joint_position,
        trajectory.timestamp_sec, axis=0), list(view.arm_joint_names),
        default_motion_policy(runtime).get("velocity_limits_rad_s"))
    pose_error = float(np.hypot(pos, .35 * angle).max())
    failed = []
    if not inside:
        failed.append("joint_limits")
    if not pose_error <= .002:
        failed.append("pose_error")
    if velocity["status"] != "PASS":
        failed.append("target_velocity")
    return dict(status="FAIL" if failed else "PASS", failed_checks=failed,
                max_position_error_m=float(pos.max()), max_angle_error_rad=float(angle.max()),
                inside_joint_limits=inside, target_velocity=velocity)


def write_report(output, result):
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], "output:", output)
    for category, checks in result["failed_checks"].items():
        print(f"  {category}: {', '.join(checks)}")
    earliest = result["interactions"]["earliest_unmet"]
    if earliest is not None:
        print("  earliest unmet interaction:", earliest)
    return result


def run(bundle, edit, output, resume_dir=None, save_prefix=False):
    model, points = load_candidate(bundle, edit)
    runtime = read(bundle / "runtime.json")
    contract = requirements.validate(read(bundle / "requirements.json"))
    success = read(bundle / "success.json")
    success["checks"] = success["required_checks"] + success["diagnostic_checks"]
    trajectory = compile_motion(model, runtime, points)
    intervals = requirements.associate(points, contract)
    validity = preflight(model, runtime, trajectory)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status=validity["status"], failed_checks={}, rollout_executed=False,
                  motion=validity, windows=intervals, max_ik_residual=float(trajectory.ik_error_m_rad.max()),
                  physical_spec=success, reused_steps=0, reused_events=[])
    if validity["status"] != "PASS":
        result.update(failed_checks={"motion_preflight": validity["failed_checks"]},
                      interactions={"status": "NOT_RUN", "earliest_unmet": None},
                      physics={"status": "NOT_RUN"})
        # A repeated run can reuse an output directory. Retire any older trajectory.
        (output / "rollout.npz").unlink(missing_ok=True)
        return write_report(output, result)

    resume = None
    if resume_dir is not None:
        prior = read(resume_dir / "report.json")
        if prior.get("physical_spec") == success and prior.get("motion", {}).get("status") == "PASS":
            resume = checkpoints.select_prefix(resume_dir, prior.get("checkpoints", []),
                                                model, runtime, trajectory, contract, intervals)
    dense, physics, _ = _execute(model, runtime, trajectory, success, requirements=contract, resume=resume)
    report = requirements.evaluate(contract, intervals, dense)
    failed_checks = {
        "initial_conditions": [p["key"] for p in report["constraints"]["G"] if p["status"] != "PASS"],
        "interactions": [e["id"] for e in report["events"] if e["status"] != "PASS"],
        "rollout_constraints": [p["key"] for p in report["constraints"]["c"] if p["status"] != "PASS"],
        "physics": [c["type"] for c in physics["checks"] if c["role"] == "required" and not c["passed"]],
    }
    result.update(status="PASS" if report["status"] == physics["status"] == "PASS" else "FAIL",
                  rollout_executed=True, interactions=report, physics=physics,
                  failed_checks={name: checks for name, checks in failed_checks.items() if checks},
                  reused_steps=int(dense["reused_step_count"]), reused_events=resume["event_ids"] if resume else [])
    if save_prefix:
        result["checkpoints"] = checkpoints.save_prefixes(output, output/"checkpoints", 1, model,
            runtime, trajectory, contract, intervals, dense, success)
    np.savez_compressed(output / "rollout.npz", **dense)
    return write_report(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--edit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-prefixes", action="store_true")
    parser.add_argument("--resume", type=Path, help="Directory of an earlier run with saved prefixes")
    args = parser.parse_args()
    run(args.bundle, read(args.edit) if args.edit else {}, args.out, args.resume, args.save_prefixes)


if __name__ == "__main__":
    main()
