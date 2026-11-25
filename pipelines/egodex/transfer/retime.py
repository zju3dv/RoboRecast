"""Retiming trials on an accepted control path, evaluated with the original physical predicates."""

import argparse
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from fixed_robot.robot_runtime import runtime_contract
from fixed_robot.rollout import simulate_task_rollout
from fixed_robot.store import Store


def accelerated_clock(t, q, vmax, amax, ceiling, keep):
    """Bound path speed and acceleration; use original full-rate joint controls as the path."""
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Expected increasing trajectory time")
    # Path parameter is original time. Each LP variable is squared path speed at a knot.
    v = np.diff(q, axis=0) / dt[:, None]
    speed = np.max(np.abs(np.gradient(q, t, axis=0)), axis=1)
    upper = np.minimum(ceiling**2, (vmax / np.maximum(1e-8, speed)) ** 2)
    upper[keep] = 1.0
    rr = []
    cc = []
    vv = []
    limit = []
    for i in range(1, len(t) - 1):
        h = (dt[i - 1] + dt[i]) * 0.5
        for j in range(q.shape[1]):
            mid = (v[i, j] + v[i - 1, j]) * 0.5
            curve = (v[i, j] - v[i - 1, j]) / h
            for sign in (-1.0, 1.0):
                row = len(limit)
                rr.extend([row, row, row])
                cc.extend([i - 1, i, i + 1])
                vv.extend([-sign * mid / (4 * h), sign * curve, sign * mid / (4 * h)])
                limit.append(amax)
    constraints = coo_matrix((vv, (rr, cc)), shape=(len(limit), len(t))).tocsr()
    weights = np.r_[dt[0], (dt[1:] + dt[:-1]) / 2, dt[-1]]
    opt = linprog(
        -weights,
        A_ub=constraints,
        b_ub=limit,
        bounds=[(0.25, max(0.25, x)) for x in upper],
        method="highs",
    )
    if not opt.success:
        raise ValueError(opt.message)
    local_speed = np.sqrt(opt.x)
    new_dt = 2 * dt / (local_speed[1:] + local_speed[:-1])
    # The simulator interpolates joint controls between samples. Verify discrete rate changes.
    for _ in range(18):
        new_v = np.diff(q, axis=0) / new_dt[:, None]
        a = np.max(np.abs(np.diff(new_v, axis=0)), axis=1) / (
            (new_dt[1:] + new_dt[:-1]) / 2
        )
        boost = np.maximum(1.0, np.max(np.abs(new_v), axis=1) / vmax)
        needed = np.sqrt(np.maximum(1.0, a / amax))
        boost[:-1] = np.maximum(boost[:-1], needed)
        boost[1:] = np.maximum(boost[1:], needed)
        if boost.max() <= 1.005:
            break
        new_dt *= boost
    return np.r_[t[0], t[0] + np.cumsum(new_dt)]


def copy_dependencies(original, destination, head="trajectory"):
    """Fork only immutable dependencies into a fresh candidate store. Preserve the accepted run."""
    ctx = Store(destination, original.input_dir)
    latest = {}

    def copy_record(kind):
        if kind in latest:
            return
        rec = original.latest(kind)
        for dep in rec["dependencies"]:
            copy_record(dep)
        file = Path(".harness/records") / (rec["id"] + ".json")
        for rel in [*rec["files"], file]:
            dest = ctx.path(rel)
            if not dest.exists():
                shutil.copy2(original.path(rel), dest)
        latest[kind] = str(file)

    copy_record(head)
    copy_record("source_requirements")
    state = json.loads(ctx.state_path.read_text())
    state["latest"] = latest
    ctx.state_path.write_text(json.dumps(state))
    return ctx


def trial(
    original,
    folder,
    speedup,
    keep_grasps=True,
    grip_lead=0.0,
    grasp_speed=None,
    release_lead=0.0,
    release_speed=None,
    release_side=None,
    overlap=False,
):
    if not np.isfinite(speedup) or speedup < 1:
        raise ValueError("speedup must be finite and at least one")
    if any(not 0 <= lead < 1 for lead in (grip_lead, release_lead)):
        raise ValueError("Finger leads must be in [0,1)")
    if any(
        v is not None and (not np.isfinite(v) or v < 1)
        for v in (grasp_speed, release_speed)
    ):
        raise ValueError("Phase speed ceilings must be finite and at least one")
    initial_rollout = original.latest("rollout")
    if (
        not initial_rollout["payload"]["authoritative"]
        or initial_rollout["payload"]["status"] != "PASS"
    ):
        raise ValueError("Trial requires an authoritative passing parent")
    ctx = copy_dependencies(original, folder)
    rec = ctx.latest("trajectory")
    traj = rec["payload"]
    spec = json.loads(ctx.path(traj["spec_json"]).read_text())
    model = mujoco.MjModel.from_xml_path(
        str(ctx.path(ctx.latest("placement")["payload"]["scene_xml"]))
    )
    ids = [
        model.actuator(name).id
        for arm in runtime_contract(model)["arms"].values()
        for name in arm["actuators"]
    ]
    with np.load(ctx.path(traj["trajectory_npz"])) as z:
        arr = {k: z[k] for k in z}
    t = arr["time"]
    keep = np.zeros(len(t), bool)
    if keep_grasps:
        for phase in traj["time_map"]:
            if any(
                x.get("mode") in ("grasp", "release")
                for x in phase["hand_objects"].values()
            ):
                keep |= (t >= phase["sim_start"]) & (t <= phase["sim_end"])
    nt = accelerated_clock(
        t,
        arr["ctrl"][:, ids],
        float(spec.get("joint_velocity_limit", 1.5)),
        float(spec.get("joint_acceleration_limit", 5.0)),
        speedup,
        keep,
    )
    for phase in traj["time_map"]:
        lo, hi = phase["sim_start"], phase["sim_end"]
        selected = np.where((t >= lo - 1e-8) & (t <= hi + 1e-8))[0]
        if len(selected) < 2:
            continue
        for side, binding in phase["hand_objects"].items():
            if binding.get("mode") == "release" and release_lead:
                col = model.actuator(
                    runtime_contract(model)["arms"][side]["gripper"]["actuator"]
                ).id
                sample = t[selected]
                cmd = arr["ctrl"][selected, col].copy()
                if cmd[-1] > cmd[0]:
                    arr["ctrl"][selected, col] = np.interp(
                        lo + (sample - lo) / (1 - release_lead), sample, cmd
                    )
            duration_scale = (
                release_speed
                if binding.get("mode") == "release" and release_side in (None, side)
                else grasp_speed
                if binding.get("mode") == "grasp"
                else None
            )
            if binding.get("mode") not in ("grasp", "release"):
                continue
            if grip_lead and binding.get("mode") == "grasp":
                col = model.actuator(
                    runtime_contract(model)["arms"][side]["gripper"]["actuator"]
                ).id
                sample = t[selected]
                cmd = arr["ctrl"][selected, col].copy()
                if cmd[-1] < cmd[0]:
                    arr["ctrl"][selected, col] = np.interp(
                        lo + (sample - lo) / (1 - grip_lead), sample, cmd
                    )
            if duration_scale:
                a, b = selected[0], selected[-1]
                desired = (t[b] - t[a]) / duration_scale
                if nt[b] - nt[a] < desired:
                    extra = desired - (nt[b] - nt[a])
                    nt[b + 1 :] += extra
                    nt[a : b + 1] += (t[a : b + 1] - t[a]) / (t[b] - t[a]) * extra
    if nt[-1] >= t[-1]:
        return {"status": "NO_SHORTER_PATH", "duration_s": float(nt[-1])}
    for phase in traj["time_map"]:
        for key in ("sim_start", "sim_end"):
            phase[key] = float(np.interp(phase[key], t, nt))
        for event in phase.get("events", []):
            event["sim_time"] = float(np.interp(event["sim_time"], t, nt))
        phase["retime_factor"] = (phase["sim_end"] - phase["sim_start"]) / max(
            1e-9, phase["source_end"] - phase["source_start"]
        )
    arr["time"] = nt
    if overlap:
        from transfer.overlap import overlap_unloaded

        traj["unloaded_overlap"] = overlap_unloaded(
            arr, traj["time_map"], model, runtime_contract(model)["arms"]
        )
    # Finger scheduling and overlapping phases can change discrete joint acceleration.
    # Preserve the new geometry while stretching only intervals that exceed its limits.
    before = arr["time"]
    dt = np.diff(before)
    joints = arr["ctrl"][:, ids]
    vmax = float(spec.get("joint_velocity_limit", 1.5))
    amax = float(spec.get("joint_acceleration_limit", 5.0))
    for _ in range(80):
        v = np.diff(joints, axis=0) / dt[:, None]
        a = np.diff(v, axis=0) / ((dt[:-1] + dt[1:])[:, None] * 0.5)
        if np.max(np.abs(v)) <= vmax * 1.005 and np.max(np.abs(a)) <= amax * 1.005:
            break
        # A uniform stretch inside a small neighborhood preserves adjacent slopes.
        scale = np.maximum(1.0, np.max(np.abs(v), axis=1) / vmax)
        needed = np.sqrt(np.maximum(1.0, np.max(np.abs(a), axis=1) / amax))
        scale[:-1] = np.maximum(scale[:-1], needed)
        scale[1:] = np.maximum(scale[1:], needed)
        from scipy.ndimage import maximum_filter1d

        dt *= maximum_filter1d(scale, size=5, mode="nearest")
    else:
        return dict(status="RATE_LIMIT", parent_rollout=initial_rollout["id"])
    after = np.r_[before[0], before[0] + np.cumsum(dt)]
    if after[-1] >= t[-1]:
        return dict(status="NO_SHORTER_PATH", duration_s=float(after[-1]))
    for phase in traj["time_map"]:
        for key in ("sim_start", "sim_end"):
            phase[key] = float(np.interp(phase[key], before, after))
        for event in phase.get("events", []):
            event["sim_time"] = float(np.interp(event["sim_time"], before, after))
        phase["retime_factor"] = (phase["sim_end"] - phase["sim_start"]) / max(
            1e-9, phase["source_end"] - phase["source_start"]
        )
    arr["time"] = after
    out = ctx.path("retiming/trajectory.npz")
    np.savez_compressed(out, **arr)
    traj["trajectory_npz"] = str(out.relative_to(ctx.root))
    traj["retimed_from"] = rec["id"]
    ctx.put(
        "trajectory",
        traj,
        dependencies=["placement"],
        files=[traj["trajectory_npz"], traj["spec_json"], traj["resolved_spec_json"]],
    )
    success = ctx.path("retiming/success.json")
    shutil.copy2(original.path(initial_rollout["payload"]["spec_json"]), success)
    result = simulate_task_rollout(ctx, dict(success_json_path=str(success)))
    return dict(
        status=result["payload"]["status"],
        rollout_id=result["id"],
        parent_rollout=initial_rollout["id"],
        duration_s=float(arr["time"][-1]),
        parent_duration_s=float(t[-1]),
        speed_ceiling=speedup,
        diagnostics=str(ctx.path(result["payload"]["diagnostics_json"])),
        note="New physical rollout; a separate matching visual review is required for delivery.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workspace", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--speedup", type=float, default=1.8)
    p.add_argument("--adjust-grasps", action="store_true")
    p.add_argument("--grip-lead", type=float, default=0.0)
    p.add_argument("--grasp-speed", type=float)
    p.add_argument("--release-lead", type=float, default=0.0)
    p.add_argument("--release-speed", type=float)
    p.add_argument("--release-side", choices=("left", "right"))
    p.add_argument("--overlap", action="store_true")
    a = p.parse_args()
    if a.candidate.exists():
        p.error("Use a new candidate workspace")
    start = time.monotonic()
    result = trial(
        Store(a.workspace),
        a.candidate,
        a.speedup,
        not a.adjust_grasps,
        a.grip_lead,
        a.grasp_speed,
        a.release_lead,
        a.release_speed,
        a.release_side,
        a.overlap,
    )
    result["wall_seconds"] = time.monotonic() - start
    (a.candidate / "timing_trial.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
