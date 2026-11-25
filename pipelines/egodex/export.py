"""Deliver a verified paired-VX300S rollout without advancing simulation time."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from fixed_robot import checkpoints
from fixed_robot.robot_runtime import runtime_contract
from fixed_robot.store import Store
from transfer.blender_export import export as export_visual
from roborecast.operation_timing import motion_window


def delivery(work, output, fps=24, render=False):
    tick = time.monotonic()
    from transfer.selection import selected

    ctx = selected(Store(work))
    from transfer.review import accepted, operation_window

    rollout, review, assessment = accepted(ctx)
    r = rollout["payload"]
    scene = ctx.path(ctx.latest("placement")["payload"]["scene_xml"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    arms = runtime_contract(model)["arms"]
    states = ctx.path(r["physics_steps_npz"])
    with np.load(states) as z:
        ts = z["time"]
        integration = z["integration_state"]
        controls = z["control"]
    if len(ts) != len(controls) or np.any(np.diff(ts) <= 0) or len(ts) < 2:
        raise ValueError("Incomplete physics stream")
    requirements = ctx.latest("source_requirements")["payload"]
    qpos = []
    qvel = []
    ee = {side: [] for side in arms}
    quats = {side: [] for side in arms}
    for row in integration:
        mujoco.mj_setState(model, data, row, checkpoints.STATE)
        mujoco.mj_forward(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        for side, arm in arms.items():
            site = model.site(arm["eef_site"]).id
            ee[side].append(data.site_xpos[site].copy())
            quats[side].append(
                Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_quat()[
                    [3, 0, 1, 2]
                ]
            )
    qpos = np.asarray(qpos)
    qvel = np.asarray(qvel)
    output.mkdir(parents=True, exist_ok=True)
    by_arm = {side: motion_window(ts, ee[side], quats[side]) for side in arms}
    observed = output / "observations_actions.npz"
    # Row i's actuator command is applied between measured states i and i+1.
    np.savez_compressed(
        observed,
        timestamp_sec=ts,
        qpos=qpos,
        qvel=qvel,
        action=controls[:-1],
        observation_index=np.arange(len(ts) - 1),
        next_observation_index=np.arange(1, len(ts)),
    )
    report = dict(
        episode=str(ctx.input_dir),
        scene=str(scene),
        rollout_id=rollout["id"],
        review_id=review["id"],
        interaction_coverage_verified="source_requirements" in rollout["dependencies"],
        state_source=str(states),
        state_source_sha256=hashlib.sha256(states.read_bytes()).hexdigest(),
        physics_diagnostics=str(ctx.path(r["diagnostics_json"])),
        visual_review=assessment,
        source_requirements=requirements,
        operation_timing=operation_window(ctx),
        motion_from_measured_pinch_poses=by_arm,
        control_transitions=len(controls) - 1,
        actuator_order=[model.actuator(i).name for i in range(model.nu)],
        coordinates=arms,
        training_npz=str(observed),
        task_acceptance_epoch=datetime.fromisoformat(review["created_at"]).timestamp(),
    )
    trajectory = ctx.latest("trajectory")["payload"]["time_map"]
    report["compiled_phase_times"] = [
        {
            k: phase[k]
            for k in ("phase", "source_start", "source_end", "sim_start", "sim_end")
        }
        for phase in trajectory
    ]
    report["export_seconds"] = time.monotonic() - tick
    (output / "final_result.json").write_text(json.dumps(report, indent=2))
    if not render:
        return report
    visual = output / "blender"
    export_visual(scene, observed, visual, fps)
    # Add a presentation camera to evaluated geometry. The physical states stay source-bound.
    mfile = visual / "manifest.json"
    meta = json.loads(mfile.read_text())
    vfile = visual / "states.npz"
    with np.load(vfile) as z:
        arrays = {k: z[k] for k in z}
    units = ctx.latest("units")["payload"]["units"]
    relevant = {k for k, v in units.items() if v["role"] == "task_object"}
    ids = [
        i
        for i in range(model.ngeom)
        if model.body(model.geom_bodyid[i]).name in relevant
    ]
    if not ids:
        raise ValueError("Expected named task bodies for presentation camera")
    cloud = arrays["geom_pos"][:, ids].reshape(-1, 3)
    low, high = cloud.min(0), cloud.max(0)
    target = (high + low) / 2
    span = max(0.34, np.linalg.norm(high - low) + 0.12)
    target[2] += 0.045
    # Look from the demonstrator side, slightly oblique to reveal the gripper and open container.
    camera_source = np.load(ctx.input_dir / "reference.npz")["T_world_camera"][0]
    align = np.array(
        json.loads((ctx.input_dir / "meta.json").read_text())["T_sim_source_world"]
    )
    horizontal = (align @ camera_source)[:3, 3] - target
    horizontal[2] = 0
    horizontal /= max(np.linalg.norm(horizontal), 1e-6)
    from transfer.camera import select_camera

    active_sides = requirements.get("active_hands") or list(arms)
    probes = np.unique(np.linspace(0, len(qpos) - 1, 6).astype(int))
    score, eye, rotation, counts = select_camera(
        model, qpos[probes], sorted(relevant), active_sides, target, span, horizontal
    )
    rot = rotation.flatten()
    meta["presentation_view"] = dict(
        eye=eye.tolist(),
        target=target.tolist(),
        score=score,
        objects=sorted(relevant),
        visible_pixels=counts,
        probe_indices=probes.tolist(),
    )
    n = len(arrays["time"])
    ci = arrays["camera_pos"].shape[1]
    arrays["camera_pos"] = np.concatenate(
        [arrays["camera_pos"], np.tile(eye, (n, 1, 1))], axis=1
    )
    arrays["camera_rot"] = np.concatenate(
        [arrays["camera_rot"], np.tile(rot, (n, 1, 1))], axis=1
    )
    np.savez_compressed(vfile, **arrays)
    meta["cameras"].append(dict(id=ci, name="operation", fovy=42))
    meta["appearance_profile"] = "egodex-source-srgb"
    cloth_bodies = {
        name
        for name, unit in units.items()
        if unit["role"] == "support"
        and any(x in str(unit["evidence"]).lower() for x in ("cloth", "fabric"))
    }
    meta["cloth_geoms"] = [
        int(i)
        for i in range(model.ngeom)
        if model.body(model.geom_bodyid[i]).name in cloth_bodies
    ]
    mfile.write_text(json.dumps(meta, indent=2))
    blender = os.environ.get(
        "BLENDER",
        "blender",
    )
    cmd = [
        blender,
        "-b",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).parent / "transfer/render_blender.py"),
        "--",
        "--input",
        str(visual),
        "--camera",
        "operation",
        "--width",
        "768",
        "--height",
        "432",
        "--samples",
        "12",
        "--device",
        os.environ.get("ROBORECAST_RENDER_DEVICE", "CUDA"),
    ]
    with (visual / "blender.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    video = output / "simulation.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(visual / "frames/%05d.png"),
            "-frames:v",
            str(meta["frame_count"]),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    report.update(blender_video=str(video), delivery_seconds=time.monotonic() - tick)
    from transfer.comparison import compare

    timing = report["operation_timing"]
    report["comparison"] = compare(
        ctx.input_dir / "source/rgb.mp4",
        video,
        output / "comparison.mp4",
        timing["source_window"]["start"],
        timing["corresponding_sim_seconds"][0],
        fps,
    )
    report["delivery_seconds"] = time.monotonic() - tick
    (output / "final_result.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    result = delivery(
        args.workspace.resolve(), args.output.resolve(), render=args.render
    )
    print(result["rollout_id"], result["control_transitions"])
