"""Inspect settled support, joint travel and collision geometry in a prepared scene."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from .harness import initial_state, load_scene, read


def travel(model, joint, samples=121, penetration_m=0.001):
    data = initial_state(model)
    j = model.joint(joint)
    if not j.limited[0] or int(j.type[0]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ValueError("Select a limited hinge or slide joint")
    start = float(data.qpos[j.qposadr[0]])
    moving = {int(j.bodyid[0])}
    for idx in range(model.nbody):
        if model.body_parentid[idx] in moving:
            moving.add(idx)
    def pairs():
        out = {}
        for contact in data.contact:
            a, b = map(int, contact.geom)
            if (int(model.geom_bodyid[a]) in moving) != (int(model.geom_bodyid[b]) in moving):
                pair = tuple(sorted((a, b)))
                out[pair] = min(out.get(pair, 0.), float(contact.dist))
        return out
    baseline = pairs()
    rows = []
    for end in j.range:
        first = None
        for value in np.linspace(start, float(end), samples):
            data.qpos[j.qposadr] = value
            mujoco.mj_forward(model, data)
            for pair, dist in pairs().items():
                if dist < baseline.get(pair, 0.) - penetration_m:
                    first = dict(position=float(value), pair=[model.geom(g).name for g in pair], penetration_m=-dist)
                    break
            if first:
                break
        rows.append(dict(limit=float(end), first_obstruction=first))
    return dict(joint=joint, axis_local=j.axis.tolist(), start=start, directions=rows)


def settle(model, seconds=.4):
    data = initial_state(model)
    start = data.xpos.copy()
    depth = 0.
    for _ in range(round(seconds/model.opt.timestep)):
        mujoco.mj_step(model, data)
        depth = max(depth, max((-float(c.dist) for c in data.contact), default=0.))
    mujoco.mj_forward(model, data)
    return dict(seconds=seconds, maximum_penetration_m=depth,
                body_drift_m={model.body(i).name or str(i): float(np.linalg.norm(data.xpos[i]-start[i]))
                              for i in range(1, model.nbody)})


def render(model, data, camera, collision=False):
    options = mujoco.MjvOption()
    options.geomgroup[3] = int(collision)
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        return Image.fromarray(renderer.render().copy())


def overlay(model, data, body, path):
    center = data.body(body).xpos.copy()
    frames = []
    for az, el in [(90,0), (0,0), (90,-89.9)]:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.distance, camera.azimuth, camera.elevation = .95, az, el
        camera.orthographic = 1
        frames.append(render(model, data, camera, collision=True))
    sheet = Image.new("RGB", (640*3, 360))
    for i, frame in enumerate(frames):
        sheet.paste(frame, (640*i,0))
    sheet.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--edit", type=Path)
    p.add_argument("--joint", help="Optional bounded hinge or slide to probe")
    p.add_argument("--body")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    path = args.bundle / "scene.xml" if args.bundle.is_dir() else args.bundle
    model = load_scene(path, read(args.edit) if args.edit else {})
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(settling=settle(model))
    if args.joint:
        report["travel"] = travel(model, args.joint)
    if args.body:
        overlay(model, initial_state(model), args.body, args.out/"collision-views.jpg")
    (args.out/"inspection.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report.get("travel", report["settling"]), indent=2))


if __name__ == "__main__":
    main()
