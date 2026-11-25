"""Export consecutive simulated states, applied controls and optional camera frames."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from acceptance import accepted_run

import h5py
import mujoco
import numpy as np


def export(run, output, cameras, fps):
    accepted_run(run)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Image sample rate must be positive")
    model = mujoco.MjModel.from_xml_path(str(run / "bundle/scene.xml"))
    runtime = json.loads((run / "bundle/runtime.json").read_text())
    for name in cameras:
        model.camera(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with NamedTemporaryFile(dir=output.parent, suffix=".h5", delete=False) as temp:
        candidate = Path(temp.name)
    try:
        with np.load(run / "final/rollout.npz") as trace, h5py.File(candidate, "w") as dataset:
            times = trace["timestamp_sec"]
            qpos = trace["qpos"]
            dataset.attrs.update(arm_joint_names=json.dumps(runtime["arm_joint_names"]),
                                 actuator_names=json.dumps([model.actuator(i).name for i in range(model.nu)]),
                                 state_convention="Post-step states. action/ctrl[i] advances observation i to i+1.")
            dataset.create_dataset("timestamp_sec", data=times)
            for key in ("qvel", "arm_joint_position", "ee_position_m", "ee_quaternion_wxyz"):
                dataset.create_dataset("observation/" + key, data=trace[key], compression="gzip")
            dataset.create_dataset("observation/qpos", data=qpos, compression="gzip")
            dataset.create_dataset("action/ctrl", data=trace["ctrl"][1:], compression="gzip")
            if cameras:
                data = mujoco.MjData(model)
                indices = np.unique(np.minimum(np.searchsorted(times, np.arange(times[0], times[-1], 1/fps)), len(times)-1))
                dataset.create_dataset("images/state_index", data=indices)
                with mujoco.Renderer(model, height=360, width=640) as renderer:
                    streams = {cam: dataset.create_dataset(f"images/{cam}", (len(indices),360,640,3), dtype="u1",
                                                           chunks=(1,360,640,3), compression="lzf") for cam in cameras}
                    for j, i in enumerate(indices):
                        data.qpos[:] = qpos[i]
                        mujoco.mj_forward(model, data)
                        for cam, stream in streams.items():
                            renderer.update_scene(data, camera=cam)
                            stream[j] = renderer.render()
        candidate.replace(output)
    finally:
        candidate.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera", action="append", default=[])
    p.add_argument("--fps", type=float, default=10)
    a = p.parse_args()
    export(a.run, a.out, a.camera, a.fps)
