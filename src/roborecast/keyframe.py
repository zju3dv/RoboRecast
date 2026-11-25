"""Render one recorded simulation state using its corresponding scene candidate."""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from .harness import load_candidate, read, initial_state
from .inspection import render


def frame(model, qpos, camera, collision=False):
    data = initial_state(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return render(model, data, camera, collision)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("rollout", type=Path)
    p.add_argument("--edit", type=Path)
    p.add_argument("--time", type=float, required=True)
    p.add_argument("--camera", default="ext1")
    p.add_argument("--collision", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    model, _ = load_candidate(a.bundle, read(a.edit) if a.edit else {})
    with np.load(a.rollout) as dense:
        i = abs(dense["timestamp_sec"]-a.time).argmin()
        image = frame(model, dense["qpos"][i], a.camera, a.collision)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(a.out)


if __name__ == "__main__":
    main()
