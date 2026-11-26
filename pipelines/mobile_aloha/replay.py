"""Replay the delivered scene and synchronized chassis/arm actions without the original episode."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from mobile_harness.stages import restore
from mobile_robot.control import step_mobile


def replay(scene, states, joint_names):
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    restore(model, data, states["initial_state"])
    joints = [model.joint(name).id for name in joint_names]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    control, twists = states["step_control"], states["step_twist"]
    end_step = states["row_end_step"]
    qp, qv = states["qpos"], states["qvel"]
    if (len(control) != len(twists) or len(qp) != len(end_step) or end_step[0] != 0
            or end_step[-1] != len(control) or np.any(np.diff(end_step) < 0)):
        raise ValueError("Invalid control-to-observation alignment")
    errors = np.zeros(2)
    step = 0
    for row, end in enumerate(end_step):
        while step < end:
            data.ctrl[:] = control[step]
            step_mobile(model, data, qadr, vadr, twists[step])
            step += 1
        errors = np.maximum(errors, [np.max(np.abs(data.qpos - qp[row])),
                                     np.max(np.abs(data.qvel - qv[row]))])
    if not np.all(errors < [1e-7, 1e-6]):
        raise ValueError(f"Exported controls disagree with recorded states: {errors}")
    return dict(substeps=step, timestep_sec=model.opt.timestep,
                max_qpos_error=float(errors[0]), max_qvel_error=float(errors[1]),
                actuator_names=[model.actuator(i).name for i in range(model.nu)],
                body_names=[model.body(i).name for i in range(model.nbody)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path)
    root = parser.parse_args().delivery
    info = json.loads((root / "final_result.json").read_text())
    with np.load(root / "observations_actions.npz") as states:
        result = replay(root / "scene.xml", states, info["base_pose_joints"])
    print(json.dumps(result, indent=2))
