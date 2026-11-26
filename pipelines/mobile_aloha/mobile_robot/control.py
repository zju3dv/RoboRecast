from __future__ import annotations

import math

import mujoco
import numpy as np


def planar_pose_step(pose: np.ndarray, v: float, omega: float, dt: float,
                     lateral: float = 0.0) -> np.ndarray:
    """Integrate body-frame `(forward, lateral, yaw_rate)` on the plane."""
    x, y, yaw = map(float, pose)
    middle = yaw + .5 * omega * dt
    cosine, sine = math.cos(middle), math.sin(middle)
    return np.array([x + (cosine * v - sine * lateral) * dt,
                     y + (sine * v + cosine * lateral) * dt,
                     yaw + omega * dt])

def step_mobile(model, data, base_qpos, base_dofs, twist):
    """Advance arm/object dynamics and integrate the commanded chassis velocity."""
    forward, lateral, turn = twist
    yaw = data.qpos[base_qpos[2]]
    next_pose = planar_pose_step(data.qpos[base_qpos], forward, turn, model.opt.timestep, lateral)
    data.qvel[base_dofs] = (math.cos(yaw)*forward - math.sin(yaw)*lateral,
                            math.sin(yaw)*forward + math.cos(yaw)*lateral, turn)
    mujoco.mj_step(model, data)
    data.qpos[base_qpos] = next_pose
    yaw = next_pose[2]
    data.qvel[base_dofs] = (math.cos(yaw)*forward - math.sin(yaw)*lateral,
                            math.sin(yaw)*forward + math.cos(yaw)*lateral, turn)
    mujoco.mj_forward(model, data)
