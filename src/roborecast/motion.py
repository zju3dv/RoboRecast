from __future__ import annotations
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from .robot_runtime_view import RobotRuntimeView

def _minimum_jerk(values: np.ndarray, source_time: np.ndarray,
                  target_time: np.ndarray) -> np.ndarray:
    output = np.empty((len(target_time),) + values.shape[1:], dtype=np.float64)
    for index, time_value in enumerate(target_time):
        left = min(max(int(np.searchsorted(source_time, time_value, side="right") - 1), 0), len(source_time) - 2)
        width = source_time[left + 1] - source_time[left]
        u = float(np.clip((time_value - source_time[left]) / width, 0.0, 1.0))
        s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
        output[index] = (1.0 - s) * values[left] + s * values[left + 1]
    return output

def sample_cartesian(source_time, positions, quaternions, target_time, orientation_time_law="linear"):
    """Shared compiler/FK targets; legacy records keep their original time law."""
    xyz = _minimum_jerk(positions, source_time, target_time)
    if orientation_time_law not in ("linear", "quintic"):
        raise ValueError("Unknown orientation_time_law")
    times = np.asarray(target_time, dtype=float)
    if orientation_time_law == "quintic":
        left = np.clip(np.searchsorted(source_time, times, side="right") - 1, 0, len(source_time) - 2)
        duration = source_time[left + 1] - source_time[left]
        u = np.clip((times - source_time[left]) / duration, 0, 1)
        smooth = u**3 * (10 + u * (-15 + 6*u))
        times = np.clip(source_time[left] + duration * smooth, source_time[0], source_time[-1])
    quat = Slerp(source_time, Rotation.from_quat(quaternions))(times).as_quat()
    return xyz, quat

def _target_timestamps(start: float, end: float, dt: float = 0.02) -> np.ndarray:
    """Build a dense timeline whose final sample never exceeds the source range."""
    target = np.arange(start, end, dt, dtype=np.float64)
    if not len(target) or target[-1] < end - 1e-9:
        target = np.append(target, end)
    else:
        target[-1] = end
    return target

def _solve_ik(model: mujoco.MjModel, initial_qpos: np.ndarray, positions: np.ndarray,
              quaternions_xyzw: np.ndarray,
              runtime_view: RobotRuntimeView) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    body_id = runtime_view.end_effector_body_id
    qpos_addresses = np.asarray(runtime_view.arm_qpos_addresses, dtype=np.intp)
    dof_addresses = np.asarray(runtime_view.arm_dof_addresses, dtype=np.intp)
    targets, errors = [], []
    for position, quaternion in zip(positions, quaternions_xyzw):
        desired_rotation = Rotation.from_quat(quaternion).as_matrix()
        error_norm = float("inf")
        for _ in range(120):
            mujoco.mj_forward(model, data)
            current_rotation = data.xmat[body_id].reshape(3, 3)
            position_error = position - data.xpos[body_id]
            rotation_error = Rotation.from_matrix(desired_rotation.dot(current_rotation.T)).as_rotvec()
            error = np.concatenate((position_error, 0.35 * rotation_error))
            error_norm = float(np.linalg.norm(error))
            if error_norm < 5e-4:
                break
            jacobian_position = np.zeros((3, model.nv))
            jacobian_rotation = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacobian_position, jacobian_rotation, body_id)
            jacobian = np.vstack((jacobian_position[:, dof_addresses],
                                  0.35 * jacobian_rotation[:, dof_addresses]))
            delta = jacobian.T.dot(np.linalg.solve(jacobian.dot(jacobian.T) + np.eye(6) * 1e-4, error))
            for local, joint_id in enumerate(runtime_view.arm_joint_ids):
                address = qpos_addresses[local]
                data.qpos[address] += float(np.clip(delta[local], -0.06, 0.06))
                if model.jnt_limited[joint_id]:
                    data.qpos[address] = np.clip(data.qpos[address],
                                                 model.jnt_range[joint_id, 0] + .01,
                                                 model.jnt_range[joint_id, 1] - .01)
        targets.append(data.qpos[qpos_addresses].copy())
        errors.append(error_norm)
    return np.asarray(targets), np.asarray(errors)
