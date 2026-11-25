"""One differential-drive midpoint integrator for route prediction and execution."""
from __future__ import annotations
import mujoco
import numpy as np
from .robot_runtime import runtime_contract, base_pose, set_base_pose

def integrate_pose(pose, command, dt):
    pose, command = np.asarray(pose, float), np.asarray(command, float)
    if pose.shape != (3,) or command.shape != (2,) or not np.isfinite(np.r_[pose,command,dt]).all() or dt < 0:
        raise ValueError('Expected finite x,y,yaw and differential [v,omega], dt >= 0')
    v, omega = command
    mid = pose[2] + .5*omega*dt
    return pose + np.array([v*np.cos(mid)*dt, v*np.sin(mid)*dt, omega*dt])

def world_twist(pose, command):
    v, omega = np.asarray(command, float)
    return np.array([v*np.cos(pose[2]), v*np.sin(pose[2]), omega])

def validate_command(contract, command, previous=None, dt=None):
    command = np.asarray(command, float)
    if command.shape != (2,) or not np.isfinite(command).all():
        raise ValueError('base command must contain finite [v_mps,omega_radps]; lateral commands are unsupported')
    if contract['motion_mode'] != 'mobile' and np.any(np.abs(command)>1e-12):
        raise ValueError('not_mobile input forbids base movement')
    limits = contract['base']['limits']
    if np.any(np.abs(command) > np.array([limits['v_mps'], limits['omega_radps']])+1e-9):
        raise ValueError('base command exceeds bundle velocity limits')
    if previous is not None:
        if dt is None or dt<=0: raise ValueError('positive dt required for acceleration checks')
        if np.any(np.abs(command-np.asarray(previous))/dt > np.array([limits['accel_mps2'], limits['angular_accel_radps2']])+1e-7):
            raise ValueError('base command exceeds bundle acceleration limits')
    return command

def step_base(model, data, command, previous=None):
    """Step other dynamics, then enforce the declared kinematic planar base.

    Callers check forbidden contacts before/after stepping; collision response does
    not stop the prescribed base. No tire traction model is claimed.
    """
    contract = runtime_contract(model)
    command = validate_command(contract, command, previous, float(model.opt.timestep))
    pose = base_pose(model, data)
    if contract['motion_mode'] == 'mobile':
        set_base_pose(model, data, pose, command)
    mujoco.mj_step(model, data)
    if contract['motion_mode'] == 'mobile':
        next_pose = integrate_pose(pose, command, float(model.opt.timestep))
        set_base_pose(model, data, next_pose, command)
    return base_pose(model, data)
