"""Runtime-declared gripper command conversion, with legacy linear fallback."""
import numpy as np


def width_to_control(gripper, width):
    width=float(width)
    if not np.isfinite(width): raise ValueError('gripper width must be finite')
    table=gripper.get('width_control_table')
    if table is None:
        return (1.0-width/float(gripper['maximum_width_m']))*float(gripper['closed_driver_rad'])
    x=np.asarray(table['width_m'],float); y=np.asarray(table['control_rad'],float)
    direction=gripper.get('control_direction','decreasing')
    if direction not in ('increasing','decreasing'):
        raise ValueError('invalid width/control direction')
    nonmonotonic=np.any(np.diff(y)<0) if direction=='increasing' else np.any(np.diff(y)>0)
    if x.ndim!=1 or len(x)<2 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x)<=0) or nonmonotonic:
        raise ValueError('invalid monotonic width/control calibration')
    return float(np.interp(width,x,y))
