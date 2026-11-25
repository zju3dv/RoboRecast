"""Measure end-effector motion on the recorded simulation or source clock."""
import numpy as np
from scipy.spatial.transform import Rotation


def motion_window(timestamps, positions, quaternions_wxyz, linear_threshold=.005,
                  angular_threshold=.05):
    """Unsmooth interval speeds; include internal dwell in the active envelope."""
    t = np.asarray(timestamps, float)
    p = np.asarray(positions, float)
    q = np.asarray(quaternions_wxyz, float)
    if (len(t) < 2 or t.ndim != 1 or p.shape != (len(t), 3) or q.shape != (len(t), 4)
            or not all(np.isfinite(a).all() for a in (t, p, q))
            or np.any(np.diff(t) <= 0) or np.any(np.linalg.norm(q, axis=1) < 1e-12)):
        raise ValueError('Finite ordered timestamps and complete EE poses required')
    if not np.isfinite([linear_threshold, angular_threshold]).all() or min(linear_threshold, angular_threshold) <= 0:
        raise ValueError('Positive finite velocity thresholds required')
    dt = np.diff(t)
    linear = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    angular = (rotation[:-1].inv() * rotation[1:]).magnitude() / dt
    active = (linear > linear_threshold) | (angular > angular_threshold)
    indices = np.flatnonzero(active)
    left, right = bool(active[0]), bool(active[-1])
    return dict(status='NO_DETECTED_MOTION' if not len(indices) else
                ('BOTH_CENSORED' if left and right else 'RIGHT_CENSORED' if right else
                 'LEFT_CENSORED' if left else 'OBSERVED_WINDOW'),
                linear_threshold_m_s=float(linear_threshold), angular_threshold_rad_s=float(angular_threshold),
                start_sec=float(t[indices[0]]) if len(indices) else None,
                end_sec=float(t[indices[-1]+1]) if len(indices) else None,
                active_envelope_sec=float(t[indices[-1]+1]-t[indices[0]]) if len(indices) else 0.,
                sum_active_intervals_sec=float(dt[active].sum()), active_interval_count=int(active.sum()),
                left_censored=left, right_censored=right,
                observed_below_threshold_tail_sec=float(t[-1]-t[indices[-1]+1]) if len(indices) else float(t[-1]-t[0]),
                terminal_linear_speed_m_s=float(linear[-1]), terminal_angular_speed_rad_s=float(angular[-1]),
                recorded_start_sec=float(t[0]), recorded_end_sec=float(t[-1]), recorded_span_sec=float(t[-1]-t[0]),
                sample_count=len(t), sample_interval_sec=dict(min=float(dt.min()), median=float(np.median(dt)), max=float(dt.max())))
