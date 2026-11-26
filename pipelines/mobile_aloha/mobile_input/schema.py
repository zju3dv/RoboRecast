from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


FrameReader = Callable[[int], np.ndarray]


@dataclass(frozen=True)
class CameraStream:
    """One RGB stream aligned row-for-row with a canonical trajectory."""

    role: str
    rows: int
    read_rgb: FrameReader
    source_key: str
    native_size: tuple[int, int]
    encoding: str
    notes: Mapping[str, object] = field(default_factory=dict)

    def frame(self, row: int) -> np.ndarray:
        value = np.asarray(self.read_rgb(int(row)))
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError(f"camera {self.role} row {row} must decode to uint8 HxWx3 RGB")
        return value


@dataclass(frozen=True)
class CanonicalEpisodeV2:
    episode_id: str
    dataset: str
    source_path: Path
    task_name: str
    embodiment_id: str
    timestamp_sec: np.ndarray
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    arm_command: np.ndarray
    base_twist_body: np.ndarray
    route_prior_xyyaw: np.ndarray
    cameras: Mapping[str, CameraStream]
    source_spec: Mapping[str, object]
    source_audit: Mapping[str, object]
    time_sync: Mapping[str, object]
    camera_calibration: Mapping[str, object]
    route_spec: Mapping[str, object]
    task_prior: Mapping[str, object]
    embodiment: Mapping[str, object]
    source_extras: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamp_sec, dtype=np.float64)
        rows = len(timestamps)
        if rows < 2 or timestamps.shape != (rows,):
            raise ValueError("timestamp_sec must be [T], T>=2")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamp_sec must be finite and strictly increasing")
        required = {
            "arm_position": (self.arm_position, (rows, 14)),
            "arm_velocity": (self.arm_velocity, (rows, 14)),
            "arm_command": (self.arm_command, (rows, 14)),
            "base_twist_body": (self.base_twist_body, (rows, 3)),
            "route_prior_xyyaw": (self.route_prior_xyyaw, (rows, 3)),
        }
        for name, (value, shape) in required.items():
            array = np.asarray(value)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        required_roles = {"top", "left_wrist", "right_wrist"}
        if set(self.cameras) != required_roles:
            raise ValueError(f"camera roles must be {sorted(required_roles)}")
        if any(camera.rows != rows for camera in self.cameras.values()):
            raise ValueError("camera streams must align to trajectory rows")
        if not self.episode_id or not self.embodiment_id:
            raise ValueError("episode_id and embodiment_id are required")


def integrate_body_twist(timestamp_sec: np.ndarray, base_twist_body: np.ndarray) -> np.ndarray:
    """Integrate body-frame planar velocity with a midpoint heading update."""
    timestamps = np.asarray(timestamp_sec, dtype=np.float64)
    twist = np.asarray(base_twist_body, dtype=np.float64)
    if timestamps.ndim != 1 or twist.shape != (len(timestamps), 3):
        raise ValueError("timestamps must be [T] and base_twist_body [T,3]")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    route = np.zeros((len(timestamps), 3), dtype=np.float64)
    for row, dt in enumerate(np.diff(timestamps), start=1):
        vx, vy, yaw_rate = twist[row - 1]
        middle_yaw = route[row - 1, 2] + .5 * yaw_rate * dt
        cosine, sine = np.cos(middle_yaw), np.sin(middle_yaw)
        route[row, 0] = route[row - 1, 0] + (cosine * vx - sine * vy) * dt
        route[row, 1] = route[row - 1, 1] + (sine * vx + cosine * vy) * dt
        route[row, 2] = route[row - 1, 2] + yaw_rate * dt
    return route
