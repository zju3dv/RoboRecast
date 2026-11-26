from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LoadedEpisode:
    root: Path
    schema_version: str
    episode_id: str
    timestamp_sec: np.ndarray
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    arm_command: np.ndarray
    base_twist_body: np.ndarray
    route_prior_xyyaw: np.ndarray
    camera_paths: Mapping[str, Path]
    runtime_contract: Mapping[str, object]
    runtime_scene: Path
    initial_state: Path
    episode_contract: Mapping[str, object]

    @property
    def rows(self) -> int:
        return len(self.timestamp_sec)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_v2(root: Path, manifest: dict) -> LoadedEpisode:
    contract = _json(root / "episode_contract.json")
    trajectory_path = root / str(contract["trajectory_path"])
    with np.load(trajectory_path) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in (
            "timestamp_sec", "arm_position", "arm_velocity", "arm_command",
            "base_twist_body", "route_prior_xyyaw")}
    runtime_contract_path = root / str(contract["robot_runtime_contract_path"])
    runtime = _json(runtime_contract_path)
    runtime_root = runtime_contract_path.parent
    return LoadedEpisode(
        root, manifest["schema_version"], manifest["episode_id"],
        np.asarray(arrays["timestamp_sec"], dtype=np.float64),
        np.asarray(arrays["arm_position"], dtype=np.float32),
        np.asarray(arrays["arm_velocity"], dtype=np.float32),
        np.asarray(arrays["arm_command"], dtype=np.float32),
        np.asarray(arrays["base_twist_body"], dtype=np.float32),
        np.asarray(arrays["route_prior_xyyaw"], dtype=np.float64),
        {role: root / str(path) for role, path in contract["camera_roles"].items()},
        runtime, runtime_root / "robot_base_scene.xml", runtime_root / "initial_state.npz",
        contract)


def load_episode(input_dir: Path) -> LoadedEpisode:
    root = Path(input_dir).resolve()
    manifest = _json(root / "manifest.json")
    schema = str(manifest.get("schema_version", ""))
    if schema == "mobile-real2sim-input-v2":
        return _load_v2(root, manifest)
    raise ValueError(f"unsupported mobile input schema: {schema}")
