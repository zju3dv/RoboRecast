from __future__ import annotations

import json
from pathlib import Path

from mobile_input.writer import write_json


class Store:
    def __init__(self, reconstruction_dir: Path):
        self.root = Path(reconstruction_dir).resolve()
        self.path = self.root / ".agent_run/harness_state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.is_file() else {
            "schema_version": "mobile-harness-state-v1", "stage": "BOOTSTRAPPED",
            "units": {}, "merges": [], "trajectories": [], "rollouts": [], "reviews": []}

    def save(self, state: dict) -> None:
        write_json(self.path, state)

    def artifact(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("artifact must be inside reconstruction")
        return path
