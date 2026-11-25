from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from .utils import read_json, write_json, digest, sha256


class Store:
    """Single-writer artifact graph for an episode, with stale-result rejection."""
    def __init__(self, root, input_dir=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / ".harness/state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            state = read_json(self.state_path)
            self.input_dir = Path(state["input_dir"])
            if input_dir is not None and Path(input_dir).resolve() != self.input_dir:
                raise ValueError("workspace already belongs to another input")
        else:
            if input_dir is None:
                raise ValueError("input_dir required for a new workspace")
            self.input_dir = Path(input_dir).resolve()
            write_json(self.state_path, {"version": "2.0.0", "input_dir": str(self.input_dir), "latest": {}})

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("artifact must be inside reconstruction workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _input_signature(self):
        return digest({name: sha256(self.input_dir / name) for name in ("meta.json", "reference.npz") if (self.input_dir / name).exists()})

    def latest(self, kind):
        state = read_json(self.state_path)
        if kind not in state["latest"]:
            raise ValueError(f"{kind} is not available; run its producing tool first")
        record = read_json(self.path(state["latest"][kind]))
        if record["input_signature"] != self._input_signature():
            raise ValueError(f"stale {kind}: human input changed")
        for dependency, expected in record["dependencies"].items():
            if self.latest(dependency)["id"] != expected:
                raise ValueError(f"stale {kind}: {dependency} changed")
        for relative, expected in record["files"].items():
            file = self.path(relative)
            if not file.is_file() or sha256(file) != expected:
                raise ValueError(f"stale {kind}: artifact changed: {relative}")
        return record

    def put(self, kind, payload, dependencies=(), files=()):
        dependency_ids = {key: self.latest(key)["id"] for key in dependencies}
        hashes = {}
        for filename in files:
            path = self.path(filename)
            hashes[str(path.relative_to(self.root))] = sha256(path)
        identifier = f"{kind}-{uuid4().hex[:16]}"
        record = {"id": identifier, "kind": kind, "payload": payload,
                  "dependencies": dependency_ids, "files": hashes,
                  "input_signature": self._input_signature(),
                  "created_at": datetime.now(timezone.utc).isoformat()}
        relative = f".harness/records/{identifier}.json"
        write_json(self.path(relative), record)
        state = read_json(self.state_path)
        state["latest"][kind] = relative
        write_json(self.state_path, state)
        return record
