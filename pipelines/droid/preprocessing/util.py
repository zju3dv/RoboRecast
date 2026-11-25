from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = path.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(path)))
    return h.hexdigest()


def run(cmd: Iterable[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=True, text=True, capture_output=True)


def ffprobe_video(path: Path) -> Dict[str, Any]:
    result = run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration:stream_tags=rotate",
        "-of", "json", str(path),
    ])
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}, got {len(streams)}")
    stream = streams[0]
    for key in ("width", "height", "nb_frames", "nb_read_frames"):
        if key in stream and stream[key] not in (None, "N/A"):
            stream[key] = int(stream[key])
    if "duration" in stream and stream["duration"] not in (None, "N/A"):
        stream["duration"] = float(stream["duration"])
    return stream
