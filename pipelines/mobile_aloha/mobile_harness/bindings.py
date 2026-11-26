"""Keep scene revisions, prepared sources and reviews tied to the files they describe."""

import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

from mobile_input.writer import sha256, write_json


def scene_digest(path):
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    digest = hashlib.sha256(path.read_bytes())
    files = set()
    for element in root.findall("./asset/*[@file]"):
        key = "texturedir" if element.tag == "texture" else "meshdir"
        directory = compiler.get(key, compiler.get("assetdir", "")) if compiler is not None else ""
        files.add((path.parent / directory / element.attrib["file"]).resolve())
    for asset in sorted(files):
        digest.update(bytes.fromhex(sha256(asset)))
    return digest.hexdigest()


def current_scene(store, state):
    merge = state["merges"][-1]
    if (merge["units"] != state["units"]
            or scene_digest(store.artifact(merge["scene_xml_path"])) != merge["scene_sha256"]
            or sha256(store.artifact(merge["settled_state_path"])) != merge["settled_sha256"]):
        raise ValueError("Registered units, meshes or settled state changed. Merge the scene again.")
    return merge


def bind_source(work, episode):
    manifest = json.loads((episode / "manifest.json").read_text())
    runtime = json.loads((episode / "derived/robot_runtime/robot_runtime_contract.json").read_text())
    record = dict(episode=str(episode), episode_id=manifest["episode_id"],
                  base_pose_joints=runtime["base_pose_joints"],
                  source_sha256=json.loads((episode / "specs/source.json").read_text())["source_sha256"],
                  manifest_sha256=sha256(episode / "manifest.json"))
    write_json(work / "episode.json", record)


def check_source(record, episode):
    if sha256(episode / "manifest.json") != record["manifest_sha256"]:
        raise ValueError("The prepared episode differs from the one used for this rollout")
    manifest = json.loads((episode / "manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["path"].startswith("derived/videos/") and sha256(episode / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Source video or frame mapping changed: {entry['path']}")


def bundle_assets(tree, source_dir, output):
    """Freeze mesh and texture files beside a scene with relative MJCF paths."""
    compiler = tree.find("compiler")
    assets = output / "assets"
    for element in tree.findall("./asset/*[@file]"):
        directory = "texturedir" if element.tag == "texture" else "meshdir"
        path = (source_dir / compiler.get(directory, compiler.get("assetdir", "")) / element.attrib["file"]).resolve()
        assets.mkdir(exist_ok=True)
        prefix = sha256(path)[:16] + "-"
        target = assets / (path.name if path.name.startswith(prefix) else prefix + path.name)
        if not target.exists():
            shutil.copyfile(path, target)
        if element.get("name") is None:
            element.set("name", path.stem)
        element.set("file", target.relative_to(output).as_posix())
    for key in ("meshdir", "texturedir", "assetdir"):
        compiler.attrib.pop(key, None)
