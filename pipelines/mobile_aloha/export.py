"""Export and independently replay the accepted mobile controls on the delivered scene."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from mobile_harness.store import Store
from mobile_harness.bindings import bundle_assets, check_source
from mobile_input.writer import sha256, write_json
from replay import replay


def copy_scene(source, output):
    tree = ET.parse(source)
    bundle_assets(tree, source.parent, output)
    path = output / "scene.xml"
    tree.write(path, encoding="utf-8")
    return path


def delivery(work, output, render=False, episode_override=None):
    from mobile_harness.review import accepted
    store = Store(work)
    if not accepted(store):
        raise ValueError("The latest complete rollout needs a passing owner review")
    state = store.read()
    rollout, visual = state["rollouts"][-1], state["reviews"][-1]
    source = json.loads((store.root / "episode.json").read_text())
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        dest = Path(temp) / "delivery"
        dest.mkdir()
        source_scene = store.artifact(state["merges"][-1]["scene_xml_path"])
        scene = copy_scene(source_scene, dest)
        recorded = store.artifact(rollout["rollout_path"])
        with np.load(recorded) as z:
            check = replay(scene, z, source["base_pose_joints"])
        shutil.copyfile(recorded, dest / "observations_actions.npz")
        for relative, target in ((rollout["diagnostics_path"], "diagnostics.json"),
                               (visual["path"], "review.json"),
                               (state["trajectories"][-1]["source_spec_path"], "phases.json")):
            shutil.copyfile(store.artifact(relative), dest / target)
        report = dict(episode=source["episode_id"], source_sha256=source["source_sha256"],
                      rollout_revision=rollout["revision"], scene_sha256=sha256(scene),
                      base_pose_joints=source["base_pose_joints"], replay=check)
        if render:
            from mobile_harness.finalize import render_comparison
            episode = episode_override or Path(source["episode"])
            check_source(source, episode)
            render_comparison(episode, store, dest)
            report["comparison"] = "comparison.mp4"
        write_json(dest / "final_result.json", report)
        dest.rename(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episode", type=Path, help="Location of the prepared episode if moved before rendering")
    args = parser.parse_args()
    print(json.dumps(delivery(args.workspace, args.output, args.render, args.episode), indent=2))
