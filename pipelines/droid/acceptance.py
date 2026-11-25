"""Tie a DROID acceptance record to its executable scene, inputs and trace."""
import mujoco

from preprocessing.util import read_json, sha256_file
from roborecast.requirements import digest
from roborecast.rollout import model_signature


def fingerprint(work):
    bundle = work / "bundle"
    model = mujoco.MjModel.from_xml_path(str(bundle / "scene.xml"))
    inputs = {name: read_json(bundle / (name + ".json")) for name in
              ("requirements", "runtime", "program", "success", "observables")}
    task = read_json(work / "task.json")
    inputs["intent"] = read_json(work / "intent.json")
    inputs["task"] = task
    if task["mode"] == "augment":
        inputs["bindings"] = read_json(work / "bindings.json")
        roles = work / "layout-roles.json"
        inputs["layout_roles"] = read_json(roles) if roles.exists() else None
    review = read_json(work / "visual-review.json")
    inputs["review"] = review
    frames = {name: sha256_file(work / review[name]) for name in ("source_frame", "rollout_frame")}
    return digest({"model": model_signature(model), "inputs": inputs, "frames": frames,
                   "trace": sha256_file(work / "final/rollout.npz")})


def accepted_run(work):
    record = read_json(work / "accepted.json")
    if record.get("status") != "PASS" or record.get("fingerprint") != fingerprint(work):
        raise ValueError(f"Run verify again for the current candidate in {work}")
    return record
