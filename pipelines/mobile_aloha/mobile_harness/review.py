from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
from .rendering import presentation_renderer, presentation_options, motion_review_clips
import numpy as np
from PIL import Image, ImageDraw

from mobile_input.writer import sha256, write_json
from mobile_input.loader import load_episode
from .evidence import evidence_rows
from .store import Store
from .bindings import current_scene, check_source


def _real_frame(video: Path, row: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, row)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot read {video} at {row}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def operation_relation_evidence(root: Path) -> dict:
    """Small author claims for source verification, not automatic acceptance."""
    notes = root / "workspace/modeling_notes.json"
    if not notes.is_file():
        return {"operation_relations": [], "modeling_notes_path": None}
    content = json.loads(notes.read_text())
    return {"operation_relations": content.get("operation_relations", []),
            "modeling_notes_path": str(notes)}


def prepare_review(input_dir: Path, store: Store) -> dict:
    loaded = load_episode(input_dir)
    state = store.read()
    if not current_pass(state):
        raise ValueError("only a passing rollout can be reviewed")
    rollout = state["rollouts"][-1]
    merge = current_scene(store, state)
    check_source(json.loads((store.root / "episode.json").read_text()), Path(input_dir))
    model = mujoco.MjModel.from_xml_path(str(store.artifact(merge["scene_xml_path"])))
    data = mujoco.MjData(model)
    with np.load(store.artifact(rollout["rollout_path"])) as record:
        qpos = np.asarray(record["qpos"])
        ctrl = np.asarray(record["ctrl"])
        arm_speed = np.asarray(record["arm_speed_radps"])
        timestamps = np.asarray(record["timestamp_sec"])
    trajectory = state["trajectories"][-1]
    phases = json.loads(store.artifact(trajectory["source_spec_path"]).read_text())["phases"]
    rows = evidence_rows(phases, arm_speed, len(qpos))
    renderer = presentation_renderer(model, height=240, width=320)
    sheet = Image.new("RGB", (6 * 320, len(rows) * 240), "black")
    draw = ImageDraw.Draw(sheet)
    camera_names = ("top", "left_wrist", "right_wrist")
    simulation_cameras = loaded.runtime_contract["cameras"]
    for line, row in enumerate(rows):
        data.qpos[:] = qpos[row]
        data.ctrl[:] = ctrl[row]
        mujoco.mj_forward(model, data)
        for col, camera in enumerate(camera_names):
            real = Image.fromarray(_real_frame(loaded.camera_paths[camera], row))
            sheet.paste(real.resize((320, 240)), (col * 320, line * 240))
            renderer.update_scene(data, camera=str(simulation_cameras[camera]), scene_option=presentation_options())
            sim = Image.fromarray(renderer.render())
            sheet.paste(sim, ((col + 3) * 320, line * 240))
            draw.text((col * 320 + 6, line * 240 + 6), f"real {camera} row {row}", fill="yellow")
            draw.text(((col + 3) * 320 + 6, line * 240 + 6), f"sim {camera} row {row}", fill="yellow")
    renderer.close()
    bundle = store.root / "reviews"
    bundle.mkdir(parents=True, exist_ok=True)
    sheet_path = bundle / "comparison_keyframes.jpg"
    sheet.save(sheet_path, quality=90)
    diagnostics = json.loads(store.artifact(rollout["diagnostics_path"]).read_text())
    compact = {key: diagnostics[key] for key in (
        "status", "required_checks", "motion_quality", "first_failed_phase",
        "path_length_m", "net_xyyaw", "max_penetration_m", "reused_stages",
        "object_quality")
        if key in diagnostics}
    compact["stages"] = [{key: value for key, value in metric.items()
                           if key not in ("contacts", "contact_rows")}
                          for metric in diagnostics.get("stage_metrics", [])]
    clips = motion_review_clips(model, qpos, timestamps, phases, bundle / "motion_clips")
    packet = {"motion_clips": clips, "schema_version": "mobile-owner-review-v1",
              "episode_id": Path(input_dir).name,
              "scene_sha256": rollout["scene_sha256"],
              "rollout_revision": rollout["revision"],
              "keyframe_sheet": str(sheet_path),
              "rollout_comparison_sheet": str(store.artifact(rollout["comparison_sheet_path"])),
              "review_rows": rows,
              "rollout_diagnostics": compact,
              "full_diagnostics_path": str(store.artifact(rollout["diagnostics_path"])),
              **operation_relation_evidence(store.root),
              "questions": ["Watch the physical-speed motion clips; static frames cannot establish temporal quality.","Does the planar base follow the source route and stop near interaction points without striking task objects?",
                            "Verify initial/precontact support, grasp access and obstacle/clearance relations against source frames, using modeling notes only as claims. A changed relation that makes the operation easier without source evidence is blocking, even if physics passes. Inspect targeted source/context frames if keyframes are insufficient; mark uncertainty rather than inventing precision.",
                            "Do both arms move naturally through contact, release and withdrawal, without twisting, snaps or sustained self-contact?",
                            "Does the strategy omit redundant retries, including unexplained release/recontact after useful progress?",
                            "Does the actual contact mode (push contact or grasp) support the claimed action and object outcome?",
                            "Do manipulated objects remain upright and stable through later stages and at task end?",
                            "Assess asset-mounted camera framing separately from action quality."],
              "review_output_path": str(bundle / "owner_review.json")}
    packet_path = bundle / "review_packet.json"
    write_json(packet_path, packet)
    state["stage"] = "REVIEW_READY"
    store.save(state)
    return {"status": "PASS", "review_packet_path": str(packet_path.relative_to(store.root)),
            "keyframe_sheet_path": str(sheet_path.relative_to(store.root)),
            "review_output_path": str((bundle / "owner_review.json").relative_to(store.root))}


def submit_review(store: Store, request: dict) -> dict:
    state = store.read()
    if state["stage"] != "REVIEW_READY":
        raise ValueError("prepare review bundle first")
    path = store.artifact(request["review_json_path"])
    review = json.loads(path.read_text())
    if review.get("rollout_revision") != state["rollouts"][-1]["revision"]:
        raise ValueError("Name the rollout revision shown in the review packet")
    if review.get("verdict") not in ("PASS", "FAIL"):
        raise ValueError("review requires verdict PASS|FAIL")
    reason = review.get("reason")
    if not reason and review.get("evidence"):
        reason = "; ".join(str(item.get("finding", "")) for item in review["evidence"][:3])
    if not reason:
        raise ValueError("review requires reason or evidence findings")
    trajectory = state["trajectories"][-1]
    if sha256(store.artifact(trajectory["source_spec_path"])) != trajectory["source_spec_sha256"]:
        raise ValueError("Phase specification changed. Compile and replay the trajectory.")
    record = {"verdict": review["verdict"], "reason": reason,
              "path": request["review_json_path"], "sha256": sha256(path),
              "rollout_revision": state["rollouts"][-1]["revision"]}
    rollout = state["rollouts"][-1]
    paths = [rollout["rollout_path"], rollout["diagnostics_path"],
             state["trajectories"][-1]["source_spec_path"]]
    record["files"] = {rel: sha256(store.artifact(rel)) for rel in paths}
    state["reviews"].append(record)
    state["stage"] = "REVIEW_PASS" if review["verdict"] == "PASS" else "REVIEW_FAIL"
    store.save(state)
    return {"status": review["verdict"], "review": record}

def current_pass(state):
    if not state["rollouts"] or not state["merges"] or not state["trajectories"]:
        return False
    rollout = state["rollouts"][-1]
    return (rollout["status"] == "PASS"
            and state["stage"] in {"ROLLOUT_PASS", "REVIEW_READY", "REVIEW_FAIL", "REVIEW_PASS"}
            and rollout["trajectory_revision"] == state["trajectories"][-1]["revision"]
            and rollout["scene_sha256"] == state["merges"][-1]["scene_sha256"])


def accepted(store):
    state = store.read()
    if not current_pass(state) or state["stage"] != "REVIEW_PASS" or not state["reviews"]:
        return False
    current_scene(store, state)
    review = state["reviews"][-1]
    return (all(sha256(store.artifact(rel)) == digest for rel, digest in review["files"].items())
            and review["verdict"] == "PASS" and review["rollout_revision"] == state["rollouts"][-1]["revision"]
            and sha256(store.artifact(review["path"])) == review["sha256"])
