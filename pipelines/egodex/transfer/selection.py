"""Offer independently replayed timing candidates to the current episode owner."""

import json
from pathlib import Path
import time
from uuid import uuid4

from fixed_robot.store import Store
from transfer.retime import trial


def propose(ctx, settings):
    allowed = {
        "speedup",
        "keep_grasps",
        "grip_lead",
        "grasp_speed",
        "release_lead",
        "release_speed",
        "release_side",
        "overlap",
    }
    if set(settings) - allowed:
        raise ValueError("Unexpected timing setting")
    folder = ctx.path("timing_trials/" + uuid4().hex)
    start = time.monotonic()
    result = trial(ctx, folder, **{"speedup": 1.8, **settings})
    result["wall_seconds"] = time.monotonic() - start
    result["candidate"] = folder.name
    (folder / "timing_trial.json").write_text(json.dumps(result, indent=2))
    if "diagnostics" in result:
        diag = Path(result["diagnostics"])
        result["first_failure"] = json.loads(diag.read_text()).get("first_failure")
        sheets = sorted((diag.parent / "evidence").glob("keyframes_*.jpg"))
        if sheets:
            result["keyframes"] = [
                str(sheets[i]) for i in sorted({0, len(sheets) // 2, len(sheets) - 1})
            ]
    return result


def candidate(ctx, name, require_pass=True):
    if (
        not isinstance(name, str)
        or len(name) != 32
        or any(c not in "0123456789abcdef" for c in name)
    ):
        raise ValueError("Invalid timing candidate")
    folder = ctx.path("timing_trials/" + name)
    report = json.loads((folder / "timing_trial.json").read_text())
    current = ctx.latest("rollout")["id"]
    child = Store(folder, ctx.input_dir)
    rr = child.latest("rollout")
    if report.get("parent_rollout") != current or report.get("rollout_id") != rr["id"]:
        raise ValueError("Timing candidate is stale")
    if (
        child.latest("source_requirements")["id"]
        != ctx.latest("source_requirements")["id"]
    ):
        raise ValueError("Candidate source requirements changed")
    if require_pass and (
        report["status"] != "PASS"
        or not rr["payload"]["authoritative"]
        or rr["payload"]["status"] != "PASS"
    ):
        raise ValueError("Candidate failed physical evaluation")
    return child


def selected(ctx):
    state = json.loads(ctx.state_path.read_text())
    if "timing_selection" not in state["latest"]:
        return ctx
    # A newer parent rollout invalidates the old choice.
    relative = state["latest"]["timing_selection"]
    if (
        json.loads(ctx.path(relative).read_text())["dependencies"]["rollout"]
        != ctx.latest("rollout")["id"]
    ):
        return ctx
    record = ctx.latest("timing_selection")["payload"]
    child = candidate(ctx, record["candidate"])
    if child.latest("owner_visual_review")["id"] != record["review_id"]:
        raise ValueError("Selected timing candidate review changed")
    return child
