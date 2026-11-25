"""Keep the owner's source comparison attached to its actual physical rollout."""

import json
import re

import numpy as np


def check(ctx, rollout, assessment):
    if isinstance(assessment, str):
        try:
            assessment = json.loads(assessment)
        except ValueError:
            # Early owner sessions embedded their explicit ID and review in one sentence.
            match = re.match(r"^(PASS|FAIL) for (rollout-[a-z0-9]+)\b", assessment)
            if not match:
                raise ValueError("Missing explicit visual verdict")
            assessment = {
                "verdict": match.group(1),
                "rollout_id": match.group(2),
                "visual_review": assessment,
            }
    if (
        not isinstance(assessment, dict)
        or assessment.get("rollout_id") != rollout["id"]
        or assessment.get("verdict") not in ("PASS", "FAIL")
    ):
        raise ValueError("Review must identify its rollout and verdict")
    requirements = ctx.latest("source_requirements")
    if requirements["created_at"] > rollout["created_at"]:
        raise ValueError("Source requirements changed after this rollout")
    if assessment["verdict"] == "FAIL":
        return assessment
    if not (
        rollout["payload"]["authoritative"] and rollout["payload"]["status"] == "PASS"
    ):
        raise ValueError("Authoritative physical pass required")
    evidence = assessment.get("evidence")
    nested = evidence if isinstance(evidence, dict) else {}
    visual = (
        assessment.get("visual_review")
        or nested.get("visual_review")
        or nested.get("visual_findings")
    )
    if not visual:
        raise ValueError(
            "Describe the source and simulated sparse frames that you inspected"
        )
    return assessment


def accepted(ctx):
    rollout = ctx.latest("rollout")
    review = ctx.latest("owner_visual_review")
    if review["payload"]["rollout_id"] != rollout["id"]:
        raise ValueError("Visual review belongs to a different rollout")
    summary = check(ctx, rollout, review["payload"]["summary"])
    if summary["verdict"] != "PASS":
        raise ValueError("Visual review did not pass")
    return rollout, review, summary


def operation_window(ctx):
    req = ctx.latest("source_requirements")["payload"]
    trajectory = ctx.latest("trajectory")["payload"]
    phases = trajectory["time_map"]
    observed = req.get("source_operation_window") or {
        "start": req.get("source_active_start", phases[0]["source_start"]),
        "end": req.get("source_active_end", phases[-1]["source_end"]),
    }
    source = [float(observed[k]) for k in ("start", "end")]
    with np.load(ctx.path(trajectory["trajectory_npz"])) as path:
        if (
            source[0] >= source[1]
            or source[0] < path["source_time"][0] - 1e-6
            or source[1] > path["source_time"][-1] + 1e-6
        ):
            raise ValueError("Source operation is outside the compiled path")
        sim = np.interp(source, path["source_time"], path["time"]).tolist()
    return dict(
        source_window=observed,
        corresponding_sim_seconds=sim,
        source_duration_seconds=source[1] - source[0],
        compiled_duration_seconds=sim[1] - sim[0],
    )
