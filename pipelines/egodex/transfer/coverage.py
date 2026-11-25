"""Match source interaction requirements to the predicates actually executed."""

import math

import numpy as np


def validate_events(events):
    if not isinstance(events, list) or not events:
        raise ValueError("Record source events with IDs and required_checks")
    ids = set()
    for event in events:
        key = event.get("id")
        if not isinstance(key, str) or not key.strip() or key in ids:
            raise ValueError("Each source event needs a unique nonempty id")
        ids.add(key)
        start, end = event.get("start"), event.get("end")
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)) or start > end:
            raise ValueError(f"{key}: invalid source interval")
        checks = event.get("required_checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"{key}: supply required_checks")
        for check in checks:
            scope = check.get("scope")
            if scope == "phase":
                if check.get("side") not in ("left", "right") or check.get("mode") not in ("idle", "grasp", "hold", "contact", "release"):
                    raise ValueError(f"{key}: phase checks need side and mode")
                if check.get("mode") != "idle" and not check.get("object"):
                    raise ValueError(f"{key}: specify the phase object")
            elif scope not in ("event", "terminal") or not check.get("kind") or not check.get("body"):
                raise ValueError(f"{key}: expected an event or terminal predicate with kind and body")


def validate_revision(previous, events):
    """Keep established obligations; allow a visually justified component rebinding."""
    current = {event["id"]: event for event in events}
    for old in previous:
        if "required_checks" not in old:  # Initial migration from a descriptive episode.
            continue
        key = old["id"]
        if key not in current:
            raise ValueError(f"Missing previously recorded interaction: {key}")
        new = current[key]
        available = list(new["required_checks"])
        for check in old["required_checks"]:
            if check in available:
                available.remove(check)
                continue
            # The owner can refine which component was contacted after inspecting the source.
            binding = {"body", "object", "geom", "support", "support_geom", "floor", "joint", "relative_to"}
            matches = [other for other in available if set(other) == set(check)
                       and all(other[k] == v for k, v in check.items() if k not in binding)]
            if not matches or not str(new.get("binding_evidence") or "").strip():
                raise ValueError(f"{key}: preserve the required check or supply evidence for a component rebinding")
            available.remove(matches[0])


def check_coverage(events, spec, phases):
    validate_events(events)
    coverage = {}
    missing = []
    for event in events:
        sources = {"terminal": [("terminal", c) for c in spec.get("terminal_checks", [])],
                   "event": [], "phase": []}
        for item in [*spec.get("events", []), *phases]:
            lo = max(event["start"], item["source_start"])
            hi = min(event["end"], item["source_end"])
            if hi < lo or (hi == lo and item["source_start"] != item["source_end"] and event["start"] != event["end"]):
                continue
            if "hand_objects" in item:
                for side, binding in item["hand_objects"].items():
                    sources["phase"].append((item["phase"], dict(binding, side=side)))
            else:
                sources["event"].extend((item.get("name", "event"), c) for c in item["checks"])
        matched = []
        for expected in event["required_checks"]:
            fields = {k: v for k, v in expected.items() if k != "scope"}
            refs = [ref for ref, actual in sources[expected["scope"]]
                    if all(k in actual and actual[k] == v for k, v in fields.items())]
            if refs:
                matched.append({"check": expected, "matched": refs})
            else:
                missing.append(f"{event['id']}: {expected}")
        coverage[event["id"]] = matched
    if missing:
        raise ValueError("Missing interaction evidence: " + " | ".join(missing))
    return coverage

def check_idle_commands(events, phases, source_time, controls, actuator_indices):
    """An idle role must also keep the declared arm and gripper commands still."""
    for event in events:
        for check in event["required_checks"]:
            if check.get("scope") != "phase" or check.get("mode") != "idle":
                continue
            side = check["side"]
            start, end = event["start"], event["end"]
            samples = (source_time >= start - 1e-8) & (source_time <= end + 1e-8)
            active = any(p["source_start"] < end and start < p["source_end"]
                         and p["hand_objects"][side]["mode"] != "idle" for p in phases)
            if active or not samples.any() or np.max(np.ptp(controls[samples][:, actuator_indices[side]], axis=0)) > 1e-6:
                raise ValueError(f"{event['id']}: {side} arm commands contradict the declared idle interval")
