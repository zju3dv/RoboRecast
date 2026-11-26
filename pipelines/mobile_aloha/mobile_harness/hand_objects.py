"""Per-hand contact ownership; optional, backward-compatible stage extension."""
import numpy as np
from .stages import object_geom, relative_pinch, object_tilt_deg, body_id


def bindings(phase):
    explicit = phase.get("hand_objects")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise ValueError("hand_objects must map left/right to object and mode")
        for side, item in explicit.items():
            if (side not in ("left", "right") or not isinstance(item, dict)
                    or not isinstance(item.get("object"), str)
                    or item.get("mode") not in ("grasp", "hold", "contact", "release")):
                raise ValueError("hand_objects requires left/right: {object, mode: grasp|hold|contact|release}")
        return explicit
    result = {}
    obj = phase.get("object")
    if not obj:
        return result
    for side in phase.get("grasp_sides", []):
        result[side] = {"object": obj, "mode": "grasp" if phase.get("intent") == "pinch_grasp" else "hold"}
    for side in phase.get("contact_sides", []):
        result[side] = {"object": obj, "mode": "contact"}
    if phase.get("intent") == "release":
        for side in ("left", "right"):
            result.setdefault(side, {"object": obj, "mode": "release"})
    return result


def normalize(phase):
    if "hand_objects" not in phase:
        return phase
    hand = bindings(phase)
    # Legacy primary-object metrics and semantic expansions retain their meaning.
    primary = phase.get("object")
    phase["grasp_sides"] = [s for s, b in hand.items()
                            if b["object"] == primary and b["mode"] in ("grasp", "hold")]
    phase["contact_sides"] = [s for s, b in hand.items()
                              if b["object"] == primary and b["mode"] == "contact"]
    return phase


def intended_contact(model, geom, label, hand):
    if label is None or label[0] not in hand:
        return False
    item = hand[label[0]]
    parts = ("left_pad", "right_pad") if item["mode"] == "release" else ("left_pad", "right_pad", "hand")
    return label[1] in parts and object_geom(model, geom, item["object"])


class HandObjectMonitor:
    def __init__(self, model, data, phase):
        self.phase = phase
        self.hand = bindings(phase)
        self.enabled = "hand_objects" in phase
        self.metrics = {}
        self.start = {}
        self.missing = {}
        for side, item in self.hand.items():
            obj = item["object"]
            body_id(model, obj)
            self.start[side] = relative_pinch(model, data, obj, side)
            self.missing[side] = 0
            self.metrics[side] = dict(object=obj, mode=item["mode"], pair_steps=0, contact_steps=0, steps=0,
                final_pair=False, max_contact_gap_sec=0., max_slip_m=0.,
                min_height_m=float(data.xpos[body_id(model, obj), 2]), max_tilt_deg=0.)

    def step(self, model, data, contacts):
        if not self.enabled:
            return
        # Object-level state is evaluated once, including shared two-hand payloads.
        objects = {b["object"] for b in self.hand.values()}
        state = {o: (float(data.xpos[body_id(model, o), 2]), object_tilt_deg(model, data, o))
                 for o in objects}
        for side, item in self.hand.items():
            metric = self.metrics[side]
            pair = {"left_pad", "right_pad"} <= contacts.get(side, set())
            metric["steps"] += 1
            metric["contact_steps"] += int(bool(contacts.get(side, set()) & {"left_pad", "right_pad", "hand"}))
            metric["pair_steps"] += int(pair)
            metric["final_pair"] = pair
            self.missing[side] = 0 if pair else self.missing[side] + 1
            metric["max_contact_gap_sec"] = max(metric["max_contact_gap_sec"],
                                                self.missing[side] * model.opt.timestep)
            metric["max_slip_m"] = max(metric["max_slip_m"], float(np.linalg.norm(
                relative_pinch(model, data, item["object"], side) - self.start[side])))
            height, tilt = state[item["object"]]
            metric["min_height_m"] = min(metric["min_height_m"], height)
            metric["max_tilt_deg"] = max(metric["max_tilt_deg"], tilt)

    def failures(self):
        if not self.enabled:
            return []
        failures = []
        for side, item in self.hand.items():
            m = self.metrics[side]
            prefix = f"{side}:{item['object']}"
            if item["mode"] in ("grasp", "hold") and not m["final_pair"]:
                failures.append(prefix + ":grasp_not_ready")
            if item["mode"] == "release" and m["final_pair"]:
                failures.append(prefix + ":release_not_complete")
            if item["mode"] == "contact" and m["contact_steps"] < .25 * m["steps"]:
                failures.append(prefix + ":hand_contact_insufficient")
            if item["mode"] == "hold":
                if m["max_contact_gap_sec"] > float(item.get("max_contact_gap_sec", .15)):
                    failures.append(prefix + ":grasp_lost")
                if m["max_slip_m"] > float(item.get("max_slip_m", self.phase.get("max_slip_m", .08))):
                    failures.append(prefix + ":grasp_slipped")
                if self.phase.get("action") == "carry_with_base" and m["min_height_m"] < float(item.get("carry_height_m", .15)):
                    failures.append(prefix + ":object_not_lifted")
            if item["mode"] in ("hold", "grasp") and "max_tilt_deg" in item and m["max_tilt_deg"] > float(item["max_tilt_deg"]):
                failures.append(prefix + ":object_tipped")
        return failures
