"""Measured gripper contact force summaries; diagnostic only, never acceptance."""
import math
import mujoco
import numpy as np
from .robot_runtime import runtime_contract


class ContactForceMonitor:
    def __init__(self, model):
        self.model = model
        self.geom_sides = {}
        arms = runtime_contract(model)["arms"]
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            while body:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                side = next((s for s in ("left", "right") if name in
                             ([arms[s]["eef_body"]] + arms[s]["gripper"]["finger_bodies"])), None)
                if side:
                    self.geom_sides[geom] = side
                    break
                body = int(model.body_parentid[body])
        self.phases = {s: {} for s in ("left", "right")}
        self.bins = {s: {} for s in ("left", "right")}
        self.invalid = 0
        self.samples = 0
        self.end = 0.

    def _target(self, geom, name):
        if not name:
            return False
        body = int(self.model.geom_bodyid[geom])
        while body:
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) == name:
                return True
            body = int(self.model.body_parentid[body])
        return False

    def step(self, data, source_time, phase):
        self.samples += 1
        self.end = float(data.time)
        phase_name = phase["phase"] if phase else "terminal_settle"
        rows = {s: {"sim_time": self.end, "source_time": float(source_time),
                    "phase": phase_name, "total_normal_force_N": 0., "contact_count": 0,
                    "target_normal_force_N": 0., "environment_normal_force_N": 0.,
                    "max_penetration_m": 0., "contacts": []} for s in self.phases}
        for index in range(data.ncon):
            contact = data.contact[index]
            geoms = (int(contact.geom1), int(contact.geom2))
            sides = {self.geom_sides[g] for g in geoms if g in self.geom_sides}
            if not sides:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            normal = float(force[0])
            if not math.isfinite(normal):
                self.invalid += 1
                continue
            # Include zero-force contacts; no distance or minimum-force filter.
            for side in sides:
                hand_index = next(i for i, g in enumerate(geoms) if self.geom_sides.get(g) == side)
                hand, other = geoms[hand_index], geoms[1-hand_index]
                target = self._target(other, (phase or {}).get("hand_objects", {}).get(side, {}).get("object"))
                row = rows[side]
                row["total_normal_force_N"] += normal
                row["contact_count"] += 1
                row["target_normal_force_N" if target else "environment_normal_force_N"] += normal
                row["max_penetration_m"] = max(row["max_penetration_m"], max(0., -float(contact.dist)))
                row["contacts"].append({"hand_geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, hand) or str(hand),
                                        "other_geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other),
                                        "other_body": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[other])) or "world",
                                        "normal_force_N": normal, "target": target})
        for side, row in rows.items():
            row["contacts"].sort(key=lambda c: c["normal_force_N"], reverse=True)
            row["contacts_omitted"] = max(0, len(row["contacts"])-8)
            row["contacts"] = row["contacts"][:8]
            for mapping, key in ((self.phases[side], phase_name), (self.bins[side], int(self.end/.2))):
                if key not in mapping or row["total_normal_force_N"] > mapping[key]["total_normal_force_N"]:
                    mapping[key] = row

    def finish(self):
        hands, windows = {}, []
        for side in self.phases:
            peaks = list(self.phases[side].values())
            hands[side] = {"peak": max(peaks, key=lambda p: p["total_normal_force_N"], default=None), "phase_peaks": peaks}
            chosen = []
            for peak in sorted(self.bins[side].values(), key=lambda p: p["total_normal_force_N"], reverse=True):
                t = peak["sim_time"]
                if peak["total_normal_force_N"] <= 0 or any(abs(t-x) <= .2 for x in chosen):
                    continue
                chosen.append(t)
                windows.append({"start": max(0., t-.1), "end": min(self.end, t+.1), "peak_time": t,
                                "reason": f"{side} gripper contact force peak", "peak_times": [t]})
                if len(chosen) == 2:
                    break
        merged = []
        for window in sorted(windows, key=lambda w: w["start"]):
            if merged and window["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], window["end"])
                merged[-1]["peak_times"] += window["peak_times"]
                merged[-1]["reason"] += "; " + window["reason"]
            else:
                merged.append(window.copy())
        return {"hands": hands, "evidence_windows": merged, "physics_samples": self.samples,
                "nonfinite_contacts_omitted": self.invalid,
                "note": "Diagnostic only; no force acceptance cap. Every physics step uses raw mj_contactForce normal forces in N, without distance/minimum-force filtering; not actuator force or net wrench. Only gripper/finger body subtrees count, each contact once per hand. Contacts list contains the largest 8 at each peak; totals include all contacts. Target means phase-bound object subtree, other contacts are environment (including robot/self). Windows select up to two separated 0.2-second-bin peaks per hand, with +/-0.1s context; merged windows retain all peak_times."}
