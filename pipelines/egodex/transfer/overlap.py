"""Overlap an outgoing idle-hand retreat with the other idle hand's approach."""

import numpy as np


def overlap_unloaded(arr, phases, model, arms):
    t = arr["time"]
    ctrl = arr["ctrl"]
    for a, b in zip(phases, phases[1:]):
        if not (a["phase"].endswith("retreat") and b["phase"].endswith("approach")):
            continue
        if any(x["mode"] != "idle" for p in (a, b) for x in p["hand_objects"].values()):
            continue
        start, mid, end = a["sim_start"], a["sim_end"], b["sim_end"]
        d1, d2 = mid - start, end - mid
        outgoing, receiving = None, None
        ctrls = {}
        joints = {}
        for side, arm in arms.items():
            ctrls[side] = [
                model.actuator(n).id
                for n in arm["actuators"] + [arm["gripper"]["actuator"]]
            ]
            joints[side] = [
                int(model.joint(n).qposadr[0])
                for n in arm["joints"] + arm["gripper"]["joints"]
            ]

        def movement(side, lo, hi):
            rows = ctrl[(t >= lo - 1e-8) & (t <= hi + 1e-8)][:, ctrls[side][:-1]]
            return np.max(np.ptp(rows, axis=0))

        outgoing = max(arms, key=lambda s: movement(s, start, mid))
        receiving = max(arms, key=lambda s: movement(s, mid, end))
        if (
            outgoing == receiving
            or movement(receiving, start, mid) > 1e-5
            or movement(outgoing, mid, end) > 0.05
        ):
            continue
        # Only combine after the released object no longer needs either gripper.
        delta = min(d1, d2)
        joined = end - delta
        first = t[(t >= start - 1e-8) & (t <= mid + 1e-8)]
        second = t[(t >= mid - 1e-8) & (t <= end + 1e-8)] - delta
        new = np.unique(
            np.round(
                np.r_[
                    t[t < start - 1e-8],
                    first,
                    second,
                    joined,
                    t[t > end + 1e-8] - delta,
                ],
                10,
            )
        )
        # Use smooth source-order time for the two non-contact intervals. Physical
        # contact events on either side keep their original phase identities.
        common = np.interp(
            new, [t[0], start, joined, t[-1] - delta], [t[0], start, end, t[-1]]
        )
        inside = (new >= start - 1e-8) & (new <= joined + 1e-8)
        original = {}
        for side in arms:
            clock = common.copy()
            if side == outgoing:
                # Finish any small outgoing-arm correction from the second phase
                # by weighting both component displacements over their own clocks.
                clock[inside] = np.minimum(new[inside], mid)
            else:
                clock[inside] = np.clip(new[inside] + delta, mid, end)
            original[side] = clock
        for key, columns in [("ctrl", ctrls), ("qpos_target", joints)]:
            old = arr[key]
            out = np.column_stack(
                [np.interp(common, t, old[:, j]) for j in range(old.shape[1])]
            )
            for side, ids in columns.items():
                for j in ids:
                    out[:, j] = np.interp(original[side], t, old[:, j])
                    if side == outgoing:
                        u = np.clip(new[inside] + delta, mid, end)
                        out[inside, j] += np.interp(u, t, old[:, j]) - np.interp(
                            mid, t, old[:, j]
                        )
            arr[key] = out
        arr["source_time"] = np.interp(common, t, arr["source_time"])
        arr["phase_index"] = np.array(
            [
                min(np.searchsorted([p["sim_end"] for p in phases], v), len(phases) - 1)
                for v in common
            ]
        )
        arr["time"] = new

        def shift(v):
            return float(
                np.interp(
                    v, [t[0], start, end, t[-1]], [t[0], start, joined, t[-1] - delta]
                )
            )

        for phase in phases:
            phase["sim_start"] = shift(phase["sim_start"])
            phase["sim_end"] = shift(phase["sim_end"])
            for event in phase.get("events", []):
                event["sim_time"] = shift(event["sim_time"])
            den = phase["source_end"] - phase["source_start"]
            if den > 0:
                phase["retime_factor"] = (phase["sim_end"] - phase["sim_start"]) / den
        return dict(
            outgoing=outgoing,
            receiving=receiving,
            phases=[a["phase"], b["phase"]],
            seconds_saved=delta,
        )
    return None
