"""Measure a payload and receptacle in the settled source and target configurations."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from roborecast.harness import initial_state, read
from roborecast.rollout import _geom_bounds_in_frame


def body_geoms(model, name):
    members = {model.body(name).id}
    for body, parent in enumerate(model.body_parentid):
        if parent in members:
            members.add(body)
    return [i for i, body in enumerate(model.geom_bodyid) if body in members]


def measure(bundle, roles):
    model = mujoco.MjModel.from_xml_path(str(bundle / "scene.xml"))
    data = initial_state(model)
    payload, bowl = roles["payload"], roles["receptacle"]
    fruit_geoms, bowl_geoms = body_geoms(model, payload), body_geoms(model, bowl)
    # Hidden measurement volumes do not contribute to visible bowl size.
    visible = [i for i in bowl_geoms if model.geom_rgba[i, 3] > 0 and model.geom_group[i] < 3]
    bounds = [_geom_bounds_in_frame(model, data, i, np.eye(3), np.zeros(3)) for i in visible]
    if not bounds:
        raise ValueError(f"No visible geometry under {bowl}")
    diameter = (np.max([hi for _, hi in bounds], axis=0) - np.min([lo for lo, _ in bounds], axis=0))[:2].max()
    gaps = [mujoco.mj_geomDistance(model, data, a, b, 2., None) for a in fruit_geoms for b in bowl_geoms
            if (model.geom_contype[a] & model.geom_conaffinity[b]) or (model.geom_contype[b] & model.geom_conaffinity[a])]
    if not gaps:
        raise ValueError("No collision-enabled fruit and bowl pair")
    vector = (data.body(bowl).xpos - data.body(payload).xpos)[:2]
    return {"payload_to_bowl_xy_m": vector.tolist(), "bowl_diameter_m": float(diameter),
            "center_spacing_m": float(np.linalg.norm(vector)), "start_clearance_m": float(min(gaps))}


def compare(source, target, recipe, bindings, roles=None):
    spec = recipe["layout_check"]
    if roles is None:
        source_bodies = read(source / "observables.json")["tail_speeds"]
        src = {role: source_bodies[key] for role, key in spec["source_observables"].items()}
        roles = {"source": src, "target": {role: bindings["entities"].get(name, name) for role, name in src.items()}}
    first, last = measure(source, roles["source"]), measure(target, roles["target"])
    u, v = [np.asarray(row["payload_to_bowl_xy_m"]) for row in (first, last)]
    shift = float(np.linalg.norm(v-u))
    angle = float(np.degrees(np.arctan2(abs(u[0]*v[1]-u[1]*v[0]), np.dot(u, v))))
    ratio = last["bowl_diameter_m"] / first["bowl_diameter_m"]
    low, high = spec["center_spacing_m"]
    dlo, dhi = spec["bowl_diameter_ratio"]
    checks = {"relative_layout": shift >= spec["min_vector_change_m"], "spacing": low <= last["center_spacing_m"] <= high,
              "comparable_bowl": dlo <= ratio <= dhi, "start_clearance": last["start_clearance_m"] >= spec["min_start_clearance_m"]}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "roles": roles,
            "source": first, "target": last, "vector_change_m": shift,
            "direction_change_deg": angle, "diameter_ratio": ratio, "checks": checks}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "recipe", "bindings"):
        p.add_argument(name, type=Path)
    p.add_argument("--roles", type=Path, help="Explicit source and target payload/receptacle body names")
    a = p.parse_args()
    result = compare(a.source, a.target, read(a.recipe), read(a.bindings), read(a.roles) if a.roles else None)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
