"""Map measured source predicates onto named target entities and spatial anchors."""
import argparse
import copy
import json
from pathlib import Path

from roborecast.harness import initial_state, load_scene


def bind(source, target, mapping):
    contract = json.loads((source / "requirements.json").read_text())
    entities = mapping.get("entities", {})
    anchors = mapping.get("anchors", {})
    states = [initial_state(load_scene(path / "scene.xml", {})) for path in (source, target)]

    def predicate(row, location):
        row = copy.deepcopy(row)
        row["key"] = ":".join(entities.get(token, token) for token in row["key"].split(":"))
        if location in anchors:
            anchor = anchors[location]
            if row["type"] not in ("initial", "final") or not row["key"].endswith(":position"):
                raise ValueError(f"Only world position bounds can follow an anchor at {location}")
            delta = (states[1].body(entities.get(anchor, anchor)).xpos - states[0].body(anchor).xpos)
            for key in ("min", "max"):
                if key in row:
                    row[key] = (delta[row["component"]] + row[key]).tolist() if "component" in row else (delta + row[key]).tolist()
        return row

    for event in contract["I"]:
        event["roles"] = {key: entities.get(value, value) for key, value in event["roles"].items()}
        event["predicates"] = [predicate(row, f"{event['id']}/{i}") for i, row in enumerate(event["predicates"])]
    for group in ("G", "c"):
        contract[group] = [predicate(row, f"{group}/{i}") for i, row in enumerate(contract[group])]
    return contract


def physical_contract(source, mapping):
    """Carry numerical tests over verbatim, rebinding only referenced entities."""
    entities = mapping.get("entities", {})
    def entity(name):
        return entities.get(name, name)

    physics = json.loads((source / "success.json").read_text())
    for row in physics["required_checks"] + physics["diagnostic_checks"]:
        for key in ("body", "joint", "geom", "volume", "geom_a", "geom_b"):
            if key in row:
                row[key] = entity(row[key])
        if "bodies" in row:
            row["bodies"] = [entity(name) for name in row["bodies"]]
    obs = json.loads((source / "observables.json").read_text())
    obs["passive_roots"] = [entity(name) for name in obs["passive_roots"]]
    obs["tail_speeds"] = {":".join(entity(token) for token in key.split(":")): entity(body)
                          for key, body in obs.get("tail_speeds", {}).items()}
    return physics, obs


def verify_physical(source, target, mapping):
    physics, obs = physical_contract(source, mapping)
    actual = json.loads((target / "success.json").read_text())
    for check in physics["required_checks"]:
        if check not in actual["required_checks"]:
            raise ValueError(f"Missing or changed required physical check: {check}")
    measured = json.loads((target / "observables.json").read_text())
    if not set(obs["passive_roots"]).issubset(measured["passive_roots"]):
        raise ValueError("Mapped passive body missing from target observables")
    for key, body in obs["tail_speeds"].items():
        if measured.get("tail_speeds", {}).get(key) != body:
            raise ValueError(f"Target observable {key} must measure {body}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source_bundle", "target_bundle", "bindings", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--initialize-physics", action="store_true",
                   help="Initialize target success.json and observables.json from source bindings")
    a = p.parse_args()
    mapping = json.loads(a.bindings.read_text())
    if a.initialize_physics:
        physics, observations = physical_contract(a.source_bundle, mapping)
        for name, data in (("success", physics), ("observables", observations)):
            (a.target_bundle / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    a.output.write_text(json.dumps(bind(a.source_bundle, a.target_bundle, mapping), indent=2) + "\n")
