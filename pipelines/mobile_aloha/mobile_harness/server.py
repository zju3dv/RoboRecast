from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .preflight import inspect_articulation_sweep, probe_contact, probe_reachability
from .review import prepare_review, submit_review
from .rollout import probe_mobile_stage, simulate_rollout
from .scene import merge_and_settle, register_modeling_unit, render_scene_preview
from .store import Store
from .operation_reference import operation_reference
from .placement import suggest_placement
from .trajectory import compile_trajectory


TOOLS = {
    "inspect_operation_reference": ("Batch source frames, optional normalized crops and source FK hand poses. Read operation-reference.md for coordinates and placement usage.", ["samples"]),
    "suggest_object_placement": ("Advisory object poses from a source contact, explicit local geometry and optional planar support; never mutates scene.", ["row", "side", "local_contact_point_m", "object_quaternions_wxyz"]),
    "register_modeling_unit": ("Register one task-relevant MuJoCo fragment; pose is {xyyaw:[x,y,yaw],z?:metres}.",
                               ["unit_id", "fragment_xml_path"]),
    "merge_and_settle_scene": ("Merge units; robot_xyyaw is the initial planar pose, settle_seconds defaults to 0.5.", []),
    "render_mobile_scene_preview": ("Render settled top and tightly framed task overview; optional focus_body.", []),
    "inspect_articulation_sweep": ("Advisory hinge/slide sweep with collision pairs and a three-pose preview.", ["joint_name"]),
    "probe_reachability": ("Advisory arm target reach and collision check from the settled scene.", ["side", "target_position_m"]),
    "probe_contact": ("Advisory short hand/object contact probe from the settled scene.", ["side", "object", "target_position_m"]),
    "compile_mobile_trajectory": ("Compile sparse dual-arm end-effector waypoints and planar base stages into frozen contract-driven targets.",
                                  ["trajectory_json_path"]),
    "simulate_mobile_rollout": ("Restartable stage rollout; returns first failed stage, contact/grasp/collision diagnostics and local sheet.",
                                ["success_json_path"]),
    "probe_mobile_stage": ("Non-authoritative local rollout through one named phase; reuses valid prefix checkpoints.",
                           ["success_json_path", "phase_name"]),
    "prepare_review_bundle": ("Make an owner review packet with contact/release keyframes after PASS.", []),
    "record_review": ("Record the owner’s visual verdict from a JSON artifact.", ["review_json_path"]),
}


def _schema(name: str, required: list[str]) -> dict:
    properties = {field: {"type": "string"} for field in required}
    if name == "inspect_operation_reference":
        properties["samples"] = {"type":"array", "minItems":1, "maxItems":12, "items":{
            "type":"object", "properties":{"row":{"type":"integer","minimum":0},
            "camera":{"type":"string","enum":["top","left_wrist","right_wrist"]},
            "box_xyxy_normalized":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":4,"maxItems":4},
            "padding":{"type":"number","minimum":0,"maximum":.5}},"required":["row","camera"],"additionalProperties":False}}
    elif name == "suggest_object_placement":
        vector={"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}
        properties.update(row={"type":"integer","minimum":0},side={"type":"string","enum":["left","right"]},
            local_contact_point_m=vector,local_support_point_m=vector,support_height_m={"type":"number"},
            object_quaternions_wxyz={"type":"array","minItems":1,"maxItems":4,"items":{
                "type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}})
    elif name == "register_modeling_unit":
        properties.update({"pose": {"type": "object", "properties": {
            "xyyaw": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "z": {"type": "number"}}, "required": ["xyyaw"]},
            "role": {"type": "string"}})
    elif name == "merge_and_settle_scene":
        properties.update({"robot_xyyaw": {"type": "array", "items": {"type": "number"},
                                              "minItems": 3, "maxItems": 3},
                           "settle_seconds": {"type": "number"},
                           "reuse_settled_state_revision": {"type": "integer", "minimum": 1}})
    elif name == "simulate_mobile_rollout":
        properties["success_revision_path"] = {"type": "string"}
    elif name == "compile_mobile_trajectory":
        properties["calibrate_after_phase"] = {"type": "string"}
        properties["calibrate_after_phases"] = {
            "type": "array", "items": {"type": "string"}, "minItems": 1}
        properties["compile_through_phase"] = {"type": "string"}
    elif name == "probe_mobile_stage":
        properties["stop_at_first_failure"] = {"type": "boolean"}
    elif name == "render_mobile_scene_preview":
        properties["focus_body"] = {"type": "string"}
        properties["reference_crop_path"] = {"type": "string"}
        properties["detail_geoms"] = {"type": "array", "items": {"type": "string"}, "maxItems": 8}
    elif name == "inspect_articulation_sweep":
        properties.update({"joint_positions": {"type": "array", "items": {"type": "number"}, "minItems": 3},
                           "sample_count": {"type": "integer", "minimum": 3, "maximum": 21},
                           "render": {"type": "boolean"}})
    elif name in ("probe_reachability", "probe_contact"):
        properties.update({"target_position_m": {"type": "array", "items": {"type": "number"},
                                                  "minItems": 3, "maxItems": 3},
                           "base_xyyaw": {"type": "array", "items": {"type": "number"},
                                          "minItems": 3, "maxItems": 3}})
        if name == "probe_contact":
            properties.update({"gripper_opening": {"type": "number", "minimum": 0, "maximum": 1},
                               "duration_sec": {"type": "number", "minimum": .01, "maximum": 1}})
    return {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}


class Server:
    def __init__(self, input_dir: Path, reconstruction_dir: Path):
        self.input_dir = Path(input_dir).resolve()
        self.store = Store(reconstruction_dir)
        self.handlers = {
            "inspect_operation_reference": lambda q: operation_reference(self.input_dir, self.store, q),
            "suggest_object_placement": lambda q: suggest_placement(self.input_dir, self.store, q),
            "register_modeling_unit": lambda q: register_modeling_unit(self.store, q),
            "merge_and_settle_scene": lambda q: merge_and_settle(self.input_dir, self.store, q),
            "render_mobile_scene_preview": lambda q: render_scene_preview(self.store, q),
            "inspect_articulation_sweep": lambda q: inspect_articulation_sweep(self.store, q),
            "probe_reachability": lambda q: probe_reachability(self.store, q),
            "probe_contact": lambda q: probe_contact(self.store, q),
            "compile_mobile_trajectory": lambda q: compile_trajectory(self.input_dir, self.store, q),
            "simulate_mobile_rollout": lambda q: simulate_rollout(self.input_dir, self.store, q),
            "probe_mobile_stage": lambda q: probe_mobile_stage(self.input_dir, self.store, q),
            "prepare_review_bundle": lambda q: prepare_review(self.input_dir, self.store),
            "record_review": lambda q: submit_review(self.store, q),
        }

    def tools(self) -> list[dict]:
        return [{"name": name, "description": desc, "inputSchema": _schema(name, required)}
                for name, (desc, required) in TOOLS.items()]

    def call(self, name: str, arguments: dict) -> dict:
        if name not in self.handlers:
            raise ValueError(f"unknown tool: {name}")
        for key in TOOLS[name][1]:
            if key not in arguments:
                raise ValueError(f"missing {key}")
        started = time.monotonic()
        started_utc = datetime.now(timezone.utc).isoformat()
        status = "PASS"
        try:
            result = self.handlers[name](arguments)
            status = str(result.get("status", "PASS"))
            return result
        except Exception:
            status = "ERROR"
            raise
        finally:
            path = self.store.root / ".agent_run/tool_timings.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps({"tool": name, "started_utc": started_utc,
                                         "elapsed_sec": round(time.monotonic() - started, 3),
                                         "status": status}, ensure_ascii=False) + "\n")


def _reply(identifier, value=None, error=None) -> None:
    message = {"jsonrpc": "2.0", "id": identifier}
    message["error" if error is not None else "result"] = error if error is not None else value
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(server: Server) -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            method, identifier = message.get("method"), message.get("id")
            if method == "initialize":
                _reply(identifier, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                    "serverInfo": {"name": "mobile-scene-harness", "version": "2.2"}})
            elif method == "notifications/initialized":
                continue
            elif method == "ping":
                _reply(identifier, {})
            elif method == "tools/list":
                _reply(identifier, {"tools": server.tools()})
            elif method == "tools/call":
                params = message.get("params", {})
                try:
                    result = server.call(params.get("name", ""), params.get("arguments", {}))
                    _reply(identifier, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})
                except Exception as exc:
                    _reply(identifier, {"content": [{"type": "text", "text": json.dumps({
                        "status": "ERROR", "type": type(exc).__name__, "message": str(exc)},
                        ensure_ascii=False)}], "isError": True})
            elif identifier is not None:
                _reply(identifier, error={"code": -32601, "message": f"unknown method {method}"})
        except Exception as exc:
            _reply(None, error={"code": -32700, "message": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    args = parser.parse_args()
    serve(Server(args.input_dir, args.reconstruction_dir))


if __name__ == "__main__":
    main()
