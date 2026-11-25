"""Minimal line-delimited MCP server exposing the v2.0 tools."""
from __future__ import annotations
import argparse
import json
import base64
from pathlib import Path
import sys
import time
from datetime import datetime,timezone
from .store import Store


def _array(items, **limits): return {"type": "array", "items": items, **limits}
NUMBER = {"type": "number"}
STRING = {"type": "string"}
INTEGER = {"type": "integer"}
V3 = _array(NUMBER, minItems=3, maxItems=3)
Q4 = _array(NUMBER, minItems=4, maxItems=4)
TARGET = {"type": "object", "properties": {"position": V3, "quaternion_wxyz": Q4}, "required": ["position"], "additionalProperties": False}
TARGETS = {"type": "object", "properties": {"left": TARGET, "right": TARGET}, "minProperties": 1, "additionalProperties": False}
EVENT = {"type": "object", "properties": {"source_time": NUMBER, "targets": TARGETS}, "required": ["source_time", "targets"], "additionalProperties": False}

# Description, fields, required fields. Hand positions are source world;
# robot Cartesian goals are explicit sim world, never implicit conversions.
SPECS = {
    "inspect_robot_runtime": ("Read explicit robot joint/pinch/camera conventions and gripper opening mapping; no source-code inspection needed.", {}, []),
    "render_execution_evidence": ("Render/cache 2x3 recorded keyframes; start/end defaults to an image sequence. Explicit format=video requests a <=8-second clip; no stepping or verdict changes.", {"record_id":STRING,"kind":{"enum":["rollout","stage_probe"]},"times":_array(NUMBER,maxItems=12),"start":NUMBER,"end":NUMBER,"format":{"enum":["keyframes","video"]},"frame_count":{"type":"integer","minimum":2,"maximum":12}},[]),
    "inspect_execution_state": ("Query recorded FK/object/gripper states and raw versus filtered contact forces at bounded timestamps, no simulation stepping.",
        {"record_id":STRING,"kind":{"enum":["rollout","stage_probe"]},"times":_array(NUMBER,minItems=1,maxItems=12),
         "time_domain":{"enum":["sim","source"]},"bodies":_array(STRING,maxItems=8),"side":{"enum":["left","right","both"]},"start":NUMBER,"end":NUMBER},[]),
    "inspect_human_reference": ("Query human wrists/declared hand-reference proxies, reliability and RGB overlays. Frames are exported indices; source_frames are separate original-recording indices. Source-world coordinates plus explicit rigid sim alignment; no object/contact ground truth.",
        {"frames": _array(INTEGER, maxItems=64), "times": _array(NUMBER, maxItems=64), "start": NUMBER, "end": NUMBER,
         "max_samples": {"type": "integer", "minimum": 1, "maximum": 64}, "overlay_count": {"type": "integer", "minimum": 0, "maximum": 8}, "joints": _array(STRING, maxItems=24)}, []),
    "inspect_scene_evidence": ("Query a selected RGB pixel against declared calibrated depth/static cloud (query_point), with provenance and validity. Conditional fallbacks: static triangulation or an explicitly assumed ray-plane intersection.",
        {"operation": {"enum": ["triangulate", "ray_plane", "query_point"]}, "static_point": {"type": "boolean"}, "correspondence_note": STRING,
         "observations": _array({"type": "object", "properties": {"frame": INTEGER, "pixel": _array(NUMBER, minItems=2, maxItems=2)}, "required": ["frame", "pixel"], "additionalProperties": False}, minItems=1, maxItems=32),
         "plane": {"type": "object", "properties": {"normal": V3, "offset": NUMBER, "evidence": STRING}, "required": ["normal", "offset", "evidence"], "additionalProperties": False},
         "frame": INTEGER, "pixel":_array(NUMBER,minItems=2,maxItems=2), "semantic_label":STRING,
         "target_kind":{"enum":["static_environment","moving_object"]}, "image_space":{"enum":["original","registered"]},
         "camera":STRING,"evidence_id":STRING,"image_path":STRING,"visibility_note":STRING}, ["operation"]),
    "merge_and_settle_scene": ("Merge registered geometry and settle task objects without the robot. Scene revisions invalidate placement and execution.",
        {"settle_seconds": NUMBER, "reuse_scene_id": STRING}, []),
    "render_scene_comparison": ("Render settled geometry externally and optionally from one moving source-camera frame. Compare source support/layout before adding robot.",
        {"source_frame": INTEGER}, []),
    "inspect_robot_placement": ("Check configured robot bundle x/y/yaw candidates against synchronized left/right event targets; render identical external views of initial and difficult event states. Discrete reach/collision tests do not certify grasp/path success.",
        {"prior_xyyaw": V3, "candidates_xyyaw": _array(V3, minItems=1, maxItems=12), "events": _array(EVENT, minItems=1),
         "initial_arm_qpos": _array(NUMBER, minItems=12, maxItems=12)}, ["prior_xyyaw", "events"]),
    "preview_robot_placement": ("Preview the selected candidate in a shared 2x3 detailed sheet; optional arm refinement reruns feasibility. Commit this exact pose.", {"preview_id":STRING,"candidate_id":STRING,"initial_arm_qpos":_array(NUMBER,minItems=12,maxItems=12)},["preview_id","candidate_id"]),
    "commit_robot_placement": ("Commit one exact current preview candidate after combined hold verification. Any adjustment needs a fresh inspection.",
        {"preview_id": STRING, "candidate_id": STRING, "selected_preview_id":STRING}, ["preview_id", "candidate_id"]),
    "compile_human_guided_trajectory": ("Compile sparse parked phases or one mobile transit from an actual reference. Probe stage, create_execution_reference, then compile next suffix; final rollout must cover full source range. See TOOL_REFERENCE.md.",
        {"trajectory_json_path": STRING, "reference_record_id":STRING}, ["trajectory_json_path"]),
    "simulate_task_rollout": ("Execute frozen controls from the committed initial state; record real contacts, task checks, motion and compact keyframe evidence. No object reset or attachment; mobile integrates frozen differential commands; failures remain evidence.",
        {"success_json_path": STRING, "reuse_checkpoints": {"type":"boolean"}}, ["success_json_path"]),
    "probe_contact": ("Advisory short contact physics from committed state; never task success or a production checkpoint.",
        {"side": {"enum": ["left", "right"]}, "object": STRING, "target_position": V3,
         "quaternion_wxyz": Q4, "gripper_opening": NUMBER, "duration": NUMBER, "approach_seconds": NUMBER,
         "close_seconds": NUMBER, "source_time": NUMBER, "minimum_contact_force": NUMBER,"reference_record_id":STRING}, ["side", "object", "target_position"]),
}


SPECS.update({
    "revalidate_robot_placement": ("After physics-only re-settle, retain prior chosen base/joints and rerun IK, collisions and initial hold without candidate renders; changed layout requires visual selection.",
        {"previous_placement_id":STRING},["previous_placement_id"]),
    "inspect_contact_interval": ("Summarize every saved physics step in a bounded interval: raw/filtered finger forces, contact gaps, slip and support interference. No simulation stepping or verdict changes.",
        {"record_id":STRING,"kind":{"enum":["rollout","stage_probe"]},"side":{"enum":["left","right"]},"body":STRING,
         "start":NUMBER,"end":NUMBER,"time_domain":{"enum":["sim","source"]}},["side","body","start","end"]),
})


class Server:
    def __init__(self, input_dir, reconstruction_dir):
        self.ctx = Store(reconstruction_dir, input_dir)
        from .inspection import inspect_robot_runtime, inspect_execution_state
        from .evidence_query import render_execution_evidence
        from .reference import inspect_human_reference
        from .geometry import inspect_scene_evidence
        from .scene import merge_and_settle_scene, render_scene_comparison
        from .placement import inspect_robot_placement, preview_robot_placement, commit_robot_placement, revalidate_robot_placement
        from .action_tools import inspect_contact_interval
        from .probes import probe_contact
        from .trajectory import compile_human_guided_trajectory
        from .rollout import simulate_task_rollout
        self.handlers = {name: fn for name, fn in locals().copy().items() if name in SPECS}

    def tools(self):
        return [{"name": name, "description": SPECS[name][0], "inputSchema": {
            "type": "object", "properties": SPECS[name][1], "required": SPECS[name][2], "additionalProperties": False}}
            for name in self.handlers]

    def call(self, name, arguments):
        if name not in self.handlers: raise ValueError(f"tool unavailable: {name}")
        _, properties, required = SPECS[name]
        if set(arguments) - set(properties): raise ValueError(f"unknown arguments: {set(arguments)-set(properties)}")
        for key in required:
            if key not in arguments: raise ValueError(f"missing argument: {key}")
        started, status = time.monotonic(), "ERROR"
        started_at=datetime.now(timezone.utc).isoformat()
        try:
            result = self.handlers[name](self.ctx, arguments)
            status = result.get("status", result.get("payload", {}).get("status", "OK"))
            return result
        finally:
            with self.ctx.path(".agent_run/tool_timings.jsonl").open("a") as out:
                out.write(json.dumps({"tool": name, "status": status, "started_at":started_at,"finished_at":datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-started})+"\n")


def evidence_images(ctx, result):
    """Embed a bounded number of returned sheets, so viewing needs no shell script."""
    paths=[]
    def visit(value, key=''):
        if isinstance(value,dict):
            for k,v in value.items():visit(v,k)
        elif isinstance(value,list):
            for v in value:visit(v,key)
        elif isinstance(value,str) and key in ('comparison_sheet','keyframes','keyframe_sheet','failure_sheet','media','visualization') and value.lower().endswith(('.png','.jpg','.jpeg')):
            p=Path(value)
            if not p.is_absolute():p=ctx.root/p
            if p.resolve().is_relative_to(ctx.root) and p.is_file() and p not in paths:paths.append(p)
    visit(result)
    limit = 9 if str(result.get('record_id','')).startswith('human_reference_evidence-') else 3
    return [{'type':'image','mimeType':'image/png' if p.suffix.lower()=='.png' else 'image/jpeg',
             'data':base64.b64encode(p.read_bytes()).decode()} for p in paths[:limit]]


def serve(server):
    def reply(identifier, result):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": identifier, "result": result}, allow_nan=False)+"\n")
        sys.stdout.flush()
    for line in sys.stdin:
        identifier = None
        try:
            message = json.loads(line); identifier = message.get("id"); method = message.get("method")
            if method == "initialize":
                reply(identifier, {"protocolVersion": message.get("params", {}).get("protocolVersion", "2025-06-18"),
                                   "capabilities": {"tools": {}}, "serverInfo": {"name": "human-scene-harness", "version": "2.0.0"}})
            elif method.startswith("notifications/"): continue
            elif method == "ping": reply(identifier, {})
            elif method == "tools/list": reply(identifier, {"tools": server.tools()})
            elif method == "tools/call":
                params = message.get("params", {})
                result = server.call(params["name"], params.get("arguments", {}))
                reply(identifier, {"content": [{"type": "text", "text": json.dumps(result, allow_nan=False)}, *evidence_images(server.ctx, result)]})
            else: raise ValueError(f"unknown MCP method: {method}")
        except Exception as exc:
            reply(identifier, {"isError": True, "content": [{"type": "text", "text": json.dumps({"status": "ERROR", "reason": str(exc), "type": type(exc).__name__})}]})
