"""Episode-scoped tools for a single fixed-base human-to-robot conversion owner."""

import argparse
import json
import re
from uuid import uuid4
import xml.etree.ElementTree as ET
from pathlib import Path

from fixed_robot.server import Server, serve
from fixed_robot.scene import register_scene_unit

TOOLS = {
    "inspect_human_reference",
    "inspect_scene_evidence",
    "merge_and_settle_scene",
    "render_scene_comparison",
    "inspect_robot_placement",
    "preview_robot_placement",
    "commit_robot_placement",
    "inspect_robot_runtime",
    "inspect_execution_state",
    "render_execution_evidence",
    "inspect_contact_interval",
    "revalidate_robot_placement",
    "probe_contact",
}
EXTRA = {
    "inspect_timing": (
        "Query a candidate's own inspect_contact_interval, inspect_execution_state or render_execution_evidence with the native arguments in json_text. Returned images and checks belong to this candidate.",
        {"candidate": "string", "tool": "string", "json_text": "string"},
    ),
    "optimize_timing": (
        "Re-execute the passing path at a faster, bounded clock. JSON settings: speedup, keep_grasps, grip_lead, grasp_speed, release_lead, release_speed, release_side, overlap. Returns physical evidence and sparse sheets. Parent is preserved.",
        {"json_text": "string"},
    ),
    "select_timing": (
        "After inspecting the candidate keyframes and contacts, select it by ID and matching structured PASS visual review. A failed candidate leaves the original available.",
        {"candidate": "string", "summary": "string"},
    ),
    "model_bowl": (
        "Build an open tapered bowl with a solid base and sloping wall collision segments. Supply json_text with name, position (sim-world base center), radii [bottom,rim], height, thickness, mass, rgba, friction [slide,twist,roll], movable and evidence. Units are metres.",
        {"json_text": "string"},
    ),
    "initial_masks": (
        "Batch SAM3 on one prepared initial RGB frame, inspect the mask sheet, save metric-camera RGB-only AgentSTAR captures. json_text is a list of objects with id, prompt and optional box_xyxy at original resolution.",
        {"json_text": "string"},
    ),
    "guide": ("Read the sparse waypoint and physical predicate contracts.", {}),
    "record_requirements": (
        "Record the visually observed task and operation window before rollout.",
        {"json_text": "string"},
    ),
    "model_unit": (
        "Create or revise an initial rigid object/support in sim-world metres. Supply role task_object/support, plain primitive-only MuJoCo XML as <mujoco><worldbody><body ...>... geometry ...</body></worldbody></mujoco>. Use a named freejoint for movable objects. For drawers use a named, passive slide joint on the moving drawer body. For bowls use open tapered walls built from ring segments; primitives only. Include the source-frame evidence and metric estimate.",
        {"unit_id": "string", "role": "string", "evidence": "string", "xml": "string"},
    ),
    "compile_plan": (
        "Compile complete two-arm action JSON using native paired VX300S IK and gripper mapping.",
        {"json_text": "string"},
    ),
    "run_physics": (
        "Evaluate a full physical rollout against a success JSON, with contact, ownership, duration and terminal checks.",
        {"json_text": "string"},
    ),
    "record_review": (
        "Record owner visual review for the current numeric rollout. Inspect returned keyframes first.",
        {"summary": "string"},
    ),
}
TAGS = {
    "mujoco",
    "compiler",
    "worldbody",
    "body",
    "geom",
    "freejoint",
    "joint",
    "inertial",
    "site",
}


class EpisodeTools:
    def __init__(self, episode, work):
        self.native = Server(episode, work)
        self.ctx = self.native.ctx

    def tools(self):
        result = [t for t in self.native.tools() if t["name"] in TOOLS]
        for name, (description, fields) in EXTRA.items():
            result.append(
                dict(
                    name=name,
                    description=description,
                    inputSchema=dict(
                        type="object",
                        properties={k: {"type": v} for k, v in fields.items()},
                        required=list(fields),
                        additionalProperties=False,
                    ),
                )
            )
        return result

    def call(self, name, args):
        if name in TOOLS:
            result = self.native.call(name, args)
            if (
                name == "render_scene_comparison"
                and result.get("source")
                and result.get("target")
            ):
                from PIL import Image
                from PIL import ImageOps

                sheet = Image.new("RGB", (1280, 360))
                for i, key in enumerate(("source", "target")):
                    with Image.open(self.ctx.path(result[key])) as im:
                        sheet.paste(
                            ImageOps.contain(im.convert("RGB"), (640, 360)),
                            (640 * i, 0),
                        )
                path = self.ctx.path(result["target"]).with_name(
                    "source_scene_comparison.jpg"
                )
                sheet.save(path)
                result["comparison_sheet"] = str(path.relative_to(self.ctx.root))
            return result
        if name not in EXTRA:
            raise ValueError("Unknown episode tool")
        if name == "inspect_timing":
            from transfer.selection import candidate

            child = candidate(self.ctx, args["candidate"], require_pass=False)
            tool = args["tool"]
            if tool not in {
                "inspect_contact_interval",
                "inspect_execution_state",
                "render_execution_evidence",
            }:
                raise ValueError("Expected a physical evidence query")
            result = Server(child.input_dir, child.root).call(
                tool, json.loads(args["json_text"])
            )

            # Native evidence paths are relative to the selected candidate.
            def scope(value):
                if isinstance(value, dict):
                    return {k: scope(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [scope(v) for v in value]
                if (
                    isinstance(value, str)
                    and value.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".npz", ".json")
                    )
                    and not Path(value).is_absolute()
                ):
                    return str(child.root / value)
                return value

            return scope(result)
        if name == "optimize_timing":
            from transfer.selection import propose

            return propose(self.ctx, json.loads(args["json_text"]))
        if name == "select_timing":
            from transfer.selection import candidate

            child = candidate(self.ctx, args["candidate"])
            assessment = json.loads(args["summary"])
            if assessment.get("verdict") != "PASS":
                raise ValueError("Only a visually accepted candidate may be selected")
            review = EpisodeTools(child.input_dir, child.root).call(
                "record_review", {"summary": args["summary"]}
            )
            return self.ctx.put(
                "timing_selection",
                {"candidate": args["candidate"], "review_id": review["id"]},
                dependencies=["rollout", "source_requirements"],
            )
        if name == "initial_masks":
            from transfer.masks import initial_masks

            return initial_masks(
                self.ctx.input_dir,
                self.ctx.path("source_masks"),
                json.loads(args["json_text"]),
            )
        if name == "guide":
            return {
                "contract": (Path(__file__).resolve().parents[1] / "motion.md").read_text(),
                "policy": "Use the current single owner for visual review. Base is fixed; no mobile actions.",
            }
        if name == "model_bowl":
            from transfer.bowl import build_bowl

            spec = json.loads(args["json_text"])
            uid = spec["name"]
            if not re.fullmatch(r"[a-zA-Z][A-Za-z0-9_]{0,63}", uid):
                raise ValueError("Bad bowl name")
            path = self.ctx.path(f"authored/{uid}_profile.xml")
            evidence = spec.pop("evidence")
            build_bowl(path, **spec)
            return self.call(
                "model_unit",
                dict(
                    unit_id=uid,
                    role="task_object",
                    evidence=evidence,
                    xml=path.read_text(),
                ),
            )
        if name == "model_unit":
            uid = args["unit_id"]
            if not re.fullmatch(r"[a-zA-Z][A-Za-z0-9_]{0,63}", uid):
                raise ValueError("Bad object name")
            fragment = ET.fromstring(args["xml"])
            if fragment.tag != "mujoco" or any(
                node.tag not in TAGS or "file" in node.attrib
                for node in fragment.iter()
            ):
                raise ValueError("Expected primitive-only body/worldbody geometry")
            draft = self.ctx.path(f"native/{uid}/{uuid4().hex}/unit.xml")
            ET.ElementTree(fragment).write(draft, encoding="unicode")
            from transfer.native import render_unit

            image = render_unit(
                self.ctx.input_dir, draft, draft.parent
            )
            result = register_scene_unit(
                self.ctx,
                {
                    "unit_id": uid,
                    "role": args["role"],
                    "relevance_reason": args["evidence"],
                    "evidence": {"source": args["evidence"]},
                    "fragment_xml_path": str(draft),
                },
            )
            return dict(
                unit_record=result.get("record_id", result.get("id")),
                comparison_sheet=str(image.relative_to(self.ctx.root)),
            )
        if name == "record_review":
            rollout = self.ctx.latest("rollout")
            if "source_requirements" not in rollout["dependencies"]:
                raise ValueError("Replay under the current interaction and collision checks before recording a new review")
            verdict = json.loads(args["summary"])
            from transfer.review import check

            verdict = check(self.ctx, rollout, verdict)
            # Freeze the actual evidence and operation bounds even when the summary is prose.
            from transfer.review import operation_window

            bound = {
                **operation_window(self.ctx),
                "diagnostics": rollout["payload"]["diagnostics_json"],
                "source_rgb": str(self.ctx.input_dir / "source/rgb.mp4"),
            }
            return self.ctx.put(
                "owner_visual_review",
                {"summary": verdict, "rollout_id": rollout["id"], "evidence": bound},
                dependencies=["rollout", "source_requirements"],
            )
        spec = json.loads(args["json_text"])
        if name == "record_requirements":
            from transfer.coverage import validate_events, validate_revision
            validate_events(spec.get("events"))
            if "source_requirements" in json.loads(self.ctx.state_path.read_text())["latest"]:
                previous = self.ctx.latest("source_requirements")["payload"]
                validate_revision(previous["events"], spec["events"])
        path = self.ctx.path(f"authored/requirements/{uuid4().hex}.json" if name == "record_requirements"
                             else "authored/" + name + ".json")
        path.write_text(json.dumps(spec, indent=2))
        if name == "record_requirements":
            return self.ctx.put(
                "source_requirements",
                spec,
                files=[str(path.relative_to(self.ctx.root))],
            )
        if name == "compile_plan":
            return self.native.call(
                "compile_human_guided_trajectory", {"trajectory_json_path": str(path)}
            )
        return self.native.call(
            "simulate_task_rollout", {"success_json_path": str(path)}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    serve(EpisodeTools(args.episode, args.work))
