"""Run a single Mobile ALOHA reconstruction owner on a prepared episode."""

import argparse
import json
import os
from pathlib import Path
import sys

from mobile_harness.server import TOOLS
from mobile_harness.store import Store
from mobile_harness.review import accepted
from mobile_harness.bindings import bind_source
from roborecast.episode_owner import run_owner

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--example", choices=["push-a-chair"])
    choice.add_argument("--task", help="Interaction to reconstruct from the source")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--minutes", type=float, default=40)
    parser.add_argument("--sandbox", choices=["workspace-write", "danger-full-access"], default="workspace-write")
    args = parser.parse_args()
    if not 0 < args.minutes <= 40:
        parser.error("Choose an episode budget between 0 and 40 minutes")
    episode, work = args.episode.resolve(), args.out.resolve()
    if not (episode / "workspace_context.json").is_file():
        parser.error("Prepare the source HDF5 first")
    operation = args.task
    if args.example:
        example = json.loads((HERE / "examples" / (args.example + ".json")).read_text())
        source = json.loads((episode / "specs/source.json").read_text())
        if source["source_sha256"] != example["source_sha256"]:
            parser.error("The example uses aloha_mobile_chair_truncated/episode_1.hdf5")
        operation = example["operation"]
    work.mkdir(parents=True, exist_ok=False)
    store = Store(work)
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(HERE.parents[1] / "src"), env.get("PYTHONPATH", "")])
    overrides = {
        "model_reasoning_effort": "medium",
        "mcp_servers.mobile.command": sys.executable,
        "mcp_servers.mobile.args": ["-m", "mobile_harness.server", "--input-dir", str(episode),
                                    "--reconstruction-dir", str(work)],
        "mcp_servers.mobile.tool_timeout_sec": 240,
        "mcp_servers.mobile.enabled_tools": sorted(TOOLS),
    }
    for key in ("PATH", "PYTHONPATH", "MUJOCO_GL", "LD_LIBRARY_PATH"):
        if key in env:
            overrides["mcp_servers.mobile.env." + key] = env[key]
    for name in TOOLS:
        overrides["mcp_servers.mobile.tools." + name + ".approval_mode"] = "approve"
    command = [args.agent, "exec", "--skip-git-repo-check", "--model", args.model]
    for key, value in overrides.items():
        command += ["-c", key + "=" + json.dumps(value)]
    command += ["--sandbox", args.sandbox, "-C", str(work)]
    prompt = (HERE / "owner.md").read_text().format(episode=episode, work=work,
                                                    guide=HERE / "references", task=operation,
                                                    python=sys.executable)
    (work / "owner_prompt.md").write_text(prompt)
    bind_source(work, episode)
    raise SystemExit(run_owner(command, env, work, prompt, args.minutes * 60, lambda: accepted(store)))


if __name__ == "__main__":
    main()
