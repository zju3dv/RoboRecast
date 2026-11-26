"""Run one prepared EgoDex episode with a fixed two-arm VX300S target."""

import argparse
import json
import os
from pathlib import Path
import sys

from roborecast.episode_owner import ConnectionTiming, attempt_prompt, run_owner
from fixed_robot.store import Store
from transfer.selection import selected
from transfer.review import accepted

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--example", choices=["stack-bowls", "toy-in-drawer"])
    operation.add_argument("--task", help="Operation to verify against the demonstration")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--minutes", type=float, default=40)
    parser.add_argument("--sandbox", choices=["workspace-write", "danger-full-access"], default="workspace-write")
    args = parser.parse_args()
    root = HERE.parents[1]
    episode, work = args.episode.resolve(), args.out.resolve()
    if not (episode / "reference.npz").is_file():
        parser.error("Prepare RGB and hand references first")
    if args.example:
        example = json.loads((HERE / "examples" / (args.example + ".json")).read_text())
        meta = json.loads((episode / "meta.json").read_text())
        if meta["episode_id"] != example["episode_id"]:
            parser.error("Example and prepared episode differ")
        args.task = example["operation"]
    if not 0 < args.minutes <= 40:
        parser.error("Choose an episode budget between 0 and 40 minutes")
    budget = args.minutes * 60
    work.mkdir(exist_ok=False, parents=True)
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), str(root / "src"), env.get("PYTHONPATH")]))
    prompt = (
        (HERE / "owner.md")
        .read_text()
        .format(
            root=root,
            episode=episode,
            work=work,
            task=args.task,
            python=sys.executable,
        )
    )
    (work / "owner_prompt.md").write_text(prompt)
    mcp_args = [
        "-m",
        "transfer.tools",
        "--episode",
        str(episode),
        "--work",
        str(work),
    ]
    tool_env = {k: env[k] for k in ("PYTHONPATH", "MUJOCO_GL", "PATH", "SAM3_PYTHON", "SAM3_CHECKPOINT", "BLENDER", "LD_LIBRARY_PATH") if k in env}
    overrides = {
        "mcp_servers.egodex.command": sys.executable,
        "mcp_servers.egodex.args": mcp_args,
        "mcp_servers.egodex.tool_timeout_sec": 180,
        **{"mcp_servers.egodex.env." + k: v for k, v in tool_env.items()},
    }
    from transfer.tools import TOOLS, EXTRA

    allowed = sorted(TOOLS | set(EXTRA))
    overrides["mcp_servers.egodex.enabled_tools"] = allowed
    for tool in allowed:
        overrides["mcp_servers.egodex.tools." + tool + ".approval_mode"] = "approve"
    command = [args.agent, "exec", "--skip-git-repo-check", "--model", args.model,
               "-c", 'model_reasoning_effort="medium"'] + [
        item for k, v in overrides.items() for item in ("-c", k + "=" + json.dumps(v))
    ]
    command += ["--sandbox", args.sandbox, "-C", str(work)]
    def is_accepted():
        if (work / ".harness/state.json").exists():
            try:
                accepted(selected(Store(work)))
                return True
            except (KeyError, ValueError, OSError):
                pass
        return False

    raise SystemExit(run_owner(command, env, work, prompt, budget, is_accepted))


if __name__ == "__main__":
    main()
