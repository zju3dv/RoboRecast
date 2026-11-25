"""Run one prepared EgoDex episode with a fixed two-arm VX300S target."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import threading
import time

from fixed_robot.store import Store
from transfer.selection import selected
from transfer.review import accepted

HERE = Path(__file__).resolve().parent


def attempt_prompt(prompt, thread, remaining):
    if thread is None:
        return prompt
    return (f"Continue this same episode. {remaining:.0f}s remain. Read prior physical and visual evidence, "
            "diagnose and repair the earliest failure and finish the current rollout with record_review. "
            "Do not redo successful stages or reschedule another agent.")


class ConnectionTiming:
    """Only subtract intervals identified by connection errors in the CLI stream."""
    markers = ("at capacity", "disconnected", "connection reset", "temporarily unavailable",
               "stream error", "reconnecting", "stream disconnected")

    def __init__(self):
        self.thread = None
        self.wait_start = None
        self.excluded = 0.
        self.transient = False
        self.errors = []

    def observe(self, event, now):
        kind = event.get("type")
        if kind in ("error", "turn.failed"):
            message = str(event.get("message", event.get("error", "")))
            self.errors.append(message)
            self.transient = any(s in message.lower() for s in self.markers)
            if self.transient and self.wait_start is None:
                self.wait_start = now
            elif not self.transient:
                self.close_wait(now)
            return
        if kind == "thread.started":
            self.thread = event["thread_id"]
        if kind in ("thread.started", "turn.started") or (kind or "").startswith("item."):
            self.close_wait(now)
            self.transient = False

    def close_wait(self, now):
        if self.wait_start is not None:
            self.excluded += now - self.wait_start
            self.wait_start = None


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
    flags = ["--json", "--output-last-message", str(work / "owner_final.md")]
    start = time.time()
    active = 0.0
    thread = None
    attempts = []

    while active < budget:
        before = time.monotonic()
        logpath = work / (
            "owner.jsonl" if not attempts else f"owner_{len(attempts) + 1}.jsonl"
        )
        tail = flags + ["-"] if thread is None else ["resume", *flags, thread, "-"]
        command_input = attempt_prompt(prompt, thread, budget - active)
        timing = ConnectionTiming()
        with logpath.open("w") as log, (work / "owner.stderr").open("a") as err:
            proc = subprocess.Popen(
                command + tail,
                stdin=subprocess.PIPE,
                text=True,
                stdout=subprocess.PIPE,
                stderr=err,
                env=env,
                cwd=work,
                start_new_session=True,
            )
            def drain():
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    timing.observe(event, time.monotonic())

            reader = threading.Thread(target=drain)
            reader.start()
            try:
                try:
                    proc.stdin.write(command_input)
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                remaining = budget - active
                while True:
                    try:
                        proc.wait(timeout=max(.1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        remaining = budget - active - (time.monotonic() - before - timing.excluded)
                        if remaining <= .1 or timing.wait_start is not None:
                            raise
                code = proc.returncode
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = 124
            finally:
                reader.join()
                proc.stdout.close()
        after = time.monotonic()
        elapsed = after - before
        timing.close_wait(after)
        thread = timing.thread or thread
        transient = timing.transient
        excluded = timing.excluded
        active += elapsed - excluded
        attempts.append(
            dict(
                elapsed=elapsed,
                effective=elapsed - excluded,
                excluded=excluded,
                errors=timing.errors,
                exit_code=code,
            )
        )
        passed = False
        if (work / ".harness/state.json").exists():
            try:
                ctx = Store(work)
                ctx = selected(ctx)
                accepted(ctx)
                passed = True
            except (KeyError, ValueError, OSError):
                pass
        if passed:
            code = 0
            break
        if (
            (code == 124 and not (transient and excluded > 0 and active < budget))
            or (not thread and not transient)
            or (code != 0 and not transient)
        ):
            code = code or 3
            break
        if transient:
            wait_start = time.monotonic()
            time.sleep(10)
            attempts[-1]["retry_wait_seconds"] = time.monotonic() - wait_start
    else:
        code = 124
    (work / "owner_runtime.json").write_text(
        json.dumps(
            dict(
                start=start,
                end=time.time(),
                exit_code=code,
                effective_seconds=active,
                thread_id=thread,
                attempts=attempts,
            ),
            indent=2,
        )
    )
    if code:
        print(f"Episode stopped before acceptance. See {work / 'owner.stderr'} and owner_runtime.json.", file=sys.stderr)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
