"""Run and resume a single Codex episode owner against a physical acceptance check."""

import json
import os
import signal
import subprocess
import sys
import threading
import time


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


def run_owner(command, env, work, prompt, budget, is_accepted):
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
        passed = is_accepted()
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
    return code
