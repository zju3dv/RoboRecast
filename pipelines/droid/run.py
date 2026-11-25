"""Run the DROID reconstruction or augmentation owner on a prepared episode."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify(work):
    """Recompile and execute the candidate; keep acceptance tied to this rollout."""
    from robots import check_identity
    from rollout import replay
    from roborecast.requirements import digest
    from timing import audit
    from acceptance import accepted_run, fingerprint

    task = read(work / "task.json")
    bundle = work / "bundle"
    outcome = work / "accepted.json"
    outcome.unlink(missing_ok=True)
    check_identity(work / "robot", bundle)
    contract = read(bundle / "requirements.json")
    if digest(contract) != digest(read(work / "intent.json")):
        raise ValueError("Candidate requirements differ from the task's bound interaction contract")
    if task["mode"] == "augment":
        from binding import bind, verify_physical
        source = Path(task["source"])
        accepted = accepted_run(source)
        if accepted.get("mode") != "reconstruct" or Path(accepted["episode"]).resolve() != Path(task["episode"]).resolve():
            raise ValueError("Expected an accepted reconstruction of this episode")
        if task.get("source_fingerprint", accepted["fingerprint"]) != accepted["fingerprint"]:
            raise ValueError("Source reconstruction changed since augmentation began")
        mapping = read(work / "bindings.json")
        verify_physical(source / "bundle", bundle, mapping)
        expected = bind(source / "bundle", bundle, mapping)
        if digest(contract) != digest(expected):
            raise ValueError("Target requirements differ from the source with declared entity bindings")
        if "layout_check" in task["recipe"]:
            from layout import compare
            roles_file = work / "layout-roles.json"
            layout = compare(Path(task["source"]) / "bundle", bundle, task["recipe"], read(work / "bindings.json"),
                             read(roles_file) if roles_file.exists() else None)
            write(work / "layout.json", layout)
            if layout["status"] != "PASS":
                raise ValueError("Requested object layout failed; inspect layout.json")
    review = read(work / "visual-review.json")
    for name in ("source_frame", "rollout_frame"):
        if not (work / review[name]).is_file():
            raise FileNotFoundError(review[name])
    if review.get("decision") != "PASS" or not review.get("observations"):
        raise ValueError("Review the source and rollout keyframes before accepting the task")
    # Cache-free final execution prevents accepting metrics from an earlier edit.
    report = replay(bundle, work / "final")
    if report["status"] != "PASS":
        raise ValueError("Final simulation failed; inspect final/report.json")
    result = audit(task["episode"], bundle / "scene.xml", work / "final/rollout.npz",
                   bundle / "runtime.json")
    write(work / "final/operation-duration.json", result)
    write(outcome, {"status": "PASS", "episode": task["episode"], "mode": task["mode"],
                    "contract_sha256": digest(contract), "fingerprint": fingerprint(work), "report": "final/report.json",
                    "operation_duration": "final/operation-duration.json"})


def launch(args):
    from robots import build
    episode, work = args.episode.resolve(), args.out.resolve()
    if not (episode / "manifest.json").is_file():
        raise FileNotFoundError(episode / "manifest.json")
    task = {"mode": args.mode, "episode": str(episode), "repository": str(ROOT),
            "python": sys.executable}
    names = ("BLENDER", "SAM3_PYTHON", "SAM3_CHECKPOINT", "LINGBOT_PYTHON",
             "LINGBOT_ROOT", "LINGBOT_CHECKPOINT", "CUACD_PYTHON", "ROBORECAST_ASSET_DIR")
    task["tool_paths"] = {name: os.environ[name] for name in names if os.environ.get(name)}
    assets = None
    if args.mode == "augment":
        from acceptance import accepted_run
        source = args.source.resolve()
        accepted = accepted_run(source)
        if accepted["mode"] != "reconstruct" or Path(accepted["episode"]).resolve() != episode:
            raise ValueError("Select an accepted reconstruction of this episode")
        recipe = read(args.recipe.resolve())
        if recipe.get("episode_id", episode.name) != episode.name:
            raise ValueError("Recipe episode_id differs from the prepared episode")
        family = read(source / "bundle/runtime.json")["robot_family"]
        robot = recipe.get("robot", "source")
        if robot == "UR5e":
            if not args.ur5e_meshes or not args.ur5e_meshes.is_dir():
                raise ValueError("The UR5e recipe needs --ur5e-meshes pointing to its assets")
            assets = args.ur5e_meshes.resolve()
        elif robot not in ("source", family) or args.ur5e_meshes:
            raise ValueError(f"Choose robot 'source', '{family}' or 'UR5e'; supply meshes only for UR5e")
        task.update(source=str(source), source_fingerprint=accepted["fingerprint"], recipe=recipe)
    work.mkdir(parents=True, exist_ok=False)
    write(work / "task.json", task)
    build(episode, work / "robot", assets)
    prompt = (HERE / "owner.md").read_text() + "\nTask paths and settings:\n" + json.dumps(task, indent=2)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    cmd = [args.agent, "exec", "--skip-git-repo-check", "--sandbox", args.sandbox,
           "--model", args.model, "-c", 'model_reasoning_effort="medium"', "-"]
    start = time.monotonic()
    try:
        with (work / "agent.log").open("w") as log:
            with subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.PIPE, stdout=log,
                                  stderr=subprocess.STDOUT, text=True, start_new_session=True) as agent:
                try:
                    agent.communicate(prompt, timeout=args.minutes * 60)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    os.killpg(agent.pid, signal.SIGKILL)
                    agent.wait()
                    raise
                if agent.returncode:
                    raise subprocess.CalledProcessError(agent.returncode, cmd)
        # The owner has already verified its final candidate in this session.
        from acceptance import accepted_run
        accepted_run(work)
    finally:
        write(work / "elapsed.json", {"wall_seconds": time.monotonic()-start,
                                      "scope": "agent and verification, includes any connection waits"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("reconstruct", "augment"):
        cmd = modes.add_parser(name)
        cmd.add_argument("--episode", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--agent", default="codex")
        cmd.add_argument("--model", default="gpt-6-astra")
        cmd.add_argument("--minutes", type=float, default=40)
        cmd.add_argument("--sandbox", choices=["workspace-write", "danger-full-access"], default="workspace-write")
        if name == "augment":
            cmd.add_argument("--source", type=Path, required=True)
            cmd.add_argument("--recipe", type=Path, required=True)
            cmd.add_argument("--ur5e-meshes", type=Path)
    check = modes.add_parser("verify")
    check.add_argument("work", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        verify(args.work.resolve())
    else:
        launch(args)


if __name__ == "__main__":
    main()
