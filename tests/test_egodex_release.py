import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipelines/egodex'))


class ReleaseChecks(unittest.TestCase):
    def test_missing_interaction(self):
        from transfer.coverage import check_coverage
        event = {'id': 'release_in_drawer', 'start': 1, 'end': 2, 'required_checks': [
            {'scope': 'event', 'kind': 'release', 'body': 'toy', 'side': 'left'}]}
        check = {'kind': 'release', 'body': 'toy', 'side': 'left'}
        spec = {'events': [{'source_start': 1.5, 'source_end': 1.6, 'checks': [check]}]}
        self.assertIn('release_in_drawer', check_coverage([event], spec, []))
        for changed in [dict(check, side='right'), dict(check, body='drawer'), {}]:
            spec['events'][0]['checks'] = [changed]
            with self.assertRaisesRegex(ValueError, 'release_in_drawer'):
                check_coverage([event], spec, [])

    def test_binding_refinement_preserves_the_obligation(self):
        from transfer.coverage import validate_revision
        check = {"scope":"phase", "mode":"contact", "side":"left", "object":"stack"}
        old = [{"id":"brace","required_checks":[check]}]
        new = [{"id":"brace","required_checks":[dict(check, object="nested_bowl")],
                "binding_evidence":"frame 120 shows the upper rim"}]
        validate_revision(old, new)
        with self.assertRaisesRegex(ValueError, "brace"):
            validate_revision(old, [])
        new[0]["required_checks"][0]["mode"]="idle"
        with self.assertRaisesRegex(ValueError, "brace"):
            validate_revision(old, new)

    def test_idle_arm_commands_are_checked(self):
        import numpy as np
        from transfer.coverage import check_idle_commands
        events = [{"id":"right_stays_clear","start":0,"end":1,"required_checks":[
            {"scope":"phase","side":"right","mode":"idle"}]}]
        phases = [{"source_start":0,"source_end":1,"hand_objects":{"right":{"mode":"idle"}}}]
        with self.assertRaisesRegex(ValueError, "right_stays_clear"):
            check_idle_commands(events, phases, np.array([0,1]), np.array([[0],[1]]), {"right":[0]})

    def test_contact_duration(self):
        from fixed_robot.contact_policy import FixedContactMonitor
        m = FixedContactMonitor(0.005)
        pair = ('bowl', 'right/gripper_link')
        for i in range(9):
            m.step(i * .005, {pair: 2.})
        m.step(.045, {})
        self.assertFalse(m.failures)
        for i in range(12):
            m.step(.05 + i * .005, {pair: 2.})
        self.assertEqual(len(m.failures), 1)
        self.assertEqual(m.failures[0]['bodies'], list(pair))

    def test_first_retry_receives_full_prompt(self):
        from run import attempt_prompt
        self.assertEqual(attempt_prompt('original task', None, 60), 'original task')
        self.assertNotEqual(attempt_prompt('original task', 'existing-thread', 60), 'original task')

    def test_retry_cli_and_report(self):
        import run
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            episode = base / "prepared"
            episode.mkdir()
            (episode / "reference.npz").touch()
            fake = base / "fake-codex"
            fake.write_text("""#!/usr/bin/env python3
import json,sys,time
from pathlib import Path
calls = Path("invocations")
n = len(calls.read_text().splitlines()) if calls.exists() else 0
with calls.open("a") as f: f.write(json.dumps([sys.argv[1:],sys.stdin.read()])+chr(10))
def emit(**kwargs): print(json.dumps(kwargs),flush=True)
if n == 1:
    emit(type="thread.started",thread_id="test-thread")
    emit(type="item.completed",item={"type":"mcp_tool_call","status":"completed"})
if n < 2:
    emit(type="error",message="stream disconnected")
    time.sleep(.02)
sys.exit(1)
""")
            fake.chmod(0o755)
            work = base / "work"
            args = ["run.py", "--episode", str(episode), "--out", str(work), "--task",
                    "place the toy", "--agent", str(fake)]
            with patch.object(sys, "argv", args), patch("run.time.sleep"), self.assertRaises(SystemExit):
                run.main()
            first, second, third = map(json.loads, (work / "invocations").read_text().splitlines())
            self.assertEqual(first[1], second[1])
            self.assertIn("place the toy", second[1])
            self.assertIn("resume", third[0])
            self.assertIn("test-thread", third[0])
            report = json.loads((work / "owner_runtime.json").read_text())
            self.assertGreater(report["attempts"][1]["excluded"], .01)
            self.assertGreater(report["attempts"][1]["effective"], 0)

    def test_failed_render_does_not_register_unit(self):
        from transfer.tools import EpisodeTools
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tools = EpisodeTools.__new__(EpisodeTools)
            tools.ctx = Mock(root=root, input_dir=root)
            tools.ctx.path.side_effect = lambda s: root / Path(s).name
            with patch('transfer.native.render_unit', side_effect=RuntimeError('Blender failure')), \
                 patch('transfer.tools.register_scene_unit') as register:
                with self.assertRaisesRegex(RuntimeError, 'Blender failure'):
                    tools.call('model_unit', {'unit_id': 'cup', 'role': 'task_object', 'evidence': 'frame 0',
                                               'xml': '<mujoco><worldbody><body name="cup"/></worldbody></mujoco>'})
                register.assert_not_called()
