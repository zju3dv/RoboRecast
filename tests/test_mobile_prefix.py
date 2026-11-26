"""An unchanged suffix may follow a failing probe, but only the passing prefix is reusable."""
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "pipelines/mobile_aloha/mobile_harness/compatibility.py"
spec = importlib.util.spec_from_file_location("mobile_compatibility", path)
compatibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compatibility)


class MobilePrefix(unittest.TestCase):
    def test_failed_stage_is_excluded_even_if_its_checkpoint_exists(self):
        record = dict(stage_hashes=["approach", "contact", "push"], passed_prefix=1,
                      checkpoint_paths=["a", "b", "c"], scene_sha256="scene",
                      settled_sha256="settled", engine_sha256="solver")
        match, count = compatibility.longest_prefix(
            [record], record["stage_hashes"], 3, "scene", "settled", "solver")
        self.assertIs(match, record)
        self.assertEqual(count, 1)
