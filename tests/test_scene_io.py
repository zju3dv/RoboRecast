import json
import tempfile
import unittest
from pathlib import Path

from roborecast.assembly import assemble
from roborecast.harness import load_scene
from roborecast.inspection import settle


class SceneIOTest(unittest.TestCase):
    def test_scene_load_preserves_meshdir_with_and_without_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"meshes").mkdir()
            (root/"meshes/test.obj").write_text("v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
            path = root/"scene.xml"
            path.write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh name="tet" file="test.obj"/></asset><worldbody><geom type="mesh" mesh="tet"/></worldbody></mujoco>')
            patch = {"world_collision_box":{"name":"block","center_m":[1,0,0],"half_extents_m":[.1,.1,.1]}}
            for edit, count in (({},1),(patch,2)):
                model = load_scene(path, edit)
                self.assertEqual(model.ngeom, count)
                self.assertEqual(settle(model, seconds=.01)["maximum_penetration_m"],0.)

    def test_assembly_rejects_mixed_angle_units(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"world.xml").write_text('<mujoco><compiler angle="radian"/><worldbody/></mujoco>')
            (root/"unit.xml").write_text('<mujoco><worldbody><body><joint range="0 90"/><geom size=".01"/></body></worldbody></mujoco>')
            manifest = root/"assembly.json"
            manifest.write_text(json.dumps({"base":"world.xml","units":[{"file":"unit.xml","name":"door"}]}))
            with self.assertRaisesRegex(ValueError,"compiler options"):
                assemble(manifest,root/"result.xml")
