"""Render an agent-authored primitive assembly with the native AgentSTAR Blender harness."""

import json
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


# The episode agent supplies numbers and primitive geometry through model_unit.
# Blender runs this project-owned builder; input strings are serialized as data.
BUILDER = """
import bpy
import json
from pathlib import Path
from mathutils import Quaternion
SPEC = json.loads(Path(__file__).with_name('parts.json').read_text())
SCALE = SPEC['scale']
REFERENCE_FRAME = 'source'
JOINTS = []
FRAMES = {'source': {'pose': SPEC['pose']}}

def build():
    for part in SPEC['parts']:
        kind, radii = part['type'], part['size']
        if kind == 'box':
            bpy.ops.mesh.primitive_cube_add(size=2)
        elif kind == 'cylinder':
            bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1, depth=2)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16)
        obj = bpy.context.object
        obj.name = part['name']
        obj.location = part['position']
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = Quaternion(part['quaternion'])
        obj.scale = radii
        mat = bpy.data.materials.new(obj.name+'_material')
        mat.diffuse_color = part['rgba']
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = part['rgba']
        bsdf.inputs['Roughness'].default_value = 0.42
        bsdf.inputs['Alpha'].default_value = part['rgba'][3]
        obj.data.materials.append(mat)
"""


def render_unit(episode, xml, output):
    from PIL import Image
    import cv2

    root = Path(__file__).resolve().parents[3]
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    meta = json.loads((episode / "meta.json").read_text())
    with np.load(episode / "reference.npz") as ref:
        k = ref["K"] if ref["K"].ndim == 2 else ref["K"][0]
        camera = np.array(meta["T_sim_source_world"]) @ ref["T_world_camera"][0]
    shapes = {
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    }
    parts = []
    for i in range(model.ngeom):
        kind = shapes.get(int(model.geom_type[i]))
        if not kind:
            raise ValueError(
                "Supported visible primitives: box, cylinder, sphere, ellipsoid"
            )
        if model.geom_rgba[i, 3] <= 0:
            continue
        size = model.geom_size[i].copy()
        if kind == "sphere":
            size[:] = size[0]
        if kind == "cylinder":
            size = size[[0, 0, 1]]
        rotation = Rotation.from_matrix(data.geom_xmat[i].reshape(3, 3))
        parts.append(
            dict(
                type=kind,
                name=model.geom(i).name or f"part_{i}",
                size=size.tolist(),
                position=data.geom_xpos[i].copy().tolist(),
                quaternion=rotation.as_quat()[[3, 0, 1, 2]].tolist(),
                rgba=model.geom_rgba[i].tolist(),
            )
        )
    if not parts:
        raise ValueError("Unit has no visible geometry")
    positions = np.array([p["position"] for p in parts])
    radii = np.array([np.linalg.norm(p["size"]) for p in parts])[:, None]
    low = (positions - radii).min(0)
    high = (positions + radii).max(0)
    center = (low + high) / 2
    scale = float(max(high - low))
    for part in parts:
        part["position"] = ((np.asarray(part["position"]) - center) / scale).tolist()
        part["size"] = (np.asarray(part["size"]) / scale).tolist()
    inv = np.linalg.inv(camera)
    pose = dict(
        quaternion=Rotation.from_matrix(inv[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist(),
        translation=(inv[:3, :3] @ center + inv[:3, 3]).tolist(),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "parts.json").write_text(
        json.dumps(dict(parts=parts, scale=scale, pose=pose))
    )
    scene = output / "scene.py"
    scene.write_text(BUILDER)
    width, height = meta["width"], meta["height"]
    intrinsics = ",".join(map(str, [k[0, 0], k[1, 1], k[0, 2], k[1, 2], width, height]))
    default = "blender"
    env = dict(
        os.environ,
        BLENDER=os.environ.get("BLENDER", default),
        RENDER_TIMEOUT="90",
        OMP_NUM_THREADS="2",
    )
    cmd = [
        "bash",
        str(root / "third_party/agenticSTAR/harness/render.sh"),
        str(scene),
        str(output / "passes"),
        "--views",
        "match",
        "--intrinsics",
        intrinsics,
        "--samples",
        "4",
        "--export",
        str(output / "object.glb"),
    ]
    with (output / "harness.log").open("w") as log:
        subprocess.run(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=110
        )
    match = next(
        sorted((output / "passes").glob("*/"), reverse=True)[0].rglob("match*.png")
    )
    video = cv2.VideoCapture(str(episode / "source/rgb.mp4"))
    try:
        ok, frame = video.read()
    finally:
        video.release()
    if not ok:
        raise ValueError("Cannot read initial RGB")
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 360))
    sheet = Image.new("RGB", (1280, 360))
    sheet.paste(rgb, (0, 0))
    with Image.open(match) as image:
        sheet.paste(image.convert("RGB").resize((640, 360)), (640, 0))
    sheet.save(output / "comparison.jpg")
    return output / "comparison.jpg"
