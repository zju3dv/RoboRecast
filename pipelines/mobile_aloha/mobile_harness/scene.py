from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from .rendering import presentation_renderer, presentation_options
import numpy as np
from PIL import Image, ImageDraw

from mobile_input.writer import sha256, write_json
from mobile_input.loader import load_episode
from mobile_robot.asset import apply_mature_arm_control_profile
from .store import Store
from .bindings import bundle_assets, scene_digest
from .compatibility import settle_layout_signature
from .stages import overview_camera


def register_modeling_unit(store: Store, request: dict) -> dict:
    unit_id = request["unit_id"]
    if not unit_id.replace("_", "").isalnum():
        raise ValueError("unit_id must be alphanumeric/underscore")
    fragment = store.artifact(request["fragment_xml_path"])
    tree = ET.parse(fragment)
    if tree.getroot().tag != "mujoco" or tree.find("worldbody") is None:
        raise ValueError("fragment must be a MuJoCo XML with worldbody")
    # Task geometry is passive. Closed chains within a modeled unit are valid.
    if list(tree.findall("./actuator/*")):
        raise ValueError("Task objects must move through robot contact; remove task actuators")
    local = {node.get("name") for node in tree.getroot().iter() if node.get("name")}
    for joint in tree.findall("./equality/*"):
        references = [value for key, value in joint.attrib.items()
                      if key.startswith(("body", "site", "joint", "tendon"))]
        if any(name not in local or "/" in name for name in references):
            raise ValueError("Equality constraints must reference only bodies and joints in this unit")
    state = store.read()
    state["units"][unit_id] = {"fragment_xml_path": str(fragment.relative_to(store.root)),
                               "sha256": sha256(fragment),
                               "pose": request.get("pose", {"xyyaw": [0, 0, 0]}),
                               "role": request.get("role", "task_object")}
    state["stage"] = "UNITS_READY"
    store.save(state)
    return {"status": "PASS", "unit_id": unit_id, "unit_count": len(state["units"])}


def _namespace(element: ET.Element, unit_id: str) -> ET.Element:
    element = copy.deepcopy(element)
    for item in element.iter():
        if item.get("name"):
            item.set("name", f"{unit_id}/{item.get('name')}")
        for key in ("body", "body1", "body2", "joint", "joint1", "joint2",
                    "site", "site1", "site2", "geom", "geom1", "geom2", "target",
                    "mesh", "material", "texture", "class", "childclass", "tendon"):
            if item.get(key) and "/" not in item.get(key):
                item.set(key, f"{unit_id}/{item.get(key)}")
    return element


def merge_fragment_assets(root, fragment, fragment_path, unit_id):
    """Preserve fragment meshes/materials and scope its defaults to its bodies."""
    source = fragment.find("asset")
    if source is not None:
        target = root.find("asset")
        if target is None: target = ET.SubElement(root, "asset")
        compiler = fragment.find("compiler")
        for asset in source:
            item = _namespace(asset, unit_id)
            if item.get("file"):
                directory = compiler.get("meshdir" if item.tag == "mesh" else "texturedir", "") if compiler is not None else ""
                item.set("file", str((fragment_path.parent / directory / item.get("file")).resolve()))
            target.append(item)
    source = fragment.find("default")
    if source is not None:
        target = root.find("default")
        if target is None: target = ET.SubElement(root, "default")
        scoped = _namespace(source, unit_id)
        scoped.set("class", f"{unit_id}/__defaults")
        target.append(scoped)


def merge_and_settle(input_dir: Path, store: Store, request: dict) -> dict:
    state = store.read()
    if not state["units"]:
        raise ValueError("register at least one modeling unit")
    episode = load_episode(input_dir)
    base = episode.runtime_scene
    tree = ET.parse(base)
    root = tree.getroot()
    # The merged XML is written under the reconstruction directory, while
    # bundle meshes stay beside the immutable runtime XML. Preserve the
    # runtime XML's asset lookup semantics with absolute compiler paths.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for attribute in ("meshdir", "texturedir"):
        source_dir = compiler.get(attribute, "")
        compiler.set(attribute, str((base.parent / source_dir).resolve()))
    # Runtime inputs created before v1.4 may still contain the weak legacy
    # VX300S actuator defaults. Normalize every merged scene so old episodes
    # receive the same proven controller without mutating immutable inputs.
    if episode.runtime_contract.get("embodiment_id") in (None, "mobile_aloha_tracer_vx300s"):
        apply_mature_arm_control_profile(root)
    world = root.find("worldbody")
    for unit_id, row in state["units"].items():
        fragment_path = store.artifact(row["fragment_xml_path"])
        if sha256(fragment_path) != row["sha256"]:
            raise ValueError(f"fragment changed: {unit_id}")
        fragment = ET.parse(fragment_path).getroot()
        merge_fragment_assets(root, fragment, fragment_path, unit_id)
        pose = row["pose"].get("xyyaw", [0, 0, 0])
        for body in fragment.find("worldbody"):
            item = _namespace(body, unit_id)
            if fragment.find("default") is not None:
                item.attrib.setdefault("childclass" if item.tag == "body" else "class", f"{unit_id}/__defaults")
            if item.tag == "body":
                item.set("pos", f"{pose[0]} {pose[1]} {row['pose'].get('z', 0)}")
                # Registered pose is the sole root transform. Remove any
                # alternative root orientation left in the fragment so a
                # fragment-authored absolute pose cannot be applied twice.
                for attribute in ("quat", "axisangle", "xyaxes", "zaxis"):
                    item.attrib.pop(attribute, None)
                item.set("euler", f"0 0 {pose[2]}")
            world.append(item)
        for section in ("equality", "contact", "actuator", "sensor", "tendon"):
            source = fragment.find(section)
            if source is not None:
                target = root.find(section)
                if target is None:
                    target = ET.SubElement(root, section)
                for element in source:
                    target.append(_namespace(element, unit_id))
    # Older exported non-ALOHA bundles omitted ground. Keep the input immutable
    # and supply the same physical floor as the existing ALOHA runtime.
    if root.find("./worldbody/geom[@name='floor']") is None:
        ET.SubElement(world, "geom", name="floor", type="plane", size="10 10 .1",
                      rgba=".72 .72 .7 1", friction="1.4 .02 .001")
    revision = len(state["merges"]) + 1
    scene = store.root / f"simulation/scene_r{revision}.xml"
    scene.parent.mkdir(parents=True, exist_ok=True)
    bundle_assets(tree, base.parent, scene.parent)
    ET.indent(root, space="  ")
    tree.write(scene, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    with np.load(episode.initial_state) as initial:
        robot_qpos = initial["qpos"]
        data.qpos[:len(robot_qpos)] = robot_qpos
        data.ctrl[:len(initial["ctrl"])] = initial["ctrl"]
    robot_pose = request.get("robot_xyyaw", [0, 0, 0])
    base_addresses = [int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in episode.runtime_contract["base_pose_joints"]]
    base_dofs = [int(model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in episode.runtime_contract["base_pose_joints"]]
    data.qpos[base_addresses] = robot_pose[:3]
    # A mass/friction/damping-only revision may reuse the previous settled
    # configuration. Geometry, pose, joint, or topology edits must omit this.
    reused_settle_revision = request.get("reuse_settled_state_revision")
    if reused_settle_revision is not None:
        previous = next((row for row in state["merges"]
                         if row["revision"] == int(reused_settle_revision)), None)
        if previous is None:
            raise ValueError("reuse_settled_state_revision does not exist")
        previous_model = mujoco.MjModel.from_xml_path(str(store.artifact(previous["scene_xml_path"])))
        if settle_layout_signature(previous_model) != settle_layout_signature(model):
            raise ValueError("settle reuse requires unchanged geometry/layout; merge without reuse_settled_state_revision")
        with np.load(store.artifact(previous["settled_state_path"])) as prior:
            if prior["qpos"].shape != data.qpos.shape:
                raise ValueError("physical-only settle reuse requires unchanged model topology")
            if not np.allclose(prior["qpos"][base_addresses], robot_pose[:3], atol=1e-9, rtol=0):
                raise ValueError("initial base pose changed; merge without reuse_settled_state_revision")
            data.qpos[:] = prior["qpos"]
            data.qvel[:] = 0
            if prior["ctrl"].shape == data.ctrl.shape:
                data.ctrl[:] = prior["ctrl"]
        if "robot_xyyaw" in request:
            data.qpos[base_addresses] = robot_pose[:3]
        mujoco.mj_forward(model, data)
        steps = 0
    else:
        steps = max(1, int(float(request.get("settle_seconds", .5)) / model.opt.timestep))
    # The base is prescribed in the plane; settle only the arms and task objects.
    base_pose = data.qpos[base_addresses].copy()
    max_penetration = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)
        data.qpos[base_addresses] = base_pose
        data.qvel[base_dofs] = 0
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all():
            raise ValueError("settle produced nonfinite state")
        for contact in data.contact:
            max_penetration = max(max_penetration, -float(contact.dist))
    settled = store.root / f"simulation/settled_r{revision}.npz"
    np.savez_compressed(settled, qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
    record = {"revision": revision, "scene_xml_path": str(scene.relative_to(store.root)),
              "settled_state_path": str(settled.relative_to(store.root)),
              "scene_sha256": scene_digest(scene), "settled_sha256": sha256(settled),
              "units": state["units"].copy(), "nq": model.nq,
              "max_penetration_m": max_penetration,
              "reused_settled_state_revision": reused_settle_revision,
              "base_initial_xy": data.qpos[base_addresses[:2]].tolist()}
    state["merges"].append(record)
    state["stage"] = "MERGED"
    store.save(state)
    write_json(store.root / f"simulation/merge_r{revision}.json", record)
    return {"status": "PASS", **record}


def render_scene_preview(store: Store, request: dict) -> dict:
    state = store.read()
    if not state["merges"]:
        raise ValueError("merge scene first")
    merge = state["merges"][-1]
    model = mujoco.MjModel.from_xml_path(str(store.artifact(merge["scene_xml_path"])))
    data = mujoco.MjData(model)
    with np.load(store.artifact(merge["settled_state_path"])) as settled:
        data.qpos[:] = settled["qpos"]
        data.qvel[:] = settled["qvel"]
        data.ctrl[:] = settled["ctrl"]
    mujoco.mj_forward(model, data)
    focus = request.get("focus_body")
    reference = request.get("reference_crop_path")
    image = Image.new("RGB", (960 if reference else 640, 240), "black")
    detail = []
    for name in request.get("detail_geoms", []):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"unknown detail geom: {name}")
        detail.append({"name": name, "type": int(model.geom_type[gid]),
                       "size_parameters_m": model.geom_size[gid].tolist(),
                       "world_position_m": data.geom_xpos[gid].tolist(),
                       "note": "MuJoCo size convention: box half extents; cylinder radius/half height; mesh size is not a measured object dimension"})
    overview = overview_camera(model, data, focus)
    if focus:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, focus)
        if bid < 0: raise ValueError(f"unknown focus body: {focus}")
        bodies = {bid}
        for b in range(bid+1, model.nbody):
            if int(model.body_parentid[b]) in bodies: bodies.add(b)
        geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]
        if geoms:
            lo = np.min([data.geom_xpos[g]-model.geom_rbound[g] for g in geoms], axis=0)
            hi = np.max([data.geom_xpos[g]+model.geom_rbound[g] for g in geoms], axis=0)
            overview.lookat[:] = (lo+hi)/2
            overview.distance = max(.25, float(np.max(hi-lo))*1.9)
        else: overview.lookat[:] = data.xpos[bid]

    draw = ImageDraw.Draw(image)
    renderer = presentation_renderer(model, height=240, width=320)
    try:
        top = next((name for name in ("top", "head", "camera_top")
                    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0), None)
        if top is None:
            top = 0
        for index, camera in enumerate((top, overview)):
            renderer.update_scene(data, camera=camera, scene_option=presentation_options())
            image.paste(Image.fromarray(renderer.render().copy()), (index * 320, 0))
            draw.rectangle((index * 320, 0, index * 320 + 319, 29), fill=(24, 24, 24))
            draw.text((index * 320 + 6, 5), "top" if index == 0 else "task overview", fill="white")
    finally:
        renderer.close()
    output = store.root / f"simulation/scene_r{merge['revision']}_preview.jpg"
    if reference:
        with Image.open(store.artifact(reference)) as source:
            source = source.convert("RGB")
            source.thumbnail((320,210))
            image.paste(source,(640+(320-source.width)//2,30+(210-source.height)//2))
        draw.text((646,5), "source crop (not camera-aligned)", fill="white")
    image.save(output, quality=88)
    return {"status": "PASS", "preview_path": str(output.relative_to(store.root)), "geometry_details": detail}
