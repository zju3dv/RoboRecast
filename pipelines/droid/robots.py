"""Prepare an episode-specific robot base with the complete Robotiq 2F-85."""
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
GRIPPER = HERE.parents[1] / "robots/robotiq_2f85"
MOUNT = ".//body[@name='robotiq_arg2f_base_link']"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def relocate(tree, resolve, destination):
    for node in tree.findall("asset/*[@file]"):
        source = resolve(node)
        if not source.is_file():
            raise FileNotFoundError(source)
        node.set("file", os.path.relpath(source, destination))


def add_section(tree, name):
    section = tree.find(name)
    return section if section is not None else ET.SubElement(tree.getroot(), name)


def install_gripper(tree, mount, out):
    """Expand the shared MJCF into a portable robot base with rebased meshes."""
    for node in ET.parse(GRIPPER / "hand.xml").getroot():
        mount.append(node)
    for section in ET.parse(GRIPPER / "components.xml").getroot():
        for mesh in section.findall("*[@file]"):
            mesh.set("file", os.path.relpath(GRIPPER / mesh.get("file"), out))
        add_section(tree, section.tag).extend(section)


def source_robot(episode, out):
    base = episode / "derived/robot_runtime"
    tree = ET.parse(base / "robot_base_scene.xml")
    runtime = read(base / "robot_runtime_contract.json")
    relocate(tree, lambda node: base / node.get("file"), out)
    # Older prepared episodes have the four-joint proxy. Keep the arm, wrist
    # calibration and mounting transform, replacing only the hand mechanism.
    if tree.find(".//joint[@name='rq_right_spring_link_joint']") is None and tree.find("default/default[@class='rq_2f85']") is None:
        old = tree.find(MOUNT)
        names = {node.get("name") for node in old.iter() if node.get("name")}
        for node in list(old):
            if node.tag not in ("camera", "site"):
                old.remove(node)
        mount_attrs = {key: old.attrib[key] for key in ("name", "pos", "quat", "euler") if key in old.attrib}
        old.attrib.clear()
        old.attrib.update(mount_attrs, childclass="rq_2f85")
        for name in ("contact", "equality", "tendon", "actuator"):
            section = add_section(tree, name)
            for node in list(section):
                if names.intersection(node.attrib.values()) or node.get("name") == "gripper_position":
                    section.remove(node)
        install_gripper(tree, old, out)
        key = tree.find("keyframe/key")
        arm = key.get("qpos").split()[:len(runtime["arm_joint_names"])]
        pose = read(GRIPPER / "runtime.json")["open_qpos"]
        hand = [str(pose[joint.get("name")]) for joint in old.iter("joint")]
        key.set("qpos", " ".join(arm + hand))
        key.attrib.pop("qvel", None)
    tree.find("keyframe/key").set("name", "settled_initial")
    geometry = read(episode / "specs/camera_spec.json")["image_geometry"]
    for camera in tree.iter("camera"):
        k = geometry[camera.get("name")]["proxy_intrinsics"]
        w, h = k["width"], k["height"]
        camera.attrib.pop("fovy", None)
        camera.attrib.update(sensorsize="1 1", resolution=f"{w} {h}",
                             focalpixel=f"{k['fx']} {k['fy']}",
                             principalpixel=f"{(w-1)/2-k['cx']} {(h-1)/2-k['cy']}")
    return tree, runtime


def build(episode, out, target=None):
    out.mkdir(parents=True)
    if target is None:
        tree, runtime = source_robot(episode, out)
    else:
        profile = HERE / "robots/ur5e"
        tree = ET.parse(profile / "scene.xml")
        runtime = read(profile / "runtime.json")

        relocate(tree, lambda node: target / Path(node.get("file")).name, out)
        mount = tree.find(MOUNT)
        mount.remove(mount.find("include"))
        tree.getroot().remove(tree.find("include"))
        install_gripper(tree, mount, out)
    runtime["gripper"] = read(GRIPPER / "runtime.json")["gripper"]
    path = out / "scene.xml"
    tree.write(path, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nu != len(runtime["arm_joint_names"]) * 2 + 1:
        raise ValueError("Robot actuator count differs from its contract")
    write(out / "runtime.json", runtime)


def check_identity(base, bundle):
    """Compare the compiled arm chain and gripper joints to the selected robot."""
    expected = read(base / "runtime.json")
    actual = read(bundle / "runtime.json")
    for key in ("robot_family", "arm_joint_names", "end_effector_body"):
        if expected[key] != actual[key]:
            raise ValueError(f"Candidate robot differs from the selected {key}")
    models = [mujoco.MjModel.from_xml_path(str(folder / "scene.xml")) for folder in (base, bundle)]
    for name in expected["arm_joint_names"] + ["left_finger_joint", "right_finger_joint"]:
        reference, candidate = [m.joint(name) for m in models]
        for joint in (reference, candidate):
            if joint.id < 0:
                raise ValueError(f"Missing robot joint {name}")
        for field in ("axis", "pos", "type"):
            if not np.allclose(getattr(reference, field), getattr(candidate, field), atol=1e-7):
                raise ValueError(f"Joint geometry differs at {name}: {field}")
        links = [m.body(int(joint.bodyid[0])) for m, joint in zip(models, (reference, candidate))]
        if any(not np.allclose(getattr(links[0], field), getattr(links[1], field), atol=1e-7)
               for field in ("pos", "quat")):
            raise ValueError(f"Robot link transform differs at {name}")
