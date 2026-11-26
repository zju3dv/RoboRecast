from __future__ import annotations

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


VX = Path(__file__).resolve().parents[3] / "robots/vx300s"
JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
RADIUS_M = 0.11
TRACK_M = 0.54
BASE_JOINTS = ("tracer/x", "tracer/y", "tracer/yaw")


# ALOHA2 parameters already used by the mature fixed-base dual-arm pipeline.
# The older VX300S defaults sag several centimetres under a modest payload,
# causing agents to compensate with geometrically false end-effector targets.
MATURE_ARM_DEFAULTS = {
    "waist": ({"damping": "5.76"}, {"kp": "43"}),
    "shoulder": ({"armature": "0.395", "frictionloss": "2.0", "damping": "20.0"},
                 {"kp": "265", "forcerange": "-144 144"}),
    "elbow": ({"armature": "0.383", "frictionloss": "1.15", "damping": "18.49"},
              {"kp": "227", "forcerange": "-59 59"}),
    "forearm_roll": ({"armature": "0.14", "damping": "6.78"},
                     {"kp": "78", "forcerange": "-22 22"}),
    "wrist_angle": ({"armature": "0.008", "damping": "6.28"}, {"kp": "37"}),
    "wrist_rotate": ({"damping": "1.2"}, {"kp": "10.4"}),
    "finger": ({"armature": "0.243", "damping": "40"}, {"kp": "2000", "kv": "124"}),
}


def apply_mature_arm_control_profile(root: ET.Element) -> None:
    """Apply the fixed-base ALOHA2 control profile to prefixed or raw defaults."""
    for default in root.iter("default"):
        class_name = default.get("class", "").split("/")[-1]
        profile = MATURE_ARM_DEFAULTS.get(class_name)
        if profile is None:
            continue
        joint_attrs, position_attrs = profile
        joint = default.find("joint")
        position = default.find("position")
        if joint is not None:
            joint.attrib.update(joint_attrs)
        if position is not None:
            position.attrib.update(position_attrs)


def _prefix(element: ET.Element, side: str) -> ET.Element:
    element = copy.deepcopy(element)
    for item in element.iter():
        for attr in ("name", "class", "childclass", "joint", "joint1", "joint2",
                     "body", "body1", "body2", "site", "site1", "site2", "target"):
            value = item.get(attr)
            if value:
                item.set(attr, f"{side}/{value}")
    return element


def _root() -> ET.Element:
    root = ET.Element("mujoco", model="mobile_aloha_tracer_vx300s")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.005", cone="elliptic", impratio="10")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.2 0.2 0.2",
                  diffuse="0.65 0.65 0.65", specular="0.1 0.1 0.1")
    ET.SubElement(root, "default")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="10 10 0.1",
                  rgba="0.72 0.72 0.7 1", friction="1.4 0.02 0.001")
    base = ET.SubElement(world, "body", name="tracer/base", pos="0 0 0.2")
    ET.SubElement(base, "joint", name=BASE_JOINTS[0], type="slide", axis="1 0 0")
    ET.SubElement(base, "joint", name=BASE_JOINTS[1], type="slide", axis="0 1 0")
    ET.SubElement(base, "joint", name=BASE_JOINTS[2], type="hinge", axis="0 0 1")
    ET.SubElement(base, "geom", name="tracer/chassis", type="box", size="0.36 0.28 0.12",
                  rgba="0.18 0.19 0.20 1", mass="25", pos="0 0 0")
    ET.SubElement(base, "geom", name="tracer/mast", type="box", size="0.10 0.17 0.32",
                  pos="-0.18 0 0.41", rgba="0.17 0.18 0.19 1", mass="8")
    ET.SubElement(base, "geom", name="tracer/upper_rack", type="box", size="0.28 0.34 0.03",
                  pos="0.08 0 0.78", rgba="0.18 0.18 0.19 1", mass="3")
    ET.SubElement(base, "camera", name="top", pos="0.35 0 1.30",
                  xyaxes="0 -1 0 0.75 0 0.66", fovy="68")
    ET.SubElement(base, "camera", name="follow", pos="-2 -1.5 1.4",
                  xyaxes="0.6 -0.8 0 0.27 0.20 0.94", fovy="65")
    # Top and wrist cameras are asset-fixed. They are shared by all episodes.
    for longitudinal, x in (("front", 0.24), ("rear", -0.24)):
        for side, y in (("left", TRACK_M / 2), ("right", -TRACK_M / 2)):
            name = f"tracer/{longitudinal}_{side}_wheel"
            wheel = ET.SubElement(base, "body", name=name, pos=f"{x} {y} -0.09")
            ET.SubElement(wheel, "geom", name=f"{name}_tire", type="cylinder",
                          size=f"{RADIUS_M} 0.047", quat="0.70710678 0.70710678 0 0",
                          mass="1.5", rgba="0.08 0.08 0.08 1", contype="0", conaffinity="0")
    for camera_name, pos, xyaxes in (
        ("route_overview", "1 -4 3.4", "1 0 0 0 0.65 0.76"),
        ("interaction_overview", "2 0 2", "0 1 0 -0.6 0 0.8"),
    ):
        ET.SubElement(world, "camera", name=camera_name, pos=pos, xyaxes=xyaxes, fovy="65")
    return root


def _assemble_arm(root: ET.Element, source: ET.Element, side: str) -> None:
    source_default = source.find("default/default")
    if source_default is None:
        raise ValueError("VX300S default missing")
    root.find("default").append(_prefix(source_default, side))
    assets = root.find("asset")
    if assets is None:
        assets = ET.Element("asset")
        root.insert(3, assets)
    if not list(assets):
        for entry in source.find("asset"):
            item = copy.deepcopy(entry)
            if item.get("file"):
                item.set("file", "assets/" + item.get("file"))
            assets.append(item)
    body = _prefix(source.find("worldbody/body"), side)
    body.set("pos", f"0.10 {'0.21' if side == 'left' else '-0.21'} 0.81")
    body.set("quat", "1 0 0 0")
    gripper = next(item for item in body.iter("body")
                   if item.get("name") == f"{side}/gripper_link")
    ET.SubElement(gripper, "camera", name=f"wrist_cam_{side}",
                  pos=f"0.09 {'0.04' if side == 'left' else '-0.04'} 0.12",
                  xyaxes="0 -1 0 0.75 0 0.66", fovy="90")
    base = root.find("worldbody/body")
    base.append(body)
    for section_name in ("actuator", "equality", "contact"):
        target = root.find(section_name)
        if target is None:
            target = ET.SubElement(root, section_name)
        source_section = source.find(section_name)
        if source_section is not None:
            for item in source_section:
                target.append(_prefix(item, side))


def build_robot_asset(output: Path) -> Path:
    source = ET.parse(VX / "vx300s.xml").getroot()
    root = _root()
    _assemble_arm(root, source, "left")
    _assemble_arm(root, source, "right")
    apply_mature_arm_control_profile(root)
    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(VX / "assets", output.parent / "assets", dirs_exist_ok=True)
    ET.ElementTree(root).write(output, encoding="unicode")
    mujoco.MjModel.from_xml_path(str(output))
    return output


def _joint(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if value < 0:
        raise ValueError(f"missing joint {name}")
    return value


def _actuator(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if value < 0:
        raise ValueError(f"missing actuator {name}")
    return value


def initial_state(model: mujoco.MjModel, source_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_qpos = np.asarray(source_qpos, dtype=float)
    if source_qpos.shape != (14,):
        raise ValueError("source qpos width must be 14")
    data = mujoco.MjData(model)
    for name in BASE_JOINTS:
        data.qpos[model.jnt_qposadr[_joint(model, name)]] = 0
    for side, offset in (("left", 0), ("right", 7)):
        for index, name in enumerate(JOINTS):
            joint_name = f"{side}/{name}"
            qvalue = float(source_qpos[offset + index])
            data.qpos[model.jnt_qposadr[_joint(model, joint_name)]] = qvalue
            data.ctrl[_actuator(model, joint_name)] = qvalue
        opening = float(np.clip(source_qpos[offset + 6], 0, 1))
        left_finger = 0.021 + 0.036 * opening
        data.qpos[model.jnt_qposadr[_joint(model, f"{side}/left_finger")]] = left_finger
        data.qpos[model.jnt_qposadr[_joint(model, f"{side}/right_finger")]] = -left_finger
        data.ctrl[_actuator(model, f"{side}/gripper")] = left_finger
    mujoco.mj_forward(model, data)
    return data.qpos.copy(), data.ctrl.copy()
