from __future__ import annotations

import copy
import shutil
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .utils import sha256, write_json
from .robot_runtime import runtime_contract, fixed_vx300s_contract, CONTRACT_TEXT, base_pose, set_base_pose

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
    root = ET.Element("mujoco", model="fixed_dual_vx300s")
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
                item.set("file", str(VX / "assets" / item.get("file")))
            assets.append(item)
    body = _prefix(source.find("worldbody/body"), side)
    body.set("pos", f"0.10 {'0.21' if side == 'left' else '-0.21'} 0.81")
    body.set("quat", "1 0 0 0")
    gripper = next(item for item in body.iter("body")
                   if item.get("name") == f"{side}/gripper_link")
    ET.SubElement(gripper, "camera", name=f"wrist_cam_{side}",
                  pos=f"0.09 {'0.04' if side == 'left' else '-0.04'} 0.12",
                  xyaxes="0 -1 0 0.75 0 0.66", fovy="90")
    for finger in ("left_finger", "right_finger"):
        finger_body = next(b for b in body.iter("body") if b.get("name") == f"{side}/{finger}_link")
        for i, geom in enumerate(finger_body.iter("geom")):
            geom.set("name", f"{side}/{finger}_{i}")
    for i, geom in enumerate(body.iter("geom")):
        if not geom.get("name"):
            geom.set("name", f"{side}/arm_geom_{i}")
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


def build_robot_asset(output: Path, xyyaw=(0, 0, 0), motion_mode='not_mobile') -> Path:
    source = ET.parse(VX / "vx300s.xml").getroot()
    root = _root()
    _assemble_arm(root, source, "left")
    _assemble_arm(root, source, "right")
    base = root.find("worldbody/body")
    contract = fixed_vx300s_contract(motion_mode, xyyaw)
    if motion_mode == "not_mobile":
        for joint in list(base.findall("joint")):
            base.remove(joint)
    x, y, yaw = map(float, xyyaw)
    if motion_mode == "not_mobile":
        base.set("pos", f"{x} {y} 0.2")
        base.set("quat", f"{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}")
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name=CONTRACT_TEXT, data=json.dumps(contract, separators=(",", ":")))
    apply_mature_arm_control_profile(root)
    # Each built model owns its resource snapshot for reproducible candidate reuse.
    resources = output.parent / "robot_assets"
    resources.mkdir(parents=True, exist_ok=True)
    for element in root.iter():
        if element.get("file"):
            source_file = Path(element.get("file"))
            destination = resources / source_file.name
            if source_file.resolve() != destination.resolve():
                shutil.copy2(source_file, destination)
            element.set("file", str(destination.resolve()))
    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
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


def initial_state(model: mujoco.MjModel, arm_configuration: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    contract = runtime_contract(model)
    arms = contract['arms']
    if arm_configuration is None:
        arm_configuration = np.concatenate([arm['default_configuration'] for arm in arms.values()])
    arm_configuration = np.asarray(arm_configuration, dtype=float)
    expected = sum(len(arm['joints']) + 1 for arm in arms.values())
    if arm_configuration.shape != (expected,):
        raise ValueError(f'robot initial configuration must contain {expected} joint/gripper values')
    data = mujoco.MjData(model)
    if contract['base']['joints']:
        set_base_pose(model, data, contract['base']['initial_pose'])
    offset = 0
    for arm in arms.values():
        for index, (joint_name, actuator_name) in enumerate(zip(arm['joints'], arm['actuators'])):
            qvalue = float(arm_configuration[offset + index])
            data.qpos[model.jnt_qposadr[_joint(model, joint_name)]] = qvalue
            data.ctrl[_actuator(model, actuator_name)] = qvalue
        gripper = arm['gripper']
        opening = float(np.clip(arm_configuration[offset + len(arm['joints'])], 0, 1))
        lo, hi = gripper['control_range']
        value = lo + (hi-lo)*opening
        for joint_name, sign in zip(gripper['joints'], gripper['joint_signs']):
            data.qpos[model.jnt_qposadr[_joint(model, joint_name)]] = value*sign
        data.ctrl[_actuator(model, gripper['actuator'])] = value
        offset += len(arm['joints']) + 1
    mujoco.mj_forward(model, data)
    return data.qpos.copy(), data.ctrl.copy()


def collision_report(model, data, tolerance=.002):
    """Report penetrating robot contacts; intended object contact remains visible."""
    result = []
    robot_root = model.body(runtime_contract(model)['base']['body']).id
    def robot_geom(geom):
        body = int(model.geom_bodyid[geom])
        while body:
            if body == robot_root: return True
            body = int(model.body_parentid[body])
        return False
    for c in data.contact[:data.ncon]:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f'geom_{g}' for g in (c.geom1,c.geom2)]
        if c.dist < -tolerance and any(robot_geom(g) for g in (c.geom1,c.geom2)):
            dynamic_task = False
            for geom in (c.geom1, c.geom2):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ''
                if robot_geom(geom): continue
                body = int(model.geom_bodyid[geom])
                while body:
                    if model.body_jntnum[body] > 0:
                        dynamic_task = True
                        break
                    body = int(model.body_parentid[body])
            result.append({'geoms':names,'penetration_m':float(-c.dist), 'dynamic_task_body': dynamic_task})
    return result


def solve_ik(model, data, targets):
    """Damped least-squares synchronized arm IK; reports, never hides collisions."""
    from scipy.spatial.transform import Rotation
    if not targets or not set(targets).issubset({'left','right'}):
        raise ValueError('targets require left and/or right')
    contract = runtime_contract(model)
    joints = [model.joint(name).id for side in targets for name in contract['arms'][side]['joints']]
    dofs = np.array([model.jnt_dofadr[j] for j in joints])
    qadr = np.array([model.jnt_qposadr[j] for j in joints])
    def residual():
        errors={}; jac=[]; vec=[]
        for side, target in targets.items():
            site=model.site(contract['arms'][side]['eef_site']).id
            pos=np.asarray(target['position'],float)
            if pos.shape!=(3,) or not np.isfinite(pos).all(): raise ValueError('invalid position')
            ep=pos-data.site_xpos[site]
            jp=np.zeros((3,model.nv)); jr=jp.copy()
            mujoco.mj_jacSite(model,data,jp,jr,site)
            jac.append(jp[:,dofs]); vec.append(ep)
            angle=0.
            if 'quaternion_wxyz' in target:
                q=np.asarray(target['quaternion_wxyz'],float)
                if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9: raise ValueError('invalid quaternion')
                desired=Rotation.from_quat(q[[1,2,3,0]])
                er=(desired*Rotation.from_matrix(data.site_xmat[site].reshape(3,3)).inv()).as_rotvec()
                angle=float(np.linalg.norm(er)); jac.append(.2*jr[:,dofs]); vec.append(.2*er)
            errors[side]={'position_m':float(np.linalg.norm(ep)),'orientation_rad':angle}
        return errors,np.vstack(jac),np.concatenate(vec)
    for _ in range(180):
        mujoco.mj_forward(model,data)
        errors,jac,vec=residual()
        if all(e['position_m']<.004 and e['orientation_rad']<.08 for e in errors.values()): break
        step=jac.T@np.linalg.solve(jac@jac.T+.001*np.eye(len(vec)),vec)
        data.qpos[qadr]=np.clip(data.qpos[qadr]+np.clip(step,-.12,.12),model.jnt_range[joints,0],model.jnt_range[joints,1])
    mujoco.mj_forward(model,data)
    errors,_,_=residual()
    all_contacts=collision_report(model,data)
    collisions=[c for c in all_contacts if not c['dynamic_task_body']]
    dynamic_contacts=[c for c in all_contacts if c['dynamic_task_body']]
    margin=np.minimum(data.qpos[qadr]-model.jnt_range[joints,0],model.jnt_range[joints,1]-data.qpos[qadr])
    return {'success':all(e['position_m']<.008 and e['orientation_rad']<.12 for e in errors.values()) and not collisions,'errors':errors,'collisions':collisions,'dynamic_object_contacts_unverified':dynamic_contacts,'dynamic_object_note':'Task object poses remain at initial state during IK; moved-object collisions require rollout.','joint_margin_rad':float(margin.min())}
