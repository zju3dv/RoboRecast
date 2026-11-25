from __future__ import annotations

import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .util import read_json, sha256_file, tree_sha256, write_json
from .image_geometry import proxy_camera_geometry


ARM_JOINTS = ["panda_joint{}".format(index) for index in range(1, 8)]
ARM_POSITION_ACTUATORS = ["arm_position{}".format(index) for index in range(1, 8)]
ARM_VELOCITY_ACTUATORS = ["arm_velocity{}".format(index) for index in range(1, 8)]
CAMERAS = ["ext1", "ext2", "wrist"]


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _atomic_xml(path: Path, tree: ET.ElementTree) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    tree.write(str(temporary), encoding="utf-8", xml_declaration=True)
    os.replace(str(temporary), str(path))


def _section(root: ET.Element, name: str) -> ET.Element:
    section = root.find(name)
    if section is None:
        section = ET.SubElement(root, name)
    return section


def _source_fr3(episode_dir: Path) -> Path:
    return episode_dir / "raw/robot/FR3/droid/droid/robot_ik/franka/fr3.xml"


def _validate_robot_selection(input_root: Path, episode_dir: Path,
                              robot_spec: Dict[str, Any]) -> Path:
    selected = robot_spec.get("selected_robot", {})
    legacy = robot_spec.get("schema_version") == "real2sim-robot-spec-v1"
    if legacy:
        lock, arm = robot_spec.get("selection_lock", {}), robot_spec.get("authoritative_arm", {})
        valid = (lock.get("arm_short_name") == "FR3" and lock.get("arm_count") == 1 and
                 arm.get("degrees_of_freedom") == 7)
        arm_asset = {"episode_path": "raw/robot/FR3",
                     "shared_path_from_input_root": "shared/robot_assets/FR3"}
    else:
        valid = selected.get("family") in {"FR3", "Panda"} and selected.get("dof") == 7
        arm_asset = robot_spec.get("arm_asset", {})
    if not valid:
        raise ValueError("robot runtime requires the episode-selected 7-DoF FR3 or Panda")
    if not legacy:
        family = selected["family"]
        if (arm_asset.get("episode_path") != "raw/robot/" + family or
                arm_asset.get("shared_path_from_input_root") != "shared/robot_assets/" + family):
            raise ValueError("robot asset paths disagree with selected family")
    episode_link = episode_dir / arm_asset["episode_path"]
    resolved = episode_link.resolve()
    expected = (input_root / arm_asset["shared_path_from_input_root"]).resolve()
    if resolved != expected:
        raise ValueError("episode FR3 link does not resolve to the contracted shared asset")
    if legacy:
        expected_xml_hash = robot_spec["authoritative_arm"]["mujoco_model"]["sha256"]
        if sha256_file(_source_fr3(episode_dir)) != expected_xml_hash:
            raise ValueError("FR3 XML hash differs from legacy robot_spec.json")
    elif tree_sha256(expected) != arm_asset["tree_sha256"]:
        raise ValueError("shared FR3 asset hash differs from robot_spec.json")
    return expected


def _relative_asset_paths(root: ET.Element, source_xml: Path, output_xml: Path) -> None:
    compiler = root.find("compiler")
    mesh_dir = (source_xml.parent / ((compiler.get("meshdir") or "") if compiler is not None else "")).resolve()
    texture_dir = (source_xml.parent / ((compiler.get("texturedir") or "") if compiler is not None else "")).resolve()
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
        compiler.set("autolimits", "true")
    for element in root.findall("./asset/*"):
        filename = element.get("file")
        if not filename:
            continue
        directory = texture_dir if element.tag == "texture" else mesh_dir
        source = (directory / filename).resolve()
        element.set("file", os.path.relpath(str(source), str(output_xml.parent)))


def _load_initial_qpos(episode_dir: Path) -> np.ndarray:
    path = episode_dir / "raw/actions/actions.npz"
    with np.load(str(path), allow_pickle=False) as archive:
        values = np.asarray(archive["joint_position"][0], dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("initial FR3 joint_position must be seven finite radians")
    return values


def _camera_transform(camera_info: Dict[str, Any], episode_dir: Path, name: str) -> np.ndarray:
    if name != "wrist":
        extrinsics = camera_info[name]["extrinsics"]
        if "T_fr3_base_camera" in extrinsics:
            return np.asarray(extrinsics["T_fr3_base_camera"], dtype=np.float64)
        if extrinsics.get("convention") == "T_camera_world" and "matrix" in extrinsics:
            return np.linalg.inv(np.asarray(extrinsics["matrix"], dtype=np.float64))
        raise ValueError("{} camera lacks a supported base transform".format(name))
    path = episode_dir / "derived/calibration/wrist_extrinsics_canonical.npz"
    with np.load(str(path), allow_pickle=False) as archive:
        valid = np.asarray(archive["valid_mask"], dtype=bool)
        transforms = np.asarray(archive["T_panda_link8_camera"], dtype=np.float64)[valid]
    if not len(transforms):
        raise ValueError("wrist mount calibration has no valid transform")
    translations = transforms[:, :3, 3]
    if float(np.max(np.linalg.norm(translations - np.median(translations, axis=0), axis=1))) > 0.005:
        raise ValueError("fixed wrist mount translations disagree by more than 5 mm")
    return transforms[0]


def _camera_element(name: str, camera_info: Dict[str, Any], transform: np.ndarray) -> ET.Element:
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("{} camera transform must be finite 4x4".format(name))
    # DROID calibration uses OpenCV +Z-forward/+Y-down. MuJoCo cameras use
    # -Z-forward/+Y-up, so flip camera-local Y and Z while preserving position.
    rotation = transform[:3, :3].dot(np.diag([1.0, -1.0, -1.0]))
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    intrinsics = camera_info[name]["intrinsics"]
    fovy = math.degrees(2.0 * math.atan(float(intrinsics["height"]) /
                                        (2.0 * float(intrinsics["fy"]))))
    return ET.Element("camera", {
        "name": name,
        "pos": " ".join("{:.17g}".format(row) for row in transform[:3, 3]),
        "quat": " ".join("{:.17g}".format(row) for row in quat_wxyz),
        "fovy": "{:.17g}".format(fovy),
    })


def _complete_intrinsics(camera_info: Dict[str, Any], episode_dir: Path) -> Dict[str, Any]:
    for name, row in camera_info.items():
        intrinsics = row["intrinsics"]
        if "width" in intrinsics and "height" in intrinsics:
            continue
        video = episode_dir / "raw/videos/{}.mp4".format(name)
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=width,height", "-of", "json", str(video)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=False)
        if result.returncode != 0:
            raise ValueError("cannot infer legacy {} camera dimensions: {}".format(name, result.stderr))
        stream = json.loads(result.stdout)["streams"][0]
        intrinsics["width"], intrinsics["height"] = int(stream["width"]), int(stream["height"])
    return camera_info


def _attach_robotiq(root: ET.Element) -> None:
    template = ET.parse(str(_assets_dir() / "robotiq_2f85.xml")).getroot()
    link8 = root.find(".//body[@name='panda_link8']")
    if link8 is None:
        raise ValueError("FR3 source is missing panda_link8")
    for child in list(link8.findall("body")):
        link8.remove(child)
    ET.SubElement(link8, "site", {"name": "ee_site", "size": ".005"})
    link8.append(template.find("worldbody/body"))
    for name in ("equality", "contact"):
        destination = _section(root, name)
        source = template.find(name)
        if source is not None:
            for child in list(source):
                destination.append(child)


def _configure_robot(root: ET.Element, profile: Dict[str, Any]) -> None:
    for body in root.findall(".//body"):
        if (body.get("name") or "").startswith("panda_link"):
            body.set("gravcomp", "1")
    joints = {joint.get("name"): joint for joint in root.findall(".//joint")}
    for index, name in enumerate(ARM_JOINTS):
        if name not in joints:
            raise ValueError("FR3 source is missing {}".format(name))
        joints[name].set("damping", str(profile["joint_damping"][index]))
    actuator = _section(root, "actuator")
    actuator.clear()
    for index, joint in enumerate(ARM_JOINTS):
        limit = profile["velocity_force_limits_nm"][index]
        common = {"joint": joint, "forcerange": "-{} {}".format(limit, limit),
                  "forcelimited": "true"}
        ET.SubElement(actuator, "position", dict(common, name=ARM_POSITION_ACTUATORS[index],
                                                  kp=str(profile["position_kp"][index])))
    for index, joint in enumerate(ARM_JOINTS):
        limit = profile["force_limits_nm"][index]
        common = {"joint": joint, "forcerange": "-{} {}".format(limit, limit),
                  "forcelimited": "true"}
        ET.SubElement(actuator, "velocity", dict(common, name=ARM_VELOCITY_ACTUATORS[index],
                                                  kv=str(profile["velocity_kv"][index])))
    ET.SubElement(actuator, "position", {
        "name": "gripper_position", "joint": "right_finger_joint",
        "kp": str(profile["gripper_kp"]), "ctrlrange": "0 .8", "ctrllimited": "true",
        "forcerange": "-{0} {0}".format(profile["gripper_force_limit_n"]), "forcelimited": "true",
    })
    old_option = root.find("option")
    if old_option is not None:
        root.remove(old_option)
    ET.SubElement(root, "option", {"timestep": str(profile["timestep_sec"]),
                                   "integrator": "implicit", "iterations": "80", "cone": "elliptic"})


def _attach_cameras(root: ET.Element, camera_info: Dict[str, Any], episode_dir: Path) -> Dict[str, Any]:
    worldbody = _section(root, "worldbody")
    link8 = root.find(".//body[@name='panda_link8']")
    output = {}
    for name in CAMERAS:
        transform = _camera_transform(camera_info, episode_dir, name)
        camera = _camera_element(name, camera_info, transform)
        (link8 if name == "wrist" else worldbody).append(camera)
        geometry = proxy_camera_geometry(camera_info[name]["intrinsics"], 640, 360)
        output[name] = {
            "parent": "panda_link8" if name == "wrist" else "robot_base",
            "T_parent_camera_opencv": transform.tolist(),
            "intrinsics": geometry["source_intrinsics"],
            "source_intrinsics": geometry["source_intrinsics"],
            "proxy_intrinsics": geometry["proxy_intrinsics"],
            "source_to_proxy_pixel": geometry["source_to_proxy_pixel"],
            "fovy_deg": float(camera.get("fovy")),
        }
    return output


def _update_image_geometry_specs(episode_dir: Path, camera_info: Dict[str, Any]) -> None:
    geometry = {name: proxy_camera_geometry(camera_info[name]["intrinsics"], 640, 360)
                for name in CAMERAS}
    camera_spec_path = episode_dir / "specs/camera_spec.json"
    if camera_spec_path.is_file():
        camera_spec = read_json(camera_spec_path)
        camera_spec["image_geometry"] = geometry
        write_json(camera_spec_path, camera_spec)
    proxy_spec_path = episode_dir / "specs/proxy_video_spec.json"
    if proxy_spec_path.is_file():
        proxy_spec = read_json(proxy_spec_path)
        for name in CAMERAS:
            if name in proxy_spec.get("streams", {}):
                proxy_spec["streams"][name]["image_geometry"] = geometry[name]
        write_json(proxy_spec_path, proxy_spec)


def _control_indices(model: mujoco.MjModel, names: Iterable[str]) -> List[int]:
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]


def validate_controller_response(scene_path: Path, contract: Dict[str, Any]) -> Dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    q0 = np.asarray(contract["initial_arm_qpos_rad"], dtype=np.float64)
    data.qpos[:7] = q0
    position_ids = _control_indices(model, contract["arm_position_actuators"])
    velocity_ids = _control_indices(model, contract["arm_velocity_actuators"])
    data.ctrl[position_ids] = q0
    data.ctrl[velocity_ids] = 0.0
    mujoco.mj_forward(model, data)
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, contract["end_effector_body"])
    positions = [data.xpos[ee].copy()]
    joint_errors = []
    steps = int(round(float(contract["controller_regression_limits"]["duration_sec"]) /
                      model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        positions.append(data.xpos[ee].copy())
        joint_errors.append(float(np.max(np.abs(data.qpos[:7] - q0))))
    positions_array = np.asarray(positions)
    path = float(np.sum(np.linalg.norm(np.diff(positions_array, axis=0), axis=1)))
    vertical_range = float(np.ptp(positions_array[:, 2]))
    limits = contract["controller_regression_limits"]
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    within_limits = bool(np.all(data.qpos[:7] >= model.jnt_range[:7, 0]) and
                         np.all(data.qpos[:7] <= model.jnt_range[:7, 1]))
    max_error = max(joint_errors or [0.0])
    # A small joint step exercises the damping path. Assess the final tail rather
    # than requiring a particular transient shape or robot inertia approximation.
    target = q0.copy()
    target[0] = min(target[0] + limits["step_joint_rad"], model.jnt_range[0, 1] - 0.05)
    data.ctrl[position_ids] = target
    step_positions = []
    step_velocities = []
    step_count = int(round(limits["step_duration_sec"] / model.opt.timestep))
    for _ in range(step_count):
        mujoco.mj_step(model, data)
        step_positions.append(float(data.qpos[0]))
        step_velocities.append(float(data.qvel[0]))
    tail_count = max(1, int(round(0.1 / model.opt.timestep)))
    step_final_error = float(abs(np.mean(step_positions[-tail_count:]) - target[0]))
    step_tail_velocity = float(np.max(np.abs(step_velocities[-tail_count:])))
    direction = 1.0 if target[0] >= q0[0] else -1.0
    step_overshoot = float(max(0.0, np.max(direction * (np.asarray(step_positions) - target[0]))))
    gripper_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                        contract["gripper"]["position_actuator"])
    data.ctrl[gripper_actuator] = 0.4
    for _ in range(int(round(0.3 / model.opt.timestep))):
        mujoco.mj_step(model, data)
    right = data.qpos[model.joint("right_finger_joint").qposadr[0]]
    left = data.qpos[model.joint("left_finger_joint").qposadr[0]]
    gripper_coupling_error = float(abs(right - left))
    finite = finite and bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    passed = (finite and within_limits and path <= limits["hold_ee_path_max_m"] and
              vertical_range <= limits["hold_vertical_range_max_m"] and
              max_error <= limits["hold_joint_error_max_rad"] and
              step_final_error <= limits["step_final_error_max_rad"] and
              step_tail_velocity <= limits["step_tail_velocity_max_rad_s"] and
              step_overshoot <= limits["step_overshoot_max_rad"] and
              gripper_coupling_error <= limits["gripper_coupling_error_max_rad"])
    return {"schema_version": "real2sim-controller-regression-v1",
            "status": "PASS" if passed else "FAIL", "finite_state": finite,
            "joint_limit_violation": not within_limits, "hold_ee_path_m": path,
            "hold_vertical_range_m": vertical_range, "hold_joint_error_max_rad": max_error,
            "step_final_error_rad": step_final_error,
            "step_tail_velocity_rad_s": step_tail_velocity,
            "step_overshoot_rad": step_overshoot,
            "gripper_coupling_error_rad": gripper_coupling_error,
            "limits": limits}


def _name_map() -> Dict[str, Any]:
    return {
        "schema_version": "real2sim-robot-name-map-v1",
        "body": {name: name for name in ["panda_link{}".format(i) for i in range(9)] +
                 ["robotiq_arg2f_base_link", "left_inner_finger", "right_inner_finger"]},
        "joint": {name: name for name in ARM_JOINTS + ["right_finger_joint", "right_inner_joint",
                                                              "left_finger_joint", "left_inner_joint"]},
        "geom": {name: name for name in ["left_finger_pad", "right_finger_pad"]},
        "site": {"ee_site": "ee_site"},
        "actuator": {name: name for name in ARM_POSITION_ACTUATORS + ARM_VELOCITY_ACTUATORS +
                     ["gripper_position"]},
        "camera": {name: name for name in CAMERAS},
    }


def build_robot_runtime(input_root: Path, episode_dir: Path) -> Dict[str, Any]:
    input_root, episode_dir = Path(input_root).resolve(), Path(episode_dir).resolve()
    robot_spec = read_json(episode_dir / "specs/robot_spec.json")
    shared_fr3 = _validate_robot_selection(input_root, episode_dir, robot_spec)
    family = robot_spec.get("selected_robot", {}).get("family", "FR3")
    source_xml = (_source_fr3(episode_dir) if family == "FR3" else
                  shared_fr3 / "droid/droid/robot_ik/franka/panda.xml")
    if not source_xml.is_file():
        raise FileNotFoundError(source_xml)
    output_dir = episode_dir / "derived/robot_runtime"
    scene_path = output_dir / "robot_base_scene.xml"
    profile = read_json(_assets_dir() / "fr3_controller.json")
    camera_info = _complete_intrinsics(read_json(episode_dir / "raw/calibration/camera_info.json"), episode_dir)
    _update_image_geometry_specs(episode_dir, camera_info)
    initial_qpos = _load_initial_qpos(episode_dir)
    tree = ET.parse(str(source_xml))
    root = tree.getroot()
    root.set("model", family + "_Robotiq_2F85_episode_runtime")
    _relative_asset_paths(root, source_xml, scene_path)
    _attach_robotiq(root)
    _configure_robot(root, profile)
    cameras = _attach_cameras(root, camera_info, episode_dir)
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)
    keyframe = ET.SubElement(root, "keyframe")
    initial_gripper = [0.0, 0.0, 0.0, 0.0]
    initial_ctrl = initial_qpos.tolist() + [0.0] * 7 + [0.0]
    ET.SubElement(keyframe, "key", {
        "name": "episode_initial",
        "qpos": " ".join("{:.17g}".format(row) for row in initial_qpos.tolist() + initial_gripper),
        "ctrl": " ".join("{:.17g}".format(row) for row in initial_ctrl),
    })
    _atomic_xml(scene_path, tree)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if not np.all((initial_qpos >= model.jnt_range[:7, 0]) &
                  (initial_qpos <= model.jnt_range[:7, 1])):
        raise ValueError("episode initial qpos is outside FR3 joint limits")
    names = _name_map()
    contract = {
        "schema_version": "real2sim-robot-runtime-v1", "robot_family": family,
        "mujoco_version": mujoco.__version__, "arm_joint_names": ARM_JOINTS,
        "arm_position_actuators": ARM_POSITION_ACTUATORS,
        "arm_velocity_actuators": ARM_VELOCITY_ACTUATORS,
        "end_effector_body": "panda_link8", "initial_arm_qpos_rad": initial_qpos.tolist(),
        "gripper": {"model": "Robotiq 2F-85", "command_unit": "opening_width_m",
                    "maximum_width_m": 0.085, "driver_joint": "right_finger_joint",
                    "position_actuator": "gripper_position", "closed_driver_rad": 0.8,
                    "open_driver_rad": 0.0, "left_pad_geom": "left_finger_pad",
                    "right_pad_geom": "right_finger_pad"},
        "cameras": cameras, "controller_profile": profile["profile"],
        "controller_regression_limits": profile["regression"],
        "trajectory_contract": {"timestamp_unit": "second", "joint_position_unit": "radian",
                                "gripper_command_unit": "opening_width_m"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "robot_runtime_contract.json"
    name_map_path = output_dir / "name_map.json"
    write_json(contract_path, contract)
    write_json(name_map_path, names)
    regression = validate_controller_response(scene_path, contract)
    if regression["status"] != "PASS":
        raise ValueError("FR3 controller regression failed: {}".format(regression))
    manifest = {
        "schema_version": "real2sim-robot-runtime-manifest-v1",
        "scene_path": "derived/robot_runtime/robot_base_scene.xml",
        "runtime_contract_path": "derived/robot_runtime/robot_runtime_contract.json",
        "name_map_path": "derived/robot_runtime/name_map.json",
        "source_fr3_path": ("raw/robot/" + family + "/droid/droid/robot_ik/franka/" + ("fr3.xml" if family == "FR3" else "panda.xml")),
        "source_robot_family": family,
        "source_fr3_tree_sha256": tree_sha256(shared_fr3),
        "files": {"robot_base_scene.xml": sha256_file(scene_path),
                  "robot_runtime_contract.json": sha256_file(contract_path),
                  "name_map.json": sha256_file(name_map_path)},
        "compiled": {"mujoco_version": mujoco.__version__, "nq": model.nq,
                     "nv": model.nv, "nu": model.nu, "nbody": model.nbody,
                     "ngeom": model.ngeom, "ncam": model.ncam},
        "controller_regression": regression,
    }
    manifest_path = output_dir / "build_manifest.json"
    write_json(manifest_path, manifest)
    return {"schema_version": "real2sim-robot-runtime-build-v1", "status": "PASS",
            "runtime_dir": output_dir.as_posix(), "manifest_path": manifest_path.as_posix(),
            "manifest_entry": manifest, "controller_regression": regression}
