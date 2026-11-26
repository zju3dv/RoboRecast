from __future__ import annotations

import json
from pathlib import Path

import mujoco


def validate_runtime_contract(scene: Path, contract_path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    contract = json.loads(Path(contract_path).read_text())
    missing: list[str] = []

    def exists(kind: mujoco.mjtObj, name: str) -> bool:
        return mujoco.mj_name2id(model, kind, name) >= 0

    for name in contract.get("base_pose_joints", []):
        if not exists(mujoco.mjtObj.mjOBJ_JOINT, name):
            missing.append(f"joint:{name}")
    if not exists(mujoco.mjtObj.mjOBJ_BODY, contract.get("base_body", "")):
        missing.append(f"body:{contract.get('base_body')}")
    for side, arm in contract.get("arms", {}).items():
        for name in arm.get("joint_names", []):
            if not exists(mujoco.mjtObj.mjOBJ_JOINT, name):
                missing.append(f"joint:{name}")
        for name in arm.get("actuator_names", []) + [arm.get("gripper_actuator", "")]:
            if not exists(mujoco.mjtObj.mjOBJ_ACTUATOR, name):
                missing.append(f"actuator:{name}")
        for name in [arm.get("end_effector_site", "")] + arm.get("finger_pad_sites", []):
            if not exists(mujoco.mjtObj.mjOBJ_SITE, name):
                missing.append(f"site:{name}")
    for role, name in contract.get("cameras", {}).items():
        if role in ("top", "left_wrist", "right_wrist") and not exists(
                mujoco.mjtObj.mjOBJ_CAMERA, name):
            missing.append(f"camera:{name}")
    if missing:
        raise ValueError(f"runtime contract names missing: {missing}")
    return {"nq": model.nq, "nv": model.nv, "nu": model.nu, "ncam": model.ncam,
            "missing": missing}
