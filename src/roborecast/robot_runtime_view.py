from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import mujoco

from .errors import StructuredToolError


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: Any,
              field: str) -> int:
    if not isinstance(name, str) or not name:
        raise StructuredToolError(
            "INVALID_ROBOT_RUNTIME", "Robot runtime names must be non-empty strings.",
            field, "A non-empty MuJoCo object name.", name, None)
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise StructuredToolError(
            "ROBOT_RUNTIME_NAME_MISSING", "Robot runtime references an absent MuJoCo object.",
            field, "A name present in the assembled scene.", name, None)
    return int(object_id)


def _name_sequence(contract: Dict[str, Any], key: str) -> Tuple[str, ...]:
    value = contract.get(key)
    if not isinstance(value, list) or not value or any(
            not isinstance(name, str) or not name for name in value):
        raise StructuredToolError(
            "INVALID_ROBOT_RUNTIME", "Robot runtime name lists must be non-empty string arrays.",
            "$.runtime.{}".format(key), "A non-empty array of MuJoCo names.", value, None)
    if len(set(value)) != len(value):
        raise StructuredToolError(
            "INVALID_ROBOT_RUNTIME", "Robot runtime name lists cannot contain duplicates.",
            "$.runtime.{}".format(key), "Unique MuJoCo names.", value, None)
    return tuple(value)


@dataclass(frozen=True)
class RobotRuntimeView:
    """Resolved, model-specific addresses for an immutable robot runtime contract."""

    arm_joint_names: Tuple[str, ...]
    arm_joint_ids: Tuple[int, ...]
    arm_qpos_addresses: Tuple[int, ...]
    arm_dof_addresses: Tuple[int, ...]
    position_actuator_ids: Tuple[int, ...]
    velocity_actuator_ids: Tuple[int, ...]
    gripper_actuator_id: int
    end_effector_body_id: int

    @property
    def arm_dof(self) -> int:
        return len(self.arm_joint_names)

    @classmethod
    def from_contract(cls, model: mujoco.MjModel,
                      contract: Dict[str, Any]) -> "RobotRuntimeView":
        if not isinstance(contract, dict):
            raise StructuredToolError(
                "INVALID_ROBOT_RUNTIME", "Robot runtime must be a JSON object.",
                "$.runtime", "A robot runtime contract object.", contract, None)
        joint_names = _name_sequence(contract, "arm_joint_names")
        position_names = _name_sequence(contract, "arm_position_actuators")
        velocity_names = _name_sequence(contract, "arm_velocity_actuators")
        if len(position_names) != len(joint_names) or len(velocity_names) != len(joint_names):
            raise StructuredToolError(
                "ROBOT_RUNTIME_DOF_MISMATCH",
                "Arm joint, position-actuator, and velocity-actuator counts must agree.",
                "$.runtime", "Equal-length arm name arrays.",
                {"joints": len(joint_names), "position_actuators": len(position_names),
                 "velocity_actuators": len(velocity_names)}, None)

        joint_ids = tuple(_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name,
                                    "$.runtime.arm_joint_names[{}]".format(index))
                          for index, name in enumerate(joint_names))
        single_dof_types = {
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        invalid = [joint_names[index] for index, joint_id in enumerate(joint_ids)
                   if int(model.jnt_type[joint_id]) not in single_dof_types]
        if invalid:
            raise StructuredToolError(
                "ROBOT_RUNTIME_JOINT_TYPE_UNSUPPORTED",
                "Arm joints must each contribute exactly one qpos and one velocity DoF.",
                "$.runtime.arm_joint_names", "Only hinge or slide joints.", invalid, None)

        position_ids = tuple(_named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name,
            "$.runtime.arm_position_actuators[{}]".format(index))
            for index, name in enumerate(position_names))
        velocity_ids = tuple(_named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name,
            "$.runtime.arm_velocity_actuators[{}]".format(index))
            for index, name in enumerate(velocity_names))
        for field, actuator_ids in (("arm_position_actuators", position_ids),
                                    ("arm_velocity_actuators", velocity_ids)):
            actual_joint_ids = tuple(int(model.actuator_trnid[actuator_id, 0])
                                     for actuator_id in actuator_ids)
            if actual_joint_ids != joint_ids:
                raise StructuredToolError(
                    "ROBOT_RUNTIME_ACTUATOR_ORDER_MISMATCH",
                    "Arm actuators must address arm joints in declared joint order.",
                    "$.runtime.{}".format(field), list(joint_names),
                    [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                     if joint_id >= 0 else None for joint_id in actual_joint_ids], None)

        gripper = contract.get("gripper")
        if not isinstance(gripper, dict):
            raise StructuredToolError(
                "INVALID_ROBOT_RUNTIME", "Robot runtime gripper must be an object.",
                "$.runtime.gripper", "A gripper runtime object.", gripper, None)
        gripper_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper.get("position_actuator"),
            "$.runtime.gripper.position_actuator")
        ee_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY,
                          contract.get("end_effector_body"),
                          "$.runtime.end_effector_body")
        return cls(
            arm_joint_names=joint_names,
            arm_joint_ids=joint_ids,
            arm_qpos_addresses=tuple(int(model.jnt_qposadr[joint_id])
                                     for joint_id in joint_ids),
            arm_dof_addresses=tuple(int(model.jnt_dofadr[joint_id])
                                    for joint_id in joint_ids),
            position_actuator_ids=position_ids,
            velocity_actuator_ids=velocity_ids,
            gripper_actuator_id=gripper_id,
            end_effector_body_id=ee_id,
        )
