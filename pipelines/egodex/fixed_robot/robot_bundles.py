"""Install self-contained, independently maintained Human robot snapshots."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from .robot import build_robot_asset, initial_state, VX
from .robot_runtime import runtime_contract
from .utils import sha256, write_json

def install_robot_bundle(output_dir: Path, motion_mode: str, xyyaw=(0,0,0), bundle_id='fixed_dual_vx300s') -> dict:
    if bundle_id != 'fixed_dual_vx300s':
        raise ValueError(f'Unsupported robot bundle {bundle_id}; a validated installer is required')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = build_robot_asset(output_dir / 'robot.xml', xyyaw, motion_mode=motion_mode)
    model = mujoco.MjModel.from_xml_path(str(scene))
    contract = runtime_contract(model)
    contract['resolved_limits'] = {
        side: dict(joint_ranges=[model.jnt_range[model.joint(n).id].tolist() for n in arm['joints']],
                   actuator_ranges=[model.actuator_ctrlrange[model.actuator(n).id].tolist() for n in arm['actuators']])
        for side, arm in contract['arms'].items()}
    contract['physics'] = dict(timestep_s=float(model.opt.timestep), integrator=int(model.opt.integrator),
                             finger_solimp=[.9,.95,.001])
    contract_path = output_dir / 'runtime_contract.json'
    write_json(contract_path, contract)
    qpos, ctrl = initial_state(model)
    state_path = output_dir / 'robot_initial_state.npz'
    np.savez_compressed(state_path, qpos=qpos, qvel=np.zeros(model.nv), ctrl=ctrl)
    files = [scene, state_path, contract_path, *sorted((output_dir / 'robot_assets').iterdir())]
    hashes = {str(p.relative_to(output_dir)): sha256(p) for p in files if p.is_file()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = dict(schema_version='2.0', bundle_id=bundle_id, motion_mode=motion_mode, identity=identity,
                    files=hashes, source='VX300S assets with a fixed dual-arm assembly',
                    source_asset_sha256=sha256(VX / 'vx300s.xml'),
                    mujoco_version=mujoco.__version__)
    manifest_path = output_dir / 'manifest.json'
    write_json(manifest_path, manifest)
    return dict(bundle_id=bundle_id, motion_mode=motion_mode, identity=identity,
                scene_xml=str(scene.resolve()), initial_state=str(state_path.resolve()),
                runtime_contract=str(contract_path.resolve()), manifest=str(manifest_path.resolve()))
