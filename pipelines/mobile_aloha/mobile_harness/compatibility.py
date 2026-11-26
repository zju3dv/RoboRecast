"""Internal physical provenance and state-layout compatibility helpers."""
import hashlib
from pathlib import Path

import numpy as np


def execution_hash():
    root = Path(__file__).parent
    control = root.parent / 'mobile_robot/control.py'
    return hashlib.sha256(b''.join(p.read_bytes() for p in (
        root / 'rollout.py', root / 'stages.py', root / 'hand_objects.py', Path(__file__), control))).hexdigest()


def physical_match(record, scene_hash, settled_hash, engine_hash):
    return all(record.get(key) == value for key, value in (
        ('scene_sha256', scene_hash), ('settled_sha256', settled_hash),
        ('engine_sha256', engine_hash)))


def longest_prefix(records, hashes, limit, scene_hash, settled_hash, engine_hash):
    best, count = None, 0
    for record in reversed(records):
        if not physical_match(record, scene_hash, settled_hash, engine_hash):
            continue
        matched = 0
        available = min(limit, record.get('passed_prefix', 0))
        for old, new in zip(record.get('stage_hashes', [])[:available], hashes[:available]):
            if old != new or matched >= len(record.get('checkpoint_paths', [])):
                break
            matched += 1
        if matched > count:
            best, count = record, matched
    return best, count


def settle_layout_signature(model):
    """Compiled geometry/layout identity, deliberately excluding mass/friction/damping.

    Includes mesh contents, joint ordering, actuator mapping and reference poses.
    Physics-only edits may alter inertia without altering these arrays.
    """
    digest = hashlib.sha256(bytes(model.names))
    fields = (
        'qpos0', 'body_parentid', 'body_pos', 'body_quat',
        'jnt_type', 'jnt_bodyid', 'jnt_qposadr', 'jnt_dofadr', 'jnt_pos', 'jnt_axis',
        'jnt_range', 'jnt_limited', 'geom_type', 'geom_bodyid', 'geom_dataid',
        'geom_pos', 'geom_quat', 'geom_size', 'geom_contype', 'geom_conaffinity',
        'geom_margin', 'geom_gap', 'mesh_vert', 'mesh_face',
        'site_bodyid', 'site_pos', 'site_quat', 'site_size',
        'actuator_trntype', 'actuator_trnid', 'actuator_gear', 'actuator_ctrlrange',
        'eq_type', 'eq_obj1id', 'eq_obj2id', 'eq_data', 'exclude_signature',
        'tendon_adr', 'tendon_num', 'wrap_type', 'wrap_objid', 'wrap_prm',
    )
    for name in fields:
        value = np.asarray(getattr(model, name))
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
