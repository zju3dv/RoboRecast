import json
import numpy as np
import mujoco

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

FIELDS = {'model', 'controller', 'prefix', 'contract'}

def capture(model, data):
    state = np.empty(mujoco.mj_stateSize(model, SPEC))
    mujoco.mj_getState(model, data, state, SPEC)
    return state

def save(path, model, data, identity, validated=False, touched_geoms=None):
    if set(identity) != FIELDS or not all(identity.values()) or not validated:
        raise ValueError('Validated prefix and complete dependency identity required')
    proof = {} if touched_geoms is None else {'friction': model.geom_friction.copy(), 'touched_geoms': touched_geoms}
    np.savez_compressed(path, state=capture(model, data), identity=json.dumps(identity, sort_keys=True), spec=int(SPEC), **proof)

def supports_local_friction(model):
    integrators = (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICIT,
                   mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    callbacks = ('get_mjcb_control', 'get_mjcb_passive', 'get_mjcb_contactfilter', 'get_mjcb_sensor',
                 'get_mjcb_act_dyn', 'get_mjcb_act_gain', 'get_mjcb_act_bias')
    return (model.nplugin == 0 and int(model.opt.integrator) in tuple(map(int, integrators))
            and all(getattr(mujoco, name)() is None for name in callbacks))

def _local_friction_edits(path, model, old_hash):
    """Reconstruct the entire old compiled model by undoing only cold world-geom friction edits."""
    import copy
    from .rollout import model_signature
    if not supports_local_friction(model):
        return []
    with np.load(path, allow_pickle=False) as state:
        if 'friction' not in state or 'touched_geoms' not in state:
            return []
        old = state['friction']
        touched = np.unpackbits(state['touched_geoms'])[:model.ngeom]
    if old.shape != model.geom_friction.shape or len(touched) != model.ngeom:
        return []
    changed = np.flatnonzero(np.any(model.geom_friction != old, axis=1))
    bodies = model.geom_bodyid[changed]
    if not len(changed) or np.any(model.body_weldid[bodies] != 0) or np.any(touched[changed]):
        return []
    for body in bodies:
        while body:
            if model.body_mocapid[body] >= 0:
                return []
            body = model.body_parentid[body]
    original = copy.copy(model)
    original.geom_friction[:] = old
    return changed.tolist() if model_signature(original) == old_hash else []

def restore(path, model, data, identity, local_friction=False):
    from .rollout import model_signature
    if identity.get('model') != model_signature(model) and not (local_friction and _local_friction_edits(path, model, identity['model'])):
        raise ValueError('Compiled model changed; restart from zero')
    with np.load(path, allow_pickle=False) as archive:
        if set(identity) != FIELDS or json.loads(str(archive['identity'])) != identity:
            raise ValueError('Checkpoint invalidated by dependency changes; restart from zero')
        state = archive['state']
        if int(archive['spec']) != int(SPEC) or state.shape != (mujoco.mj_stateSize(model, SPEC),) or not np.isfinite(state).all():
            raise ValueError('Invalid integration state')
        mujoco.mj_setState(model, data, state, SPEC)
        # Forward refreshes derived quantities, but preserve warm-start and other integration state.
        mujoco.mj_forward(model, data)
        mujoco.mj_setState(model, data, state, SPEC)
    return float(data.time)

def replay_suffix(path, model, identity, dense, boundary_index):
    """Execute recorded controls from a validated checkpoint; returns fresh states.

    Full trajectories remain the default for optimization. This conservative replay
    API never authorizes a changed prefix or model and exposes numerical equivalence.
    """
    data = mujoco.MjData(model)
    restore(path, model, data, identity)
    if not np.isclose(data.time, dense['timestamp_sec'][boundary_index], atol=1e-10):
        raise ValueError('Checkpoint boundary does not match evidence')
    states = []
    for index in range(boundary_index + 1, len(dense['timestamp_sec'])):
        data.ctrl[:] = dense['ctrl'][index]
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        states.append(capture(model, data))
    return np.asarray(states)

def prefix_identity(model, runtime, expanded, requirements, intervals, index):
    from .requirements import digest
    from .rollout import model_signature
    times = np.arange(index+1) * model.opt.timestep
    # A prefix beyond a shortened new trajectory is always incompatible.
    if times[-1] > expanded.timestamp_sec[-1] + 1e-9:
        return None
    positions = np.column_stack([np.interp(times, expanded.timestamp_sec, expanded.joint_position[:, j]) for j in range(expanded.joint_position.shape[1])])
    width = np.interp(times, expanded.timestamp_sec, expanded.gripper_width_m)
    velocity = None
    if runtime.get('velocity_feedforward'):
        gradients = np.gradient(expanded.joint_position, expanded.timestamp_sec, axis=0)
        velocity = np.column_stack([np.interp(times, expanded.timestamp_sec, gradients[:, j]) for j in range(gradients.shape[1])]).tolist()
    boundary = float(times[-1] + model.opt.timestep)
    prefix_intervals = {key: [[start, min(end, boundary)] for start, end in windows if start < boundary - 1e-9] for key, windows in intervals.items()}
    return {'model': model_signature(model), 'controller': digest(runtime), 'contract': digest(requirements),
            'prefix': digest({'positions': positions.tolist(), 'width': width.tolist(), 'velocity': velocity, 'intervals': prefix_intervals})}

def select_prefix(root, candidates, model, runtime, expanded, contract, intervals):
    """Select the longest compatible, unmodified saved prefix across earlier rounds."""
    import hashlib
    from pathlib import Path
    root = Path(root)
    for checkpoint in sorted(candidates, key=lambda row: row['index'], reverse=True):
        expected = prefix_identity(model, runtime, expanded, contract, intervals, checkpoint['index'])
        old_identity = checkpoint['identity']
        if expected is None or any(expected[k] != old_identity[k] for k in ('controller', 'prefix', 'contract')):
            continue
        path, evidence_path = root / checkpoint['path'], root / checkpoint['evidence_path']
        if (not path.is_file() or not evidence_path.is_file() or
                hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint.get('state_sha256') or
                hashlib.sha256(evidence_path.read_bytes()).hexdigest() != checkpoint.get('evidence_sha256')):
            continue
        local_edits = []
        if expected['model'] != old_identity['model']:
            # Derived scene-wide observation predicates must be freshly assessed.
            prefix_checks = [p for event in contract['I'] for p in event['predicates']] + contract['G']
            if any(not p['key'].startswith(('joint:', 'body:', 'contact:')) for p in prefix_checks):
                continue
            local_edits = _local_friction_edits(path, model, old_identity['model'])
            if not local_edits:
                continue
        return {**checkpoint, 'path': path, 'evidence_path': evidence_path, 'changed_static_friction_geoms': local_edits}
    return None


def save_prefixes(root, output_dir, round_number, model, runtime, expanded, contract, intervals, dense, success_spec):
    """Shared accepted-event-boundary checkpoint policy for both execution entry points."""
    import hashlib
    from pathlib import Path
    from .requirements import evaluate
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    checkpoints = []
    boundaries = sorted({int(np.argmin(abs(dense['timestamp_sec'] - end))) for windows in intervals.values() for _, end in windows})
    limits = [float(check['max_m']) for check in success_spec['required_checks'] if check['type'] == 'maximum_penetration']
    for index in boundaries:
        boundary = float(dense['timestamp_sec'][index])
        completed = [event for event in contract['I'] if intervals.get(event['id']) and
                     max(end for _, end in intervals[event['id']]) <= boundary + model.opt.timestep * .1]
        completed_ids = {event['id'] for event in completed}
        if not completed or any(set(event.get('after', [])) - completed_ids for event in completed):
            continue
        prefix = {key: value[:index+1] for key, value in dense.items() if value.ndim > 0}
        if not all(np.isfinite(prefix[key]).all() for key in ('qpos', 'qvel', 'ctrl')):
            continue
        if limits and float(np.max(prefix['penetration_m'])) > min(limits):
            continue
        if 'unexpected_robot_contact' in prefix:
            from roborecast.motion_validity import velocity_report
            velocity = velocity_report(prefix['timestamp_sec'], prefix['arm_joint_velocity'],
                runtime['arm_joint_names'], runtime.get('motion_validation', {}).get('velocity_limits_rad_s'))
            from roborecast.motion_validity import dwell_report
            holding = dwell_report(expanded.source_path, prefix)
            if np.any(prefix['unexpected_robot_contact'][:, 0] >= 0) or velocity['status'] != 'PASS' or holding['status'] != 'PASS':
                continue
        if evaluate({'I': completed, 'c': [], 'G': contract['G']}, intervals, prefix)['status'] != 'PASS':
            continue
        identity = prefix_identity(model, runtime, expanded, contract, intervals, index)
        mujoco.mj_setState(model, data, dense['integration_state'][index], SPEC)
        path = output_dir / ('checkpoint_r%03d_%06d.npz' % (round_number, index))
        evidence_path = path.with_suffix('.evidence.npz')
        save(path, model, data, identity, validated=True,
             touched_geoms=prefix['contacted_geom_bits'][-1] if 'contacted_geom_bits' in prefix else None)
        np.savez_compressed(evidence_path, **prefix)
        checkpoints.append({'path': path.relative_to(root).as_posix(), 'evidence_path': evidence_path.relative_to(root).as_posix(),
                            'index': index, 'identity': identity, 'event_ids': sorted(completed_ids),
                            'review_verdict_path': ((output_dir.parent if output_dir.name == 'checkpoints' else output_dir) / 'prefix_review.json').relative_to(root).as_posix(),
                            'state_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'evidence_sha256': hashlib.sha256(evidence_path.read_bytes()).hexdigest()})
    return checkpoints


