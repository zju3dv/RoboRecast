"""Measure recorded robot motion, independently of planned/video durations.

This is a soft timing diagnostic, never a physical acceptance decision. A
censored recording cannot prove natural operation duration or completion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': _sha(path)}


def motion_window(timestamps, positions, quaternions_wxyz, linear_threshold=.005,
                  angular_threshold=.05):
    """Unsmooth interval speeds; include internal dwell in the active envelope."""
    t = np.asarray(timestamps, float)
    p = np.asarray(positions, float)
    q = np.asarray(quaternions_wxyz, float)
    if (len(t) < 2 or t.ndim != 1 or p.shape != (len(t), 3) or q.shape != (len(t), 4)
            or not all(np.isfinite(a).all() for a in (t, p, q))
            or np.any(np.diff(t) <= 0) or np.any(np.linalg.norm(q, axis=1) < 1e-12)):
        raise ValueError('Finite ordered timestamps and complete EE poses required')
    if not np.isfinite([linear_threshold, angular_threshold]).all() or min(linear_threshold, angular_threshold) <= 0:
        raise ValueError('Positive finite velocity thresholds required')
    dt = np.diff(t)
    linear = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    angular = (rotation[:-1].inv() * rotation[1:]).magnitude() / dt
    active = (linear > linear_threshold) | (angular > angular_threshold)
    indices = np.flatnonzero(active)
    left, right = bool(active[0]), bool(active[-1])
    return dict(status='NO_DETECTED_MOTION' if not len(indices) else
                ('BOTH_CENSORED' if left and right else 'RIGHT_CENSORED' if right else
                 'LEFT_CENSORED' if left else 'OBSERVED_WINDOW'),
                linear_threshold_m_s=float(linear_threshold), angular_threshold_rad_s=float(angular_threshold),
                start_sec=float(t[indices[0]]) if len(indices) else None,
                end_sec=float(t[indices[-1]+1]) if len(indices) else None,
                active_envelope_sec=float(t[indices[-1]+1]-t[indices[0]]) if len(indices) else 0.,
                sum_active_intervals_sec=float(dt[active].sum()), active_interval_count=int(active.sum()),
                left_censored=left, right_censored=right,
                observed_below_threshold_tail_sec=float(t[-1]-t[indices[-1]+1]) if len(indices) else float(t[-1]-t[0]),
                terminal_linear_speed_m_s=float(linear[-1]), terminal_angular_speed_rad_s=float(angular[-1]),
                recorded_start_sec=float(t[0]), recorded_end_sec=float(t[-1]), recorded_span_sec=float(t[-1]-t[0]),
                sample_count=len(t), sample_interval_sec=dict(min=float(dt.min()), median=float(np.median(dt)), max=float(dt.max())))


def _model(path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return model, hashlib.sha256(buffer.tobytes()).hexdigest()


def _runtime(value):
    if isinstance(value, dict):
        return value, {'canonical_json_sha256': hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}
    if value is None:
        return None, None
    return json.loads(Path(value).read_text()), _binding(value)


def _source(source_input):
    import mujoco
    root = Path(source_input)
    observations = root/'raw/observations/robot_camera_observations.npz'
    if not observations.exists():
        return {'status': 'UNKNOWN', 'reason': 'SOURCE_OBSERVATIONS_UNAVAILABLE'}, None
    runtime_path = root/'derived/robot_runtime/robot_runtime_contract.json'
    scene = runtime_path.parent/'robot_base_scene.xml'
    runtime, runtime_binding = _runtime(runtime_path)
    model, signature = _model(scene)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    body = model.body(runtime['end_effector_body']).id
    addresses = [model.joint(name).qposadr[0] for name in runtime['arm_joint_names']]
    provenance = dict(observations=_binding(observations), runtime=runtime_binding,
                      scene=_binding(scene), compiled_model_sha256=signature)
    obs_spec = observations.with_name('observations_spec.json')
    if obs_spec.exists(): provenance['observations_spec'] = _binding(obs_spec)
    with np.load(observations, allow_pickle=False) as z:
        required = ('joint_position_observed', 'robot_read_start_ms', 'robot_read_end_ms')
        if not all(k in z for k in required):
            return {'status': 'UNKNOWN', 'reason': 'MEASURED_JOINTS_OR_HOST_TIMESTAMPS_MISSING', 'provenance': provenance}, None
        joints = z['joint_position_observed']
        read_start, read_end = z['robot_read_start_ms'].astype(float), z['robot_read_end_ms'].astype(float)
    if joints.shape != (len(read_start), len(addresses)) or read_end.shape != read_start.shape or not np.isfinite(joints).all() or np.any(read_end < read_start):
        raise ValueError('Invalid measured source observations')
    mid = (read_start + read_end)/2
    origin = float(mid[0]); origin_kind = 'first_host_robot_read_midpoint'
    actions = root/'raw/actions/actions.npz'
    if actions.exists():
        with np.load(actions, allow_pickle=False) as z:
            if 'timestamp_ms' in z and len(z['timestamp_ms']):
                origin = float(z['timestamp_ms'][0]); origin_kind = 'first_host_control_start'
                provenance['host_control_origin_file'] = _binding(actions)
    p, q = [], []
    for row in joints:
        data.qpos[addresses] = row
        mujoco.mj_forward(model, data)
        p.append(data.xpos[body].copy()); q.append(data.xquat[body].copy())
    record = dict(status='AVAILABLE', provider='actual_observed_joint_FK', end_effector_body=runtime['end_effector_body'],
                  time_basis='host robot read_start/read_end midpoint; no robot hardware clock mixing',
                  local_time_origin=origin_kind, local_time_origin_host_ms=origin,
                  read_latency_ms=dict(min=float((read_end-read_start).min()), max=float((read_end-read_start).max())),
                  provenance=provenance)
    return record, ((mid-origin)/1000, np.asarray(p), np.asarray(q))


def _generated(scene, dense_path, runtime_value):
    import mujoco
    model, signature = _model(scene)
    runtime, runtime_binding = _runtime(runtime_value)
    provenance = dict(scene=_binding(scene), compiled_model_sha256=signature, dense_rollout=_binding(dense_path))
    if runtime_binding: provenance['runtime'] = runtime_binding
    with np.load(dense_path, allow_pickle=False) as z:
        t = z['timestamp_sec']
        if all(k in z for k in ('ee_position_m', 'ee_quaternion_wxyz')):
            p, q = z['ee_position_m'], z['ee_quaternion_wxyz']; provider = 'executed_dense_EE_pose'
        elif runtime and 'integration_state' in z:
            data = mujoco.MjData(model); body = model.body(runtime['end_effector_body']).id
            p, q = [], []
            states = z['integration_state']; state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            if states.shape != (len(t), mujoco.mj_stateSize(model, state_spec)):
                raise ValueError('Generated integration_state does not match model')
            for state in states:
                mujoco.mj_setState(model, data, state, state_spec); mujoco.mj_forward(model, data)
                p.append(data.xpos[body].copy()); q.append(data.xquat[body].copy())
            p, q = np.asarray(p), np.asarray(q); provider = 'executed_integration_state_FK'
        else:
            return {'status':'UNKNOWN', 'reason':'EXECUTED_EE_POSES_OR_RUNTIME_INTEGRATION_STATE_UNAVAILABLE', 'provenance':provenance}, None
    return dict(status='AVAILABLE', provider=provider, end_effector_body=runtime.get('end_effector_body') if runtime else None,
                time_basis='actual dense simulation timestamp_sec; no target/proxy video timeline', provenance=provenance), (t,p,q)


def audit(source_input, generated_scene, dense_rollout, generated_runtime=None,
          planned_record=None, output=None, linear_threshold=.005, angular_threshold=.05):
    """Read-only soft diagnostic. ``output`` must be a new file if provided."""
    started = time.monotonic()
    try:
        source, source_data = _source(source_input)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        # Keep the physical rollout's diagnostics even when source FK cannot
        # be measured. UNKNOWN is not a duration-comparison success.
        source = dict(status='UNKNOWN', reason='SOURCE_MOTION_UNAVAILABLE',
                      source_input=str(Path(source_input).resolve()), detail=str(error))
        source_data = None
    generated, generated_data = _generated(generated_scene, dense_rollout, generated_runtime)
    measured, recorded, sensitivity = {}, {}, {}
    for name, data in [('source',source_data), ('generated',generated_data)]:
        if data is None: continue
        try:
            measured[name] = motion_window(*data, linear_threshold, angular_threshold)
            recorded[name] = {k:v for k,v in measured[name].items() if k.startswith('recorded_') or k in ('sample_count','sample_interval_sec')}
            sensitivity[name] = {str(scale):motion_window(*data, linear_threshold*scale, angular_threshold*scale) for scale in (.5,1.,2.)}
        except ValueError as error:
            # Failed physics may reset simulation time. Preserve its evidence;
            # this optional soft audit must not interrupt failure reporting.
            failure = dict(status='UNKNOWN', reason='INVALID_EXECUTED_OR_OBSERVED_POSE_SERIES',
                           detail=str(error), error_type=type(error).__name__)
            measured[name] = failure
            (source if name == 'source' else generated).update(failure)
            recorded[name] = dict(status='UNKNOWN', sample_count=len(data[0]), reason=failure['reason'])
    comparison = dict(status='UNKNOWN', preference_ratio=[.8,1.2], natural_stop_duration_proven=False,
                      reason='Measured source and generated motion windows are both required; missing observations or no detected motion cannot establish a ratio.')
    if len(measured)==2 and all(measured[n].get('active_envelope_sec', 0)>0 for n in measured):
        s, g = measured['source'], measured['generated']; ratio = g['active_envelope_sec']/s['active_envelope_sec']
        censored = any(w['left_censored'] or w['right_censored'] for w in (s,g))
        within = .8 <= ratio <= 1.2
        comparison.update(status='UNKNOWN' if censored else 'PASS' if within else 'WARN',
                          recorded_motion_envelope_ratio=ratio, within_soft_ratio_band=within,
                          ratio_scope='recorded observed robot-motion envelopes, including internal dwell',
                          censoring_present=censored,
                          reason=('Recording boundary is active; complete operation duration is unknown.' if censored else
                                  'Observed motion-envelope ratio is within the soft preference band.' if within else
                                  'Observed motion-envelope ratio is outside the soft preference band.'),
                          retime_hint=dict(duration_multiplier_to_match_source=1/ratio,
                                           duration_multiplier_for_preference_band=[.8/ratio,1.2/ratio],
                                           provisional=censored,
                                           instruction='Planning hint only: retime controls, rerun dynamics and export the actual resulting rollout. Never rescale timestamps or count an unexported hold tail.'))
    if planned_record is not None and not isinstance(planned_record, dict):
        planned = dict(declared=json.loads(Path(planned_record).read_text()), provenance=_binding(planned_record))
    else: planned = dict(declared=planned_record)
    report = dict(schema_version='roborecast-operation-duration-v1',status=comparison['status'],soft_preference_only=True,
                  planned=planned,recorded=recorded,measured=measured,threshold_sensitivity=sensitivity,
                  source=source,generated=generated,comparison=comparison,
                  definition='First to last observed EE motion interval; external idle excluded, internal dwell retained. Not video duration or an automatic semantic task-completion detector.',
                  limitations=['Actual observed joint FK uses host read midpoint, an acquisition-time approximation.',
                               'Source and generated end-effector bodies must represent comparable tool frames, especially across robot changes.',
                               'EE-only motion does not detect gripper-only opening/closing; neither task success nor natural stopping is inferred.',
                               'Censored records cannot establish complete natural operation duration. Unrecorded or unexported hold tails never count.'],
                  wall_seconds=time.monotonic()-started)
    if output is not None:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-input',required=True);parser.add_argument('--generated-scene',required=True)
    parser.add_argument('--dense-rollout',required=True);parser.add_argument('--generated-runtime')
    parser.add_argument('--planned-record');parser.add_argument('--output',required=True)
    parser.add_argument('--linear-threshold',type=float,default=.005);parser.add_argument('--angular-threshold',type=float,default=.05)
    args=parser.parse_args();report=audit(**vars(args))
    print(json.dumps({'status':report['status'],'comparison':report['comparison'],'output':args.output},indent=2))


if __name__=='__main__':main()
