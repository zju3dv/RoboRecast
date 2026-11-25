# Fixed-arm motion and physical checks

## Sparse action format

Default positions/quaternions are **sim-world** pinch-site goals; explicit relative target types below are resolved once before IK. `source_start/end` and event times use exported input-local `timestamps` seconds. Original recording `source_timestamps` and frame indices remain provenance, not command times. Waypoint `t` is seconds local to the unretimed phase and includes both `0` and `duration`. Grippers use 0 closed, 1 open. Both roles must be explicit, even when one arm is idle. Use the committed `initial_end_effectors` for the first waypoint and preserve preceding phase endpoints; discontinuous jumps are rejected.

```json
{
  "base_mode": "fixed",
  "sample_hz": 30,
  "joint_velocity_limit": 1.5,
  "joint_acceleration_limit": 5.0,
  "phases": [{
    "name": "approach",
    "source_start": 0.0,
    "source_end": 0.8,
    "duration": 2.0,
    "roles": {"left": "idle", "right": "approach visible grasp region"},
    "hand_objects": {"left": {"mode": "idle", "object": null}, "right": {"mode": "idle", "object": null}},
    "waypoints": [
      {"t": 0.0, "targets": {"right": {"position": [0.3, -0.2, 1.1]}}, "grippers": {"right": 1}},
      {"t": 2.0, "targets": {"right": {"position": [0.4, -0.2, 1.0]}}, "grippers": {"right": 1}}
    ],
    "events": [{"source_time": 0.8, "meaning": "pre-contact"}]
  }]
}
```

The example numbers only illustrate the schema, **not an executable trajectory or an approved task solution**. Choose actual goals from source and candidate FK, then add grasp/carry/release phases. Do not convert the entire human time series to robot waypoints. Compiler interpolates sparse goals and may lengthen phases to meet kinematic limits. `time_map` records explicit source/simulation intervals and event correspondence; static periods remain visible in this mapping. IK checks are not global path planning.

## Physical task checks

```json
{
  "events": [{
    "name": "right_grasp",
    "source_start": 1.0,
    "source_end": 1.1,
    "checks": [{"kind": "grasp", "side": "right", "body": "object"}]
  }],
  "terminal_checks": [
    {"kind": "support", "body": "object", "support": "container"},
    {"kind": "release", "side": "right", "body": "object"},
    {"kind": "speed", "body": "object", "max": 0.03}
  ],
  "settle_seconds": 1.0,
  "terminal_window_seconds": 0.5,
  "record_hz": 30,
  "max_penetration_m": 0.008,
  "minimum_contact_force": 0.01,
  "max_tracking_error_rad": 0.5
}
```

Replace body names, times and checks with source-backed task criteria. Each event interval must contain physics steps, and every physics step in it must meet each ordinary check (grasp_acquired uses the explicit dwell rule below); record_hz controls saved media states, not event checking frequency. At least one physical grasp event and nonempty terminal checks are required. `grasp` measures contact against both finger geoms of the selected hand; `release` requires neither finger to touch. `support` measures a contact relation, not full load-bearing proof. Other checks: `position {body,target,tolerance}`, `relative_position {body,relative_to,target,tolerance}`, `height {body,min,max}`, `speed {body,max}`. Add sustained carry/grasp and support/release evidence per object/phase; do not claim success from final position alone.

Initial research thresholds live with their responsible geometry, robot and execution routines and are copied to diagnostics when applied; they have not been calibrated on the two tasks. The episode owner compares source and simulation keyframes for hand roles, grasp regions and geometry, and records that assessment against the passing rollout.

For a drawer, use `joint_range {body,joint,min,max}` for its opened and closed positions in metres. Limit `grasp` to a named handle geom with `geom`. Pair `inside_tray {body,floor,inset_m}` with a force-qualified `support` check using `support_geom` equal to the drawer floor. The first checks the object's collision footprint against the rectangular floor in its local frame. Release and support are checked over the terminal window.

## Explicit target frames

Existing `{position:[x,y,z], quaternion_wxyz:[w,x,y,z]}` remains a world target. These alternatives use metres and normalized quaternions:

```json
{"type":"world_pose","position_m":[0.4,0.1,1.0]}
{"type":"body_relative","body":"bowl","offset_m":[0,0,0.05]}
{"type":"contact_frame","site":"bowl_contact","point_m":[0,0,0],"normal_m":[0,0,1],"approach_offset_m":0.04}
```

`body_relative` position and optional orientation are body-local. `contact_frame` requires exactly one `body`, `geom` or `site`. Optional `frame_position_m` and `frame_quaternion_wxyz` define a local contact frame within that entity. `point_m` and normalized `normal_m * approach_offset_m` are in that contact frame; `quaternion_wxyz` is the desired pinch orientation relative to it. Resolution uses the **committed initial state**, once before all IK. `resolved_spec.json` stores frozen world goals. It does not follow a moving object or silently estimate its future pose; moved-object goals require supported explicit world targets. Placement events remain world goals.

## Per-hand object binding

Every phase requires both `roles` descriptions and `hand_objects` for both hands:

```json
"hand_objects": {
  "left": {"mode":"hold","object":"bowl_a","max_contact_gap_sec":0.15,"max_slip_m":0.03},
  "right": {"mode":"idle","object":null}
}
```

Modes: `idle`, `grasp`, `hold`, `contact`, `release`. Non-idle objects are task-body names, not robot bodies. Use `grasp` while establishing a pinch, `hold` during transport/stabilization, `release` while opening. A hand may be idle during pre-contact approach. Source evidence determines actual object ownership, not a filename or task label.

## Advisory contact probe

`probe_contact` accepts `side`, `object`, `target_position` (sim-world metres), optional `quaternion_wxyz`, `gripper_opening` (default 0), `approach_seconds` (1), `close_seconds` (0.3), `duration` (hold, 0.5), `source_time` (0), `minimum_contact_force` (0.01). It starts in an isolated copy of the committed state, ramps the selected arm to the IK-derived control target, closes and holds, records actual two-finger contact and collision categories. The other arm retains its committed command. This is a diagnostic actuator ramp, not a certified Cartesian path. Results/evidence live under `probes/`; no placement, trajectory, rollout or review state is replaced.

## Checkpoint semantics and terminal windows

`simulate_task_rollout` and `probe_human_stage` accept optional `reuse_checkpoints` (default true); false provides a fresh physics execution for comparison. The cache stores `mjSTATE_INTEGRATION` at physics steps, including controller state and solver warm start. Reuse checks placement/input/file identity, MuJoCo version, execution code and the exact interpolated actuator-control prefix. Source-time mappings and current predicates are reevaluated from restored prefix states. A changed suffix resumes at the first changed command. A changed scene/placement or execution implementation invalidates reuse. Changing only success predicates does not force identical physics to be repeated and never reuses an old verdict.

This implementation replays cached states for evaluation/rendering; it saves integration work, not all processing or video encoding. Only the latest physics cache is searched. Both local probes and full executions can contribute compatible physics; a probe's advisory verdict is never promoted to authoritative PASS. The episode owner checks the current physical evidence and corresponding frames.

`terminal_window_seconds` defaults to 0.5 and must be positive and fit the full execution. Every terminal predicate must pass on **every physics step** in the final window, including final settling. If the window overlaps task motion those states are checked too; choose settle_seconds and the window intentionally.

## Focused failure evidence

Rollouts retain full diagnostic JSON, states and compact evidence bundles. Full delivery videos are generated after review PASS. Compact returned diagnostics point to the first failing phase/time, ownership metrics and collision counts. Failure sheets show nearest recorded states at failure minus 0.25 seconds, failure, and plus 0.25 seconds in source/both wrists/three external views. Saved sampling resolution may miss a brief exact substep; detailed physics/contact diagnostics retain the actual failure time. These sheets prioritize inspection but never replace whole-task review.


Binding monitor semantics (each physics step):

| Mode | Required evidence |
|---|---|
| idle | No object ownership assertion |
| grasp | Two-finger contact established by phase end |
| hold | Endpoint two-finger contact, max gap <= `max_contact_gap_sec` (0.15 s), max relative translation <= `max_slip_m` (0.08 m) |
| contact | Finger contact fraction >= `min_contact_fraction` (0.25) |
| release | Neither finger touching the bound object at phase end |

Hold rotation slip can be limited by `max_rotation_slip_deg` (default 180). Relative object translation/rotation are measured in the gripper-link frame; hold reference is the phase's first state, grasp reference starts when two-finger contact appears. For grasp/hold, optional `max_tilt_deg` limits object local +Z tilt from world +Z and `carry_height_m` requires the body's world Z to exceed that height throughout the phase. These are body-frame conventions, not inferred mass centers or universal definitions of upright. Metrics reset per phase; sustained transport must be declared as hold rather than repeated grasp phases. Contact mode measures finger contact by default. A named contact_geom binds contact to that specific surface and allows the gripper housing as well as its fingers. All tolerances are research parameters and must be justified for the task.



## Tie checks to the demonstration

Before proposing motion, give every required source event a stable `id`, its observed `start` and `end` in source seconds, and a list of `required_checks`. For example, after observing a toy released inside a drawer, record an event like this with the appropriate times and simulated body names.

```json
{
  "id": "toy_released_in_tray",
  "start": 2.67,
  "end": 2.83,
  "required_checks": [
    {"scope": "event", "kind": "release", "side": "left", "body": "toy"},
    {"scope": "terminal", "kind": "inside_tray", "body": "toy", "floor": "drawer_floor"},
    {"scope": "terminal", "kind": "support", "body": "toy", "support": "drawer", "support_geom": "drawer_floor"}
  ]
}
```

An `event` selector matches a rollout predicate in an overlapping source window. A `phase` selector matches the phase monitor, for example `{"scope":"phase","side":"right","mode":"contact","object":"receiver"}` for the assisting hand. A `terminal` selector matches a predicate evaluated throughout final settling. Include the slide joint and the accepted numeric range when distinguishing open and closed states. Fields in a selector must match the executed check. If the source reveals a more specific contacting component, keep its event ID and check, update the component name and supply binding_evidence on the revised event. Missing evidence is reported by source event ID before the simulation starts. A physical pass also requires the matched predicates and phase monitors to pass.

For the fixed robot, unintended arm/arm, robot/environment and object/robot-structure contacts fail when their summed normal force between the same two bodies reaches 1 N for 50 ms continuously. The overall penetration check still applies at every step. Finger contact with the intended object remains available for grasping or stabilizing. For an intentional push using the gripper housing, bind the target body's named surface with `contact_geom` in that hand's `contact` phase.
