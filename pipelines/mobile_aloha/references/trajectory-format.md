# Mobile v1.4 semantic action format

Write `workspace/trajectory.json`. Cover every source row with contiguous stages. The Agent specifies sparse end-effector waypoints; the compiler owns interpolation, IK, joint targets, and motion retiming. Never write arm joint angles.

For simultaneous objects, add `"hand_objects":{"left":{"object":"object_a/body","mode":"hold"},"right":{"object":"object_b/body","mode":"grasp"}}`. Modes: `grasp` establishes grip, `hold` preserves it, `contact` pushes, `release` lets go. Declare both hands in each affected stage; `object` remains the primary target. Use actual scene body names; both hands may name the same body. Existing single-object fields remain valid. Arm waypoints remain independently editable.

```json
{
  "schema_version": "mobile-semantic-action-v1.4",
  "phases": [
    {
      "name": "approach",
      "action": "move_base",
      "start_row": 0,
      "end_row": 350,
      "base": {"mode": "source"},
      "left": {"mode": "hold_current"},
      "right": {"mode": "hold_current"}
    },
    {
      "name": "contact",
      "action": "operate_arm",
      "intent": "contact_push",
      "start_row": 350,
      "end_row": 450,
      "base": {"mode": "stop"},
      "object": "item/body",
      "contact_sides": ["left", "right"],
      "left": {"mode": "waypoints", "waypoints": [
        {"event": "contact", "at": 1.0,
         "target": {"type": "body_relative", "body": "item/body",
                    "offset_m": [-0.18, -0.02, 0.78]},
         "gripper_opening": 0.2}
      ]},
      "right": {"mode": "waypoints", "waypoints": [
        {"event": "contact", "at": 1.0,
         "target": {"type": "body_relative", "body": "item/body",
                    "offset_m": [0.18, -0.02, 0.78]},
         "gripper_opening": 0.2}
      ]}
    },
    {
      "name": "push",
      "action": "push_with_base",
      "intent": "contact_push",
      "start_row": 450,
      "end_row": 610,
      "object": "item/body",
      "contact_sides": ["left", "right"],
      "contact_geom_prefix": "item/contact_surface",
      "base": {"mode": "source"},
      "left": {"mode": "hold_current"},
      "right": {"mode": "hold_current"}
    }
  ]
}
```

## Arm targets

Arm modes are only `hold_current` and `waypoints`. Each waypoint may set `at` in `(0,1]`; omitted values are evenly spaced. `gripper_opening` is normalized `[0,1]`. The compiler inserts the current end-effector pose at stage start.

Target types:

- `world_pose`: `position_m`, optional `quaternion_wxyz`.
- `body_relative`: `body`, `offset_m`, optional body-relative `quaternion_wxyz`.
- `contact_frame`: `body`, `point_m` or `offset_m`, optional `normal_m`, `approach_offset_m`, and body-relative quaternion.
- `source_pose`: optional `row` and `sample` (`action` by default). The compiler converts that source arm sample to an end-effector pose internally; joint values are never exposed as execution targets. Use it for free-space alignment or retreat when the demonstration is already suitable, and use object/contact coordinates where reconstructed geometry differs.

Omit quaternion when position is sufficient. This avoids unnecessary orientation constraints. Use source rows for timing; the compiler may lengthen a stationary arm phase to respect motion limits. It never shortens source timing.

## Optional geometry expansion

Explicit arm waypoints remain authoritative. For routine rigid-object motion, a phase may instead include `motion.type=grasp` or `motion.type=carry`. The Agent still chooses the arm, grasp point, approach direction, orientation, lift clearance, route hints, and destination. The compiler only resolves named geometry and fills routine intermediate points.

```json
{"motion":{"type":"grasp","grasp_geom":"payload/handle",
 "pregrasp_distance_m":0.08,
 "arms":{"left":{"contact_point_m":[0,0,0],"approach_axis":[1,0,0],
                  "quaternion_wxyz":[1,0,0,0]}}}}
```

For a bilateral object with separate handles, put `grasp_geom` inside each arm entry. The arm-level value overrides the shared value so both contacts come directly from their named scene geometry.

```json
{"motion":{"type":"grasp","arms":{
 "left":{"grasp_geom":"payload/left_handle","contact_point_m":[0,0,0]},
 "right":{"grasp_geom":"payload/right_handle","contact_point_m":[0,0,0]}}}}
```

```json
{"motion":{"type":"carry","object":"payload/body",
 "arms":{"left":{},"right":{}},
 "route":{"lift_clearance_m":0.18,
          "via":[{"type":"world_pose","position_m":[1.1,0,1.2]}]},
 "destination":{"support_geom":"destination/support","offset_xy_m":[0,0],
                "clearance_m":0.005}}}
```

For `grasp`, contact points and approach axes are in the named grasp geom frame. For `carry`, omitted arm offsets are captured from the phase-start object/hand relation; the object is lifted vertically, follows optional `via` object-center targets, moves above the support, and lowers onto it. Orientation defaults to `preserve`. Use `orientation_mode: "preserve_lift"` to hold the grasp posture through vertical lift and let insertion use a locally continuous reachable wrist posture. Use `free` only when position continuity alone is sufficient. Physical contact and slip checks remain authoritative. Add normal `left` or `right` waypoint mode to override expansion for that arm. Inspect `semantic_expansions` in compile diagnostics when an expanded point is unreachable.

Do not copy a visible failed attempt merely because it appears in the demonstration. A same-object release/recontact sequence without `necessity` produces a `redundant_release_recontact` advisory.

## Stage actions

- `move_base`: `base` is `source`, `constant`, or `stop`; arms normally hold.
- `operate_arm`: base must be `stop`.
- `push_with_base` / `pull_with_base`: declare `object` and `contact_sides` or `grasp_sides`; arms normally hold during base motion.
- `carry_with_base`: declare `object` and `grasp_sides`.

Intent is one of `contact_push`, `contact_pull`, `pinch_grasp`, `supported_carry`, or `release`. Keep existing generic evidence fields when needed: `result_joint`, `min_joint_motion_rad`, `min_hand_contact_substeps`, `min_grasp_substeps`, `max_slip_m`, `goal_direction_xy`, `min_goal_progress_m`, `max_tilt_deg`, `carry_height_m`, and `max_object_end_speed_mps`.

Use `preserve_joint_ranges` on pot approach, carry, placement, and withdrawal phases when an already-open door must remain open. Each entry names a joint and a coarse `min_rad`/`max_rad`. Rollout also rejects mechanical-arm contact with non-target scene geometry; the first blocker returns the dominant collision pair.

Base commands remain measured `(v,omega)` priors. Optional `scale`, `bias`, `stop_xy_m`, and `stop_yaw_rad` are supported. Never prescribe `(x,y,yaw)` per frame.

Call `compile_mobile_trajectory` with the trajectory path. On compiler failure, adjust the named waypoint or base stop; do not derive joint angles. After an edit, unchanged stage-prefix checkpoints remain reusable.

After a grasp probe passes, `compile_mobile_trajectory` may receive `calibrate_after_phase` naming that grasp phase. The unchanged prefix checkpoint supplies the observed object, hand, and base state for geometry expansion of the next phase. When more than one earlier physical result affects later geometry, pass ordered `calibrate_after_phases`, for example `["open_doors", "grasp_pot"]`; each boundary resolves its own latest passing, prefix-matching checkpoint. The resulting complete trajectory remains frozen; the authoritative rollout still starts from the initial state and performs no online IK.

While repairing an early blocker, pass `compile_through_phase` with the phase that will be probed. The compiler holds the unvalidated suffix instead of solving unrelated later IK. This artifact is probe-only: the harness rejects later probes and complete rollout until a final compile omits `compile_through_phase`.

## Validation workflow

Use at most one relevant geometry probe when uncertainty is concrete. Then compile and call `probe_mobile_stage` through the first risky stage. Read `first_blocker` and its local sheet. Run a complete `simulate_mobile_rollout` only after the local stage passes. A probe cannot authorize review or finalization.

Write `workspace/success.json` with route, stop, contact, task motion, rest, and terminal outcome checks. Supported checks are `base_path_length`, `base_stop`, `contact`, `body_displacement`, `joint_displacement`, `terminal_base_rest`, `joint_final_range`, `terminal_body_relation`, `terminal_support_contact`, and `terminal_contact_absent`. Prefer named geoms for visible edge gaps and use coarse visually justified thresholds.

Changing mass, inertia, friction, damping, or contact parameters may call `merge_and_settle_scene` with `reuse_settled_state_revision`. Geometry, pose, joint, or topology changes must perform a normal settle.
