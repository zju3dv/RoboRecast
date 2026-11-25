# Skills and tools

Skills describe decisions the agent makes. The Python tools and Blender harness produce measurements, scene assets, joint targets and simulation traces. Start with the [microwave quickstart](../examples/microwave/README.md) for installation and a working bundle. A coding agent can start from [AGENT_TASK.md](../AGENT_TASK.md).

| Paper stage | Skills | Tools and evidence |
|---|---|---|
| [Task intent analysis](../.agents/skills/task-intent-analysis/SKILL.md) | Identify the objects, contacts and state changes that define the task, and specify their order. | [Source keyframes and clips](../src/roborecast/observations.py), [SAM 3 masks](../src/roborecast/segment.py), [LingBot-Depth estimates](../src/roborecast/depth.py), [interaction requirements](../src/roborecast/requirements.py) |
| [Scene initialization](../.agents/skills/scene-initialization/SKILL.md) | Group objects for modeling, estimate their shapes and poses, and assign material and joint properties. | [AgentSTAR modeling in Blender](../third_party/agenticSTAR/harness), [CuACD collision meshes](../src/roborecast/collision.py), [material priors](../src/roborecast/materials.py), [MJCF assembly](../src/roborecast/assembly.py) |
| [Motion initialization and compilation](../.agents/skills/motion-initialization/SKILL.md) | Choose approach and grasp poses, waypoint modes, gripper commands and timing. | [Frame transforms, waypoint compiler](../src/roborecast/harness.py), [quintic interpolation, SLERP, sequential IK](../src/roborecast/motion.py) |
| [Validity checking](../.agents/skills/validity-checking/SKILL.md) | Check support, collision geometry, joint travel and robot reachability before rollout. | [Collision overlays, settling and penetration checks, joint sweeps](../src/roborecast/inspection.py), [forward kinematics and joint limits](../src/roborecast/harness.py) |
| [Diagnosis-driven optimization](../.agents/skills/diagnosis-driven-optimization/SKILL.md) | Find the first failed interaction, adjust the scene or waypoints, and verify the repair. | [Rollout contact and state logs](../src/roborecast/rollout.py), [rendered keyframes](../src/roborecast/keyframe.py), [scene and waypoint edits](../src/roborecast/harness.py), [prefix checkpoints](../src/roborecast/checkpoints.py) |

## Inputs and outputs

| File | Convention |
|---|---|
| [scene.xml](../examples/microwave/scene.xml) | Assembled MuJoCo scene. Bundle-relative asset files. A named `settled_initial` keyframe sets the initial configuration when present. |
| [runtime.json](../examples/microwave/runtime.json) | Target joint order, position and velocity actuator names, end-effector body, gripper driver and width calibration. Provide per-joint velocity limits for a new robot. |
| [program.json](../examples/microwave/program.json) | Waypoint positions in metres, quaternions **xyzw**, gripper widths in metres, segment durations in seconds. Reference frames are world, named body or named site. Local references are resolved in the initial configuration. |
| [requirements.json](../examples/microwave/requirements.json) | Ordered interaction IDs and measurable predicates. Each subsequent waypoint binds its incoming segment to one or more interaction IDs. MuJoCo hinge states use radians; slide states use metres. |
| [success.json](../examples/microwave/success.json) | Physical and terminal checks. Keep identical across candidates. |
| `report.json` | Numerical outcome. `failed_checks` groups the checks needing attention. `windows` and `interactions` locate each evaluated event. |
| `rollout.npz` | Recorded timeline when the rollout runs. `qpos` is measured state, `ctrl` is applied control, and contact traces count active geom pairs. |

### Trace an interaction from source to report

In the paper, *c* describes the target configuration and *G* is the scene graph. In the bundle’s `requirements.json`, the key `I` lists interactions and their precedence, `G` lists initial-state predicates, and `c` lists constraints evaluated over the full rollout.

In the demonstration the gripper takes the handle and pulls the door partway open. The [interaction contract](../examples/microwave/requirements.json) binds `initial_pull` to both finger pads, the handle and the hinge. The [waypoint program](../examples/microwave/program.json) tags the incoming segments for `close`, `pull_mid`, `partial_open` and `partial_hold` with that ID. Together they delimit **6.3–11.0 s**. In that window each pad maintains contact for **4.26 s** and the door reaches **21.6°** at the end. [Reference report](../examples/microwave/evidence/baseline.json)

## Inspect observations

Commands below run from the repository root. FFmpeg is used only for video extraction. Source-frame indices refer to the supplied video. For a resampled proxy, use the original capture timestamps to estimate operation duration.

```sh
B=examples/microwave
python -m roborecast.observations "$B/media/source-proxy.mp4" --frame 110 --out runs/observations/pull.png
python -m roborecast.observations "$B/media/source-proxy.mp4" --clip 1.6 2.4 --out runs/observations/pull.mp4
```

Install the official [SAM 3 implementation and checkpoint](https://github.com/facebookresearch/sam3) in its CUDA environment. Invoke the adapter there after installing RoboRecast or putting its `src` on `PYTHONPATH`. A single call with `--jobs jobs.json` can process multiple prompts per image with a shared embedding.

```sh
python -m roborecast.segment --image "$B/modeling/frames/000000.png" --prompt microwave --checkpoint "$SAM3_CHECKPOINT" --output runs/masks/microwave.png
```

For aligned RGB and metric depth, prepare `input_000000.npz` with arrays `rgb` (H×W×3 uint8), `depth` (H×W float32 in metres, zero for invalid) and `K` (3×3 float32 pixel intrinsics). Use the [LingBot-Depth code](https://github.com/Robbyant/lingbot-depth) at `f3a237e434ae987bc38281476d6cfb5df3e4d739` and the [v0.5 checkpoint](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5). The adapter writes a refined depth map beside each input and `receipt.json`. For RGB-only modeling, supply known cameras, masks and a scale cue.

```sh
python -m roborecast.depth --inputs runs/depth --code "$LINGBOT_ROOT" --checkpoint "$LINGBOT_CHECKPOINT"
```

## Build and position a scene unit

The [microwave Blender script](../examples/microwave/modeling/scene.py) defines geometry, native materials, a hinge and a pose in each selected frame. The object remains in a single camera convention while the door joint changes. The native renderer compares the model with calibrated observations. Set `BLENDER` to your Blender executable. The RGB-only commands below were exercised with Blender 5.1.1 and the CPU Cycles renderer.

```sh
M=examples/microwave/modeling
H=third_party/agenticSTAR/harness
bash "$H/render.sh" "$M/scene.py" runs/match --views match --tracking "$M/tracking" --frames 000000.png --engine CYCLES --device CPU --samples 2 --res 320
bash "$H/render.sh" "$M/scene.py" runs/pose --views sweep --tracking "$M/tracking" --frames 000000.png --sweep-mask "$M/mask_object/000000.png" --sweep-ranges 'dpx:-0.01,0.01' --sweep-topk 1 --sweep-de-max-evals 32 --sweep-visuals none --res 160 --samples 1 --engine CYCLES --device CPU
```

Inspect `runs/match/0001/match_000000.png` and the candidates in `runs/pose/0001/sweep.json`. Update the pose or shape in the Blender script, render again and check the contact region. See the upstream [pose convention](../third_party/agenticSTAR/conventions/POSE.md) and [sweep reference](../third_party/agenticSTAR/harness/views/sweeps/sweep.md).

### Generate collision pieces for one part

Represent simple surfaces with MJCF primitives. For complex shapes, install [CuACD](https://github.com/eliphatfs/cuacd) and `trimesh` in a CUDA environment. This command takes the named door-rim mesh at scene scale and writes convex OBJ pieces and a `report.json` listing their paths.

```sh
python -m roborecast.collision --mesh examples/microwave/assets/microwave__door_rim_visual.obj --output runs/rim-hulls --gpu 0
```

In a unit MJCF, declare each hull as a mesh asset and attach a collision geom for it under the body that owns the part. The prepared microwave shows the handoff from the [rim mesh](../examples/microwave/assets/microwave__door_rim_visual.obj) to a [rim hull](../examples/microwave/assets/microwave__door_rim_collision_0.obj) and its named geom under the hinge in the [scene XML](../examples/microwave/scene.xml).

### Assemble an existing MJCF unit

For a small standalone assembly, the [fixture manifest](../examples/unit-assembly/assembly.json) places a prepared [door unit](../examples/unit-assembly/door.xml) into a base scene. Its collision geometry is defined directly with boxes. The first command prints physical priors for the door material. The second reads the manifest and writes the assembled scene.

```sh
python -m roborecast.materials examples/unit-assembly/material.json
python -m roborecast.assembly examples/unit-assembly/assembly.json --out runs/fixture.xml
```

Use unique asset and entity names and matching compiler settings in the base and units, including angular units. Unit MJCF files can supply `compiler`, `asset` and `worldbody` sections. The manifest specifies unit orientation as a **wxyz** quaternion.

### Hinged objects

1. Use a closed view and several opening keyframes to model the fixed and moving parts in Blender. In the [microwave model](../examples/microwave/modeling/scene.py), the housing and bottom trim are fixed. The door rim, window, inner border, handle and its mounts belong to the hinge child. The joint declaration gives the pivot, local axis and travel limits.
2. Export each named part at the scene scale. Generate convex collision pieces from that part’s own visual mesh. In the [microwave MJCF](../examples/microwave/scene.xml), the door rim has eight convex pieces. All eight share the visual rim’s local placement under the hinge body. The visual geom is for rendering and the convex pieces handle contact.
3. Sweep the joint before running the robot. Inspect the first interfering pair and compare the collision geometry with the opening keyframes. The [microwave quickstart](../examples/microwave/README.md) starts from the assembled scene and shows how to diagnose a misplaced pivot and verify the repair.

## Compile, check and repair

Use the [quickstart commands](../examples/microwave/README.md) to run the remaining stages. To compile without simulating, call `load_candidate(bundle, edit)`, `compile_motion(model, runtime, points)` and `preflight(model, runtime, trajectory)` from `roborecast.harness`. The compiler samples at 50 Hz. The rollout advances at the timestep declared in the scene. Within each segment, orientation and width follow the same quintic time law as position. Dwell fixes the pose while allowing the jaws to move.

Run `python -m roborecast.inspection runs/fixture.xml --out runs/fixture-check` to settle an assembled scene before waypoints exist. Add `--body fixture` for overlays or `--joint hinge` for an articulation probe. The three orthographic views overlay collision and visual geometry. The sweep measures new interference relative to the initial state, in either direction of a bounded hinge or slide. Use the pair name and angle to distinguish an obstructed path from a missed grasp. Settling reports per-body displacement and the deepest penetration. Interpret displacement according to the unit’s mobility.

When a motion preflight fails, the harness writes `report.json` with `rollout_executed: false`. The failed motion checks are listed under `failed_checks.motion_preflight`. After simulation, `failed_checks` groups the affected initial conditions, interactions, rollout constraints and required physics checks. Read their measurements in `interactions.constraints.G`, `interactions.events`, `interactions.constraints.c` and `physics.checks`, respectively. `interactions.earliest_unmet` identifies the first unmet interaction and can be null even when a separate condition fails.

The microwave quickstart adjusts the hinge anchor in a separate scene bundle. Its grasp example applies a waypoint offset in memory. For a new task, edit the scene unit, parameters or the waypoint program, then compile again. Pass `--save-prefixes` to preserve numerical event boundaries. With `--resume PREVIOUS_RUN`, the executor checks scene, controller, controls and requirement identities and selects the longest compatible prefix. A geometry change usually requires a fresh rollout. Use separate output directories to retain evidence from each candidate.
