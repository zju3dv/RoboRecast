<h1 align="center">RoboRecast: Intent-Preserving Demonstration Augmentation via Diagnosis-Driven Optimization</h1>

<p align="center"><a href="https://autonomierlong.github.io/">Yushi Long</a><sup>1</sup>, <a href="https://juzhan.github.io/">Juzhan Xu</a><sup>1</sup>, <a href="https://brotherhuang.github.io/">Haibin Huang</a><sup>3</sup>, Chi Zhang<sup>3</sup>, Yin Chen<sup>3</sup>, <a href="https://pengsida.net/">Sida Peng</a><sup>1</sup>, <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a><sup>2,†</sup>, <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup></p>

<p align="center">¹ Zhejiang University · ² Shenzhen University · ³ China Telecom<br>† Corresponding author</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.32622"><img src="guides/assets/badges/paper.svg" alt="arXiv · 2611.32622"></a>
  <a href="https://zju3dv.github.io/RoboRecast/"><img src="guides/assets/badges/project.svg" alt="Project Page · RoboRecast"></a>
  <a href="https://zju3dv.github.io/RoboRecast/#video"><img src="guides/assets/badges/video.svg" alt="Demo Video · Watch"></a>
</p>

![Source human and robot demonstrations, diagnosis of failed interactions, and variations in layout, object, and robot.](docs/assets/teaser.webp)

Through diagnosis-driven optimization, RoboRecast augments real-world human and robot demonstrations for new layouts, objects, and robots while preserving task intent.

## Code Release Plan

- [x] General-purpose skills and tools for the RoboRecast pipeline
- [x] DROID example for Real2Sim reconstruction and demonstration augmentation
- [x] EgoDex example for Human2Robot transfer
- [x] Mobile ALOHA example for wheeled mobile manipulation

## Method Overview

RoboRecast identifies the contacts and state changes that define task intent in a real demonstration. The agent maps the roles in each interaction to entities in the target configuration, then uses these requirements to initialize the simulated scene and robot trajectory. If a validity check fails or a rollout misses a required interaction, the agent uses visual and numerical feedback to diagnose the cause, revise the relevant scene or motion parameters, and test the revision.

[Method overview](https://zju3dv.github.io/RoboRecast/#method) · [Demonstration gallery](https://zju3dv.github.io/RoboRecast/#demos)

## Skills & Tools

Open a stage to read its agent skill, or follow a tool link to inspect its implementation.

| Paper stage | Skills | Tools and evidence |
|---|---|---|
| [Task intent analysis](.agents/skills/task-intent-analysis/SKILL.md) | Identify the objects, contacts and state changes that define the task, and specify their order. | [Source keyframes and clips](src/roborecast/observations.py), [SAM 3 masks](src/roborecast/segment.py), [LingBot-Depth estimates](src/roborecast/depth.py), [interaction requirements](src/roborecast/requirements.py) |
| [Scene initialization](.agents/skills/scene-initialization/SKILL.md) | Group objects for modeling, estimate their shapes and poses, and assign material and joint properties. | [AgentSTAR modeling in Blender](third_party/agenticSTAR/harness), [CuACD collision meshes](src/roborecast/collision.py), [material priors](src/roborecast/materials.py), [MJCF assembly](src/roborecast/assembly.py) |
| [Motion initialization and compilation](.agents/skills/motion-initialization/SKILL.md) | Choose approach and grasp poses, waypoint modes, gripper commands and timing. | [Frame transforms, waypoint compiler](src/roborecast/harness.py), [quintic interpolation, SLERP, sequential IK](src/roborecast/motion.py) |
| [Validity checking](.agents/skills/validity-checking/SKILL.md) | Check support, collision geometry, joint travel and robot reachability before rollout. | [Collision overlays, settling and penetration checks, joint sweeps](src/roborecast/inspection.py), [forward kinematics and joint limits](src/roborecast/harness.py) |
| [Diagnosis-driven optimization](.agents/skills/diagnosis-driven-optimization/SKILL.md) | Find the first failed interaction, adjust the scene or waypoints, and verify the repair. | [Rollout contact and state logs](src/roborecast/rollout.py), [rendered keyframes](src/roborecast/keyframe.py), [scene and waypoint edits](src/roborecast/harness.py), [prefix checkpoints](src/roborecast/checkpoints.py) |

**Agent setup.** We tested RoboRecast with GPT-5.6-Sol, GPT-6.1-Sol, and GPT-6-Astra. To run the pipeline with your own Codex subscription, see [Configure your agent](guides/skills-and-tools.md#configure-your-agent).

**Run a prepared scene.** Follow the [microwave quickstart](examples/microwave/README.md) to reproduce the reference motion, diagnose two failed candidates and verify their repairs.

**Use your own demonstration.** Start with the [tool guide](guides/skills-and-tools.md) for observations, modeling and assembly. The harness reports the candidate’s numerical checks. The main agent compares keyframes with the source to judge visual alignment and whether the action preserves the demonstrated intent. [Instructions for a coding agent](AGENT_TASK.md)

## Quickstart · Open the microwave

A Robotiq gripper pulls the handle, releases it and engages it again to open the door wider. Compare the source demonstration with the simulated opening.

![Real demonstration and reference rollout](examples/microwave/media/opening.gif)

**A misplaced hinge axis.** The closed door looks the same with either pivot. Place the axis 15 mm toward the handle and the door rim swings into the fixed bottom trim at 30.7°. At that same angle, restoring the reference axis leaves 7.6 mm between the two collision pieces. The same pull then opens the door to 54.4°.

![Closed microwave with both hinge axes marked, above matched top sections showing rim contact and repaired clearance at 30.7 degrees](examples/microwave/media/hinge-clearance.png)

**A missed grasp.** Moving the grasp approach outwards leaves the waypoints reachable. Neither pad contacts the handle during the initial pull. Restoring the grasp position brings back contact and opens the door to 54.4°.

![Missed grasp, correct contact and repaired motion](examples/microwave/media/grasp.jpg)

[Run the two cases and inspect their reports →](examples/microwave/README.md)

## DROID · Real2Sim and Augmentation

From left to right, the DROID demonstration and its FR3 reconstruction show a tomato placed in a steel bowl. In the augmented demonstration, a UR5 places a banana in a teal bowl.

![DROID demonstration with an FR3, FR3 reconstruction, and UR5 banana augmentation, from left to right](docs/assets/additional/comparisons/place-an-object-in-a-bowl.gif)

## DROID · Articulated Examples

Each pair shows the DROID demonstration on the left and its reconstruction on the right.

<table>
  <tr>
    <td width="50%" align="center"><strong>Open the microwave</strong><br><img src="docs/assets/additional/comparisons/open-the-microwave.gif" alt="Open the microwave, real demonstration and reconstruction"></td>
    <td width="50%" align="center"><strong>Open the oven</strong><br><img src="docs/assets/additional/comparisons/open-the-oven.gif" alt="Open the oven, real demonstration and reconstruction"></td>
  </tr>
  <tr>
    <td width="50%" align="center"><strong>Place a screwdriver in the drawer and close it</strong><br><img src="docs/assets/additional/comparisons/place-a-screwdriver-in-the-drawer-and-close-it.gif" alt="Place a screwdriver in the drawer and close it, real demonstration and reconstruction"></td>
    <td width="50%" align="center"><strong>Rotate the faucet spout</strong><br><img src="docs/assets/additional/comparisons/rotate-the-faucet-spout.gif" alt="Rotate the faucet spout, real demonstration and reconstruction"></td>
  </tr>
</table>

To run Real2Sim and Augmentation on DROID data, follow the [DROID pipeline guide](pipelines/droid/README.md).

## EgoDex · Human2Robot Transfer

Each comparison shows the human demonstration on the left and the robot simulation on the right.

<table>
  <tr>
    <td width="50%" align="center"><strong>Stack Bowls</strong><br><img src="docs/assets/additional/comparisons/stack-bowls.gif" alt="Left and right grippers take turns stacking bowls"></td>
    <td width="50%" align="center"><strong>Hold a Cup and Tilt a Bottle</strong><br><img src="docs/assets/additional/comparisons/hold-a-cup-and-tilt-a-bottle.gif" alt="One hand holds a cup while the other tilts a bottle"></td>
  </tr>
  <tr>
    <td width="50%" align="center"><strong>Place a Toy in the Drawer</strong><br><img src="docs/assets/additional/comparisons/place-a-toy-in-the-drawer.gif" alt="A gripper opens a drawer, places a toy inside and closes it"></td>
    <td width="50%" align="center"><strong>Put a Green Block in a Box</strong><br><img src="docs/assets/additional/comparisons/put-a-green-block-in-a-box.gif" alt="A green block is grasped and released inside a box"></td>
  </tr>
</table>

To run Human2Robot transfer on EgoDex data, follow the [EgoDex pipeline guide](pipelines/egodex/README.md).

## Mobile ALOHA · Wheeled Mobile Manipulation

<p align="center">
  <strong>Push a Chair</strong><br>
  <img src="docs/assets/additional/comparisons/push-a-chair.gif" width="720" alt="Real demonstration on the left and the Mobile ALOHA reconstruction on the right: the robot approaches a chair, contacts the backrest and pushes it forward.">
</p>

To run wheeled mobile manipulation on Mobile ALOHA data, follow the [Mobile ALOHA pipeline guide](pipelines/mobile_aloha/README.md).

## Citation

```bibtex
@misc{long2026roborecast,
  title={RoboRecast: Intent-Preserving Demonstration Augmentation via Diagnosis-Driven Optimization},
  author={Yushi Long and Juzhan Xu and Haibin Huang and Chi Zhang and Yin Chen and Sida Peng and Ruizhen Hu and Xiaowei Zhou},
  year={2026},
  eprint={2611.32622},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2611.32622},
}
```
