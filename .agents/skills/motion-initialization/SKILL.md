---
name: motion-initialization
description: Place contact-aware waypoints and compile them to robot joint targets.
---

# Motion initialization and compilation

Inspect the gripper pads, grasp frame and handle in the assembled scene. Choose an approach direction that leaves clearance and a grasp width in metres. Bind interaction IDs to contiguous waypoint intervals. Preserve active demonstration timing including interior pauses.

Use `transit` to connect poses, `dwell` to hold the preceding resolved pose and `sustained_contact` to follow the task geometry. For a hinge, place support points along its arc. Declare `world`, `body` or `site` reference frames. The compiler resolves local poses against the initial scene and applies the shared quintic time law, SLERP and sequential IK.

Call `compile_motion` or run the prepared bundle via `python -m roborecast.harness`. Inspect FK position/orientation error, joint ranges and target velocities. Adjust approach, timing or unreachable targets before interpreting contact failures.

Read the [waypoint program](../../../examples/microwave/program.json). `contact_align` and `close` share a pose while the gripper width changes; the pull then follows the door.

[Tool commands and input formats](../../../guides/skills-and-tools.md)
