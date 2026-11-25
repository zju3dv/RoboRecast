---
name: diagnosis-driven-optimization
description: Diagnose the earliest unmet requirement and verify a targeted repair in simulation.
---

# Diagnosis-driven optimization

Run `python -m roborecast.harness` on the candidate and read `failed_checks` in `report.json`. A failed motion preflight returns before simulation. Inspect its pose error, joint limits or target velocity and correct the implicated waypoint, timing or robot specification. For an executed rollout, read the listed initial conditions, rollout constraints and required physical checks. If an interaction failed, use `interactions.earliest_unmet` and its window to locate the contact or state change. Retrieve a sparse frame there with `roborecast.keyframe`.

- **Pads miss the handle.** If the path is clear and both contact traces stay at zero during the pull, inspect a wrist frame and adjust the grasp approach or pose. [Missed-grasp report](../../../examples/microwave/evidence/grasp.json)
- **Pads engage and the door stalls.** Identify the first interfering pair and its door angle. Compare the collision overlay with the visual parts, check which parts move with the door, and compare the pivot with the source opening frames. These checks distinguish an oversized collider, a part attached to the wrong body and a misplaced axis. In the microwave example the moving rim hits the fixed bottom trim, and restoring the hinge origin recovers clearance. [Hinge comparison](../../../examples/microwave/media/hinge-clearance.png) · [Joint sweep](../../../examples/microwave/evidence/travel.json)

Apply the edit supported by the evidence, keep the requirements and thresholds fixed, repeat the affected checks and verify the later interactions.

Use `--save-prefixes` and `--resume` on successive candidate directories. Reuse a numerical prefix only when model, controller, trajectory prefix and requirements remain compatible. An earlier visual judgment can be carried forward by the agent for the same verified evidence; the CLI stores numerical evidence. Inspect the changed suffix. Continue until all required interactions and physical checks pass, or report the unresolved violation.

Compare the [two case reports](../../../examples/microwave/evidence/) with the repaired rollout and look for the first event that actually changed.

[Tool commands and input formats](../../../guides/skills-and-tools.md)
