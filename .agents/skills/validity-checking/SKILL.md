---
name: validity-checking
description: Interpret geometry, settling, joint travel and robot motion feasibility.
---

# Validity checking

Compare the initial scene with the observation. Inspect support and collision openings in three orthographic views generated from the same model. Run a short settling simulation and inspect the drifting body and the deepest penetration.

Call `python -m roborecast.inspection` with the joint to sweep and body to visualize. The travel probe moves a passive joint through its range in a diagnostic state and reports newly interfering geometry pairs. A probe is separate from physical task execution. Run the FK and joint-limit checks in `preflight` for the candidate waypoints.

Trace each anomaly to its unit, geom, joint or waypoint. After a scene edit repeat the affected checks. A reachable, collision-free end-effector target still needs a rollout to establish the required contact.

In the [microwave hinge example](../../../examples/microwave/README.md), probe the lower door rim against the fixed bottom trim. Compare the two axis positions at the same opening angle, then repeat the travel probe and rollout with the corrected pivot.

[Tool commands and input formats](../../../guides/skills-and-tools.md)
