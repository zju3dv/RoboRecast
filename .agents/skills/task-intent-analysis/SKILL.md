---
name: task-intent-analysis
description: Infer object roles, contacts, state changes and precedence from a prepared demonstration.
---

# Task intent analysis

Start with the initial state, the first visible contact, the state change and the release. Inspect a short clip around an ambiguous event. Use the same view to compare appearance across frames.

Extract frames or a clip with `python -m roborecast.observations`. For unclear object boundaries, call `roborecast.segment` on selected frames. With aligned metric depth and intrinsics, use `roborecast.depth` to refine those observations. RGB-only inputs can still support segmentation and silhouette-based modeling. Record the scale cue when metric scale is uncertain.

Assign source roles to target entities. Write `requirements.json` with `I` for ordered role-bound interactions, `c` for rollout constraints and `G` for initial-state predicates. Use contact duration or state change over a named time window as evidence. Keep the contract fixed across candidate edits.

For the microwave, identify the handle, passive hinge and left/right finger pads. The demonstrated sequence includes a partial pull, release and second contact to open wider. See the [actual contract](../../../examples/microwave/requirements.json).

[Tool commands and input formats](../../../guides/skills-and-tools.md)
