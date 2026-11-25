---
name: scene-initialization
description: Build a target scene with geometry, appearance, support, collision and joint properties.
---

# Scene initialization

Group attached parts into a modeling unit. Model the support surface together with fixtures whose placement depends on a cutout or attachment. Keep movable objects independent. Assign up to three units to parallel modelers.

Use the bundled AgentSTAR Blender harness on the initial camera view. For a passive joint, add a few frames spanning its travel. Compare silhouettes and appearance, adjust geometry or pose and render again. Preserve native materials. Match interaction geometry and the overall layout. Spend further pose iterations where they change reach or contact.

Use MJCF primitives for simple convex pieces and `roborecast.collision` (CuACD) for complex concavities. Infer a material with visual evidence, then call `roborecast.materials` for density, mass and friction priors. Set joint origin, local axis, range and damping. Assemble named units with `roborecast.assembly`; hand the assembled scene to validity checking.

For hinged objects, assign each named part to the fixed base or a moving link. Generate mesh collision pieces per part and attach them in the same local frame as its visual geom. Sweep the joint to check clearance before rollout.

The [microwave model](../../../examples/microwave/modeling/scene.py) declares the hinge and moving parts. The [unit assembly](../../../examples/unit-assembly/assembly.json) shows the smaller interface for a reusable MJCF unit.

[Tool commands and input formats](../../../guides/skills-and-tools.md)
