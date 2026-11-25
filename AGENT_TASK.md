# Work from an interaction requirement

Read the [five skills](guides/skills-and-tools.md). Start at the stage implied by the supplied inputs. A prepared simulation bundle already contains a scene, a robot contract, interaction requirements and semantic waypoints.

For a demonstration, identify contacts and state changes, map roles to the target, model relevant units, compile the motion, check feasibility and execute a rollout. Read the failed checks and their measurements. If an interaction failed, start with the earliest one. Use source and simulated keyframes to check appearance, layout and the meaning of the action. Re-run the affected checks. Preserve the source requirements while changing a candidate.

For the microwave exercise, run the supplied scene bundle or waypoint edit. Check the motion preflight first. For an executed rollout, inspect a frame inside the failed interval and compare it with the contact and state traces. Explain which variable prevented the interaction, apply one targeted repair and repeat the rollout. Return both reports, the changed parameter, and a frame showing the outcome. Use gripper and arm controls to move the passive door.

[Tool reference](guides/skills-and-tools.md) · [Microwave quickstart](examples/microwave/README.md)
