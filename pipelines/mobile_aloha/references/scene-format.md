# Modeling unit and scene format

A modeling fragment is a small MuJoCo XML file in the reconstruction directory. The harness prefixes its names with `unit_id` and applies `pose.xyyaw` to each top-level body. The robot asset already provides a ground plane. Use one freejoint per movable rigid object, simple hinge joints for doors, and only necessary static tables/obstacles.

Example `workspace/object.xml`:

```xml
<mujoco model="object">
  <worldbody>
    <body name="object" pos="0 0 0">
      <freejoint name="free"/>
      <geom name="seat" type="box" pos="0 0 0.45" size="0.22 0.22 0.03" mass="2" rgba="0.7 0.6 0.2 1"/>
      <geom name="back" type="box" pos="-0.20 0 0.72" size="0.03 0.22 0.27" mass="1" rgba="0.7 0.6 0.2 1"/>
      <geom name="legs" type="box" pos="0 0 0.22" size="0.20 0.20 0.22" mass="1" rgba="0.35 0.35 0.35 1"/>
    </body>
  </worldbody>
</mujoco>
```

Register with `{"unit_id":"item","fragment_xml_path":"workspace/object.xml","pose":{"xyyaw":[1.2,0.3,0],"z":0}}`. The free body name becomes `item/object`; `item/seat` is its seat geom. Static support bodies omit the freejoint. Add hinge joints inside a fixed parent body for a door. Use direct contact geometry and an appropriate mass/friction; do not weld a task object to the gripper.

`merge_and_settle_scene` accepts `{"robot_xyyaw":[0,0,0],"settle_seconds":0.5}` and returns the scene path, settled state, and hash. Scene revisions invalidate trajectory revisions.
