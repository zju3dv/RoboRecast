"""Construct an open tapered vessel with a smooth visible shell and convex wall segments."""

import math
import xml.etree.ElementTree as ET

import numpy as np


def build_bowl(
    path, name, position, radii, height, thickness, mass, rgba, friction, movable=True
):
    bottom, rim = map(float, radii)
    if (
        not 0 < thickness < min(bottom, rim, height) / 2
        or not 0 < bottom <= rim
        or mass <= 0
    ):
        raise ValueError(
            "Expected inner clearance and positive metric radius, height and mass"
        )
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    fmt = lambda values: " ".join(f"{v:.8g}" for v in values)
    body = ET.SubElement(world, "body", name=name, pos=fmt(position))
    if movable:
        ET.SubElement(body, "freejoint", name=name + "_free")
    segments = 40
    common = dict(friction=fmt(friction), rgba=fmt(rgba), condim="4")
    ET.SubElement(
        body,
        "geom",
        name=name + "_bottom",
        type="cylinder",
        size=fmt([bottom, thickness / 2]),
        pos=fmt([0, 0, thickness / 2]),
        mass=str(mass * 0.32),
        **common,
    )
    slope = math.atan2(rim - bottom, height)
    radial = (bottom + rim) / 2 - thickness / 2
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        radial_axis = np.array([math.cos(theta), math.sin(theta), 0.0])
        tangent = np.array([-math.sin(theta), math.cos(theta), 0.0])
        # The local vertical grows outward toward the lip.
        inward = -radial_axis * math.cos(slope) + np.array([0.0, 0.0, math.sin(slope)])
        pos = radial_axis * radial + np.array([0, 0, height / 2])
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_wall_{j}",
            type="box",
            pos=fmt(pos),
            xyaxes=fmt([*tangent, *inward]),
            size=fmt(
                [
                    rim * math.tan(math.pi / segments),
                    thickness / 2,
                    math.hypot(height, rim - bottom) / 2,
                ]
            ),
            mass=str(0.68 * mass / segments),
            **common,
        )
    ET.ElementTree(root).write(path, encoding="unicode")
