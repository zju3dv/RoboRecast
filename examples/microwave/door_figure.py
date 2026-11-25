"""Compare hinge anchors using the modeled appliance and the first door-rim contact."""
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw, ImageFont
from roborecast.harness import initial_state, load_scene


def frame_contact(model, log):
    trim = {i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("microwave__bottom_trim_collision_")}
    rim = {i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("microwave__door_rim_collision_")}
    data = initial_state(model)
    for i, qpos in enumerate(log["qpos"]):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for c in data.contact:
            a, b = map(int, c.geom)
            if {a,b} & trim and {a,b} & rim and c.dist <= 0:
                return dict(time_s=float(log["timestamp_sec"][i]), angle_rad=float(data.joint("microwave__door_hinge").qpos[0]),
                            pair=[model.geom(a).name,model.geom(b).name], point_world_m=c.pos.tolist())
    raise ValueError("No physical door-rim/trim contact in the recorded rollout")


INK = "#193538"
MUTED = "#486467"
TEAL = "#126e65"
RUST = "#a6451e"
PAPER = "#f7faf9"


def label(draw, xy, text, size=19, color=INK, bold=False, anchor="la"):
    for family in ("DejaVuSans", "Arial"):
        face = family + ("-Bold" if bold and family == "DejaVuSans" else " Bold" if bold else "")
        try:
            font = ImageFont.truetype(face + ".ttf", size)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default(size=size)
    draw.text(xy, text, fill=color, font=font, anchor=anchor)


def parts(model, data, prefix):
    body = data.body("microwave")
    for i in range(model.ngeom):
        name = model.geom(i).name or ""
        if not name.startswith(prefix):
            continue
        mesh = model.geom_dataid[i]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        world = vertices @ data.geom(i).xmat.reshape(3, 3).T + data.geom(i).xpos
        yield i, (world - body.xpos) @ body.xmat.reshape(3, 3)


def front(sheet, bad, good):
    draw = ImageDraw.Draw(sheet)
    # Orthographic front elevation. Paint the original material and triangulation of each visible part.
    def project(x, z):
        return round(576 + x * 830), round(174 - z * 830)

    closed = initial_state(bad)
    closed.joint("microwave__door_hinge").qpos[0] = 0
    mujoco.mj_forward(bad, closed)
    faces = []
    for i, verts in parts(bad, closed, "microwave__"):
        if bad.geom_group[i] != 2:
            continue
        mesh = bad.geom_dataid[i]
        start = bad.mesh_faceadr[mesh]
        indices = bad.mesh_face[start:start + bad.mesh_facenum[mesh]]
        rgb = np.sqrt(bad.mat_rgba[bad.geom_matid[i], :3]) * 255
        for tri in verts[indices]:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if abs(normal[1]) < 1e-10:
                continue
            shade = .75 + .25 * abs(normal[1]) / np.linalg.norm(normal)
            faces.append((tri[:, 1].mean(), [project(x, z) for x, _, z in tri],
                          tuple(np.clip(rgb * shade, 0, 255).astype(int))))
    for _, points, color in sorted(faces, key=lambda face: -face[0]):
        draw.polygon(points, fill=color)

    def anchor_x(model):
        rest = initial_state(model)
        origin = local(rest, rest.joint(HINGE).xanchor)
        return origin[0]

    edge, inset = anchor_x(good), anchor_x(bad)
    xe, _ = project(edge, 0)
    xi, _ = project(inset, 0)
    for x, color in ((xe, TEAL), (xi, RUST)):
        for y in range(54, 284, 11):
            draw.line((x, y, x, y+7), fill=color, width=3)
    draw.line((210, 111, 292, 111, xe, 90), fill=TEAL, width=2)
    label(draw, (40, 81), "Reference axis", 23, TEAL, True)
    draw.line((200, 217, 298, 217, xi, 186), fill=RUST, width=2)
    label(draw, (40, 187), "Offset axis", 23, RUST, True)
    label(draw, (40, 231), f"{1000*(inset-edge):.0f} mm toward the handle", 18, MUTED)
    trim = np.vstack([v for _, v in parts(bad, closed, "microwave__bottom_trim_visual")])
    tx, ty = project(.18, trim[:, 2].min())
    draw.line((tx, ty, 821, ty, 846, ty-27), fill=MUTED, width=2)
    draw.ellipse((tx-3, ty-3, tx+3, ty+3), fill=MUTED)
    label(draw, (858, ty-51), "Fixed bottom trim", 22, INK, True)


HINGE = "microwave__door_hinge"


def local(data, points):
    body = data.body("microwave")
    return (np.asarray(points) - body.xpos) @ body.xmat.reshape(3, 3)


def section(vertices, height):
    # All the inputs are convex collision pieces. The hull of crossing chords
    # gives their exact horizontal section, including beveled rim edges.
    below = vertices[vertices[:, 2] < height]
    above = vertices[vertices[:, 2] >= height]
    if not len(below) or not len(above):
        return []
    delta = above[None, :, :] - below[:, None, :]
    weight = (height - below[:, None, 2]) / delta[:, :, 2]
    xy = (below[:, None, :2] + weight[:, :, None] * delta[:, :, :2]).reshape(-1, 2)
    return xy[ConvexHull(xy).vertices]


def hinge_panel(model, contact, offset, center, height):
    data = initial_state(model)
    data.joint(HINGE).qpos[0] = contact["angle_rad"]
    mujoco.mj_forward(model, data)
    def project(points):
        return (np.asarray(points)[..., :2] - center[:2]) * [5950, -5950] + [196, 178]

    pic = Image.new("RGB", (531, 272), PAPER)
    draw = ImageDraw.Draw(pic)
    for prefix, color in (("bottom_trim", "#c2d0cd"), ("door_rim", "#718f8d")):
        for _, vertices in parts(model, data, "microwave__"+prefix+"_collision_"):
            path = section(vertices, height)
            if len(path):
                draw.polygon([tuple(p) for p in project(path)], fill=color)
    label(draw, (335, 40), "Fixed trim", 20, INK)
    label(draw, (287, 234), "Door rim", 18, "#ffffff", True)

    pivot = project(local(data, data.joint(HINGE).xanchor))
    x, y = pivot
    color = RUST if offset else TEAL
    draw.ellipse((x-7, y-7, x+7, y+7), fill="white", outline=color, width=3)
    draw.line((x-7, y+2, 140, 216, 30, 216), fill=color, width=2)
    label(draw, (30, 209), "Pivot", 18, color, anchor="ls")

    segment = np.zeros(6)
    distance = mujoco.mj_geomDistance(model, data, *[model.geom(n).id for n in contact["pair"]], 1, segment)
    if offset:
        x, y = project(local(data, contact["point_world_m"]))
        draw.ellipse((x-9, y-9, x+9, y+9), outline="white", width=6)
        draw.ellipse((x-8, y-8, x+8, y+8), outline=RUST, width=3)
        title = "First contact"
    else:
        ends = project(local(data, segment.reshape(2, 3)))
        for x, y in ends:
            draw.line((x-6, y, x+6, y), fill=TEAL, width=2)
        draw.line([tuple(p) for p in ends], fill=TEAL, width=2)
        x, y = ends.mean(axis=0)
        title = f"{distance*1000:.1f} mm clearance"
    draw.line((x-7, y, 146, 107, 21, 107), fill=color, width=2)
    label(draw, (21, 99), title, 20, color, True, "ls")
    return pic, distance


def diagram(blocked, repaired, rollout, out):
    bad = load_scene(Path(blocked)/"scene.xml", {})
    good = load_scene(Path(repaired)/"scene.xml", {})
    with np.load(rollout) as log:
        contact = frame_contact(bad, log)
    state = initial_state(good)
    center = local(state, state.joint(HINGE).xanchor)
    state.joint(HINGE).qpos[0] = contact["angle_rad"]
    mujoco.mj_forward(good, state)
    segment = np.zeros(6)
    mujoco.mj_geomDistance(good, state, *[good.geom(n).id for n in contact["pair"]], 1, segment)
    height = local(state, segment.reshape(2, 3)).mean(axis=0)[2]
    sheet = Image.new("RGB", (1152, 706), "white")
    draw = ImageDraw.Draw(sheet)
    label(draw, (32, 13), "Front view   /   Door closed", 21, INK, True)
    front(sheet, bad, good)
    draw.line((32, 323, 1120, 323), fill="#cbd8d5", width=2)
    angle = np.degrees(contact["angle_rad"])
    label(draw, (32, 340), f"Hinge close-up   /   Top section at the bottom trim   /   Door open {angle:.1f}°", 21, INK, True)
    for x, model, offset, title in ((32, bad, True, "Offset axis"), (589, good, False, "Restored axis")):
        label(draw, (x, 382), title, 25, RUST if offset else TEAL, True)
        panel, distance = hinge_panel(model, contact, offset, center, height)
        sheet.paste(panel, (x, 423))
        if not offset:
            contact["repaired_pair_clearance_m_at_same_angle"] = distance
    sheet.save(Path(out)/"hinge-clearance.png", optimize=True)
    return contact
