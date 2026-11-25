"""Select a presentation view from actual task-body and finger visibility in the rollout."""

import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def select_camera(
    model, qpos, task_names, active_sides, target, span, source_direction
):
    task_ids = []
    for name in task_names:
        body = model.body(name).id
        task_ids.append(np.where(model.geom_bodyid == body)[0])
    # Keep distinctive visible surface details in view when choosing among clear grasps.
    details = [
        [
            int(i)
            for i in ids
            if model.geom_contype[i] == 0
            and model.geom_conaffinity[i] == 0
            and model.geom_rgba[i, 3] > 0.8
        ]
        for ids in task_ids
    ]
    details = [ids for ids in details if ids]
    fingers = []
    for side in active_sides:
        ids = [
            i
            for i in range(model.ngeom)
            if model.geom(i).name
            and model.geom(i).name.startswith(
                (side + "/left_finger_", side + "/right_finger_")
            )
        ]
        if ids:
            fingers.append(ids)
    camera = next(
        i
        for i in range(model.ncam)
        if model.cam_bodyid[i] == 0 and not np.any(model.cam_sensorsize[i])
    )
    data = mujoco.MjData(model)
    width, height = 320, 180
    result = None
    model.cam_fovy[camera] = 42
    azimuth = math.atan2(source_direction[1], source_direction[0])
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.enable_segmentation_rendering()
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False) + azimuth:
            for elevation in np.deg2rad([27, 42, 58]):
                zaxis = np.array(
                    [
                        np.cos(angle) * np.cos(elevation),
                        np.sin(angle) * np.cos(elevation),
                        np.sin(elevation),
                    ]
                )
                eye = target + span * 1.76 * zaxis
                xaxis = np.cross([0.0, 0.0, 1.0], zaxis)
                xaxis /= np.linalg.norm(xaxis)
                rotation = np.column_stack([xaxis, np.cross(zaxis, xaxis), zaxis])
                model.cam_pos[camera] = eye
                model.cam_quat[camera] = Rotation.from_matrix(rotation).as_quat()[
                    [3, 0, 1, 2]
                ]
                counts = []
                for pose in qpos:
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    ids = renderer.render()[:, :, 0]
                    hist = np.bincount(ids[ids >= 0], minlength=model.ngeom)
                    counts.append(
                        [int(hist[g].sum()) for g in task_ids + fingers + details]
                    )
                counts = np.asarray(counts)
                objects = counts[:, : len(task_ids)]
                # Give each small object equal standing, especially during acquisition.
                score = float(
                    np.log1p(objects).mean()
                    + 0.55 * np.log1p(objects.min(axis=0)).mean()
                )
                if fingers:
                    score += 0.26 * float(
                        np.log1p(
                            counts[:, len(task_ids) : len(task_ids) + len(fingers)].max(
                                axis=1
                            )
                        ).mean()
                    )
                if details:
                    score += 0.60 * float(np.log1p(counts[:, -len(details) :]).mean())
                score += 0.15 * math.cos(angle - azimuth)
                if result is None or score > result[0]:
                    result = (score, eye.copy(), rotation.copy(), counts.tolist())
    return result
