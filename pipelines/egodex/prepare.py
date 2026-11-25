#!/usr/bin/env python3
"""Prepare one official EgoDex pair offline, including a bounded source contact sheet."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fixed_robot.inputs import convert_episode


def prepare(rgb, poses, output):
    episode_id = f"{rgb.parent.name}/{rgb.stem}"
    convert_episode(rgb, poses, output)
    with np.load(output / "reference.npz") as data:
        indices = np.unique(np.linspace(0, len(data["timestamps"]) - 1, 12, dtype=int))
    video = cv2.VideoCapture(str(output / "source/rgb.mp4"))
    images = []
    try:
        for index in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = video.read()
            if not ok:
                raise ValueError(f"Cannot read source frame {index}")
            h, w = image.shape[:2]
            image = cv2.resize(
                image, (400, round(400 * h / w)), interpolation=cv2.INTER_AREA
            )
            cv2.putText(
                image,
                str(index),
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
            images.append(image)
    finally:
        video.release()
    while len(images) % 3:
        images.append(np.zeros_like(images[0]))
    cv2.imwrite(
        str(output / "source_keyframes.jpg"),
        np.vstack([np.hstack(images[i : i + 3]) for i in range(0, len(images), 3)]),
    )
    return dict(
        episode_id=episode_id,
        prepared_episode=str(output.resolve()),
        timing_scope="offline",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("rgb", "poses", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.rgb, a.poses, a.output), indent=2))
