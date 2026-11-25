"""Reuse one SAM3 image embedding for the selected initial EgoDex objects."""

from pathlib import Path
import hashlib
import json
import re

import cv2

from transfer.capture import prepare
import os
import subprocess
import sys


def initial_masks(episode, output, objects):
    episode = Path(episode)
    output = Path(output)
    key = hashlib.sha256(json.dumps(objects, sort_keys=True).encode()).hexdigest()[:16]
    folder = output / key
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "initial.jpg"
    cap = cv2.VideoCapture(str(episode / "source/rgb.mp4"))
    try:
        ok, im = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError("Cannot decode prepared RGB")
    if not source.exists() and not cv2.imwrite(str(source), im):
        raise OSError("Unable to write initial observation")
    jobs = []
    ids = set()
    for obj in objects:
        name = obj["id"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) or name in ids:
            raise ValueError("Expected distinct object IDs")
        ids.add(name)
        jobs.append(
            dict(
                image=str(source),
                output=str(folder / (name + ".png")),
                prompt=obj["prompt"],
                object_id=name,
                box_xyxy=obj.get("box_xyxy"),
                guidance_evidence="owner identified the selected object in the original initial RGB",
            )
        )
    jobs_path = folder / "jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=2))
    result = subprocess.run([os.environ.get("SAM3_PYTHON", sys.executable), "-m",
                             "roborecast.segment", "--jobs", str(jobs_path),
                             "--batch-report", str(folder / "segmentation.json")],
                            capture_output=True, text=True)
    if result.returncode and not any(Path(job["output"]).is_file() for job in jobs):
        raise ValueError("SAM 3 produced no usable masks: " + (result.stderr or result.stdout)[-1500:])
    overlay = im.copy()
    captures = {}
    missing = []
    for job in jobs:
        name = job["object_id"]
        maskpath = Path(job["output"])
        if not maskpath.exists():
            missing.append(name)
            continue
        mask = cv2.imread(str(maskpath), cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (96, 235, 45), 3)
        x, y, _, _ = cv2.boundingRect(mask)
        cv2.putText(overlay, name, (x, max(24, y)), 0, 0.8, (70, 250, 50), 2)
        workspace = folder / ("model_" + name)
        if not (workspace / "model_input.json").exists():
            prepare(
                episode,
                dict(
                    id=name,
                    kind="rigid",
                    description=job["prompt"],
                    frames=[dict(frame_index=0, mask_path=str(maskpath))],
                ),
                workspace,
            )
        captures[name] = str(workspace)
    cv2.imwrite(
        str(folder / "mask_review.jpg"), cv2.resize(overlay, None, fx=0.5, fy=0.5)
    )
    return dict(
        captures=captures,
        missing=missing,
        comparison_sheet=str(folder / "mask_review.jpg"),
        note="Calibrated RGB-only captures. Masks are visual evidence; depth and metric dimensions are not inferred by SAM3.",
    )
