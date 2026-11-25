"""Build a native monocular AgentSTAR capture in the fixed target's simulation frame."""

import argparse
import hashlib
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import cv2
import numpy as np

REVISION = "a8eac9b1a4e8423bc148839e88a930f5f6aab1e5"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")



def prepare(episode, descriptor, output):
    episode, output = Path(episode).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as stage:
        work = Path(stage) / "model"
        result = _prepare(episode, descriptor, work, output)
        work.rename(output)
        return result


def _prepare(episode, descriptor, output, final_output):
    meta = json.loads((episode / "meta.json").read_text())
    alignment = np.asarray(meta["T_sim_source_world"])
    rgb = episode / "source/rgb.mp4"
    frames = descriptor["frames"]
    initial = int(descriptor.get("initial_frame", 0))
    policy = dict(
        mode="single_camera_initial", primary_camera="ego", initial_frame=initial
    )
    records, cameras, intrinsics = [], [], []
    with np.load(episode / "reference.npz") as reference:
        indices = [int(f["frame_index"]) for f in frames]
        if not indices or indices[0] != initial or indices != sorted(set(indices)):
            raise ValueError(
                "Expected selected initial frame followed by distinct ordered keyframes"
            )
        if descriptor["kind"] not in ("rigid", "articulated"):
            raise ValueError("Unsupported object kind")
        if min(indices) < 0 or max(indices) >= len(reference["timestamps"]):
            raise ValueError("Source frame outside episode")
        capture = output / "capture"
        output.mkdir(parents=True, exist_ok=False)
        for folder in ("frames", "mask_object", "tracking"):
            (capture / folder).mkdir(parents=True)
        video = cv2.VideoCapture(str(rgb))
        try:
            for view, (index, frame) in enumerate(zip(indices, frames)):
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = video.read()
                if not ok:
                    raise ValueError(f"Cannot read source frame {index}")
                h, w = image.shape[:2]
                mask = (
                    cv2.imread(str(frame["mask_path"]), cv2.IMREAD_GRAYSCALE)
                    if "mask_path" in frame
                    else None
                )
                if "polygon" in frame:
                    mask = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(mask, [np.asarray(frame["polygon"], np.int32)], 255)
                if mask is None or mask.shape != (h, w) or not np.any(mask):
                    raise ValueError(
                        "Expected a nonempty original-resolution object mask or polygon"
                    )
                name = f"{view:06d}.png"
                if not (
                    cv2.imwrite(str(capture / "frames" / name), image)
                    and cv2.imwrite(
                        str(capture / "mask_object" / name),
                        255 * (mask > 0).astype("uint8"),
                    )
                ):
                    raise OSError("Could not write capture images")
                cameras.append(alignment @ reference["T_world_camera"][index])
                k = reference["K"]
                intrinsics.append(k if k.ndim == 2 else k[index])
                records.append(
                    dict(
                        view=view,
                        frame_index=view,
                        frame_name=name,
                        source_camera="ego",
                        source_frame_index=index,
                        source_video_frame_index=index,
                        source_video=str(rgb),
                        time_s=float(reference["timestamps"][index]),
                        observation_resolution="source",
                    )
                )
        finally:
            video.release()
    result = dict(
        schema="roborecast-model-input-v1",
        observation_policy=policy,
        episode=str(episode),
        object=descriptor,
        capture=str(final_output / "capture"),
        source_views=records,
        depth_source={"provider": "none"},
        depth_initialization=[],
        geometry_references=[],
        agentstar_revision=REVISION,
        camera_convention="T_sim_world_from_source_world @ T_source_world_from_camera; OpenCV",
    )
    if descriptor["kind"] == "rigid" and len(records) != 1:
        raise ValueError("Model rigid units from their initial frame")
    if descriptor["kind"] == "articulated" and len(records) < 2:
        raise ValueError("Articulated captures require distinct operation keyframes")
    np.savez(
        capture / "tracking/cameras.npz",
        c2w=np.asarray(cameras),
        intrinsics=np.asarray(intrinsics),
        size=[w, h],
    )
    result["camera_sha256"] = hashlib.sha256(
        (capture / "tracking/cameras.npz").read_bytes()
    ).hexdigest()
    dump(capture / "tracking/keyframes.json", records)
    dump(
        capture / "capture.json",
        dict(
            format="capture/v1",
            dataset="egodex",
            source={"episode": str(episode)},
            downscale=1.0,
            undistorted=True,
            depth="none",
            image_size=[w, h],
            frames=[r["frame_name"] for r in records],
            ref_frame=records[0]["frame_name"],
        ),
    )
    dump(output / "model_input.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--object", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.episode, json.loads(args.object.read_text()), args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
