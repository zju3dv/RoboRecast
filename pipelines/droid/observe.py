"""Extract co-registered RGB, metric depth and intrinsics for selected source frames."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def extract(episode, camera, frames, out):
    geom = json.loads((episode / "specs/camera_spec.json").read_text())["image_geometry"][camera]
    k = geom["proxy_intrinsics"]
    intrinsic = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoCapture(str(episode / f"derived/videos/{camera}_proxy.mp4"))
    try:
        with h5py.File(episode / "derived/depth/depth.h5") as depth, np.load(episode / "derived/videos/proxy_frame_mapping.npz") as timeline:
            meters = depth[f"cameras/{camera}/depth_mm"]
            for i in frames:
                if not 0 <= i < len(meters):
                    raise IndexError(f"Frame {i} outside [0, {len(meters)})")
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, bgr = video.read()
                if not ok:
                    raise OSError(f"Cannot decode {camera} frame {i}")
                if bgr.shape[:2] != meters.shape[1:]:
                    raise ValueError("RGB and depth resolutions differ")
                if not cv2.imwrite(str(out / f"{i:06}.png"), bgr):
                    raise OSError(f"Cannot write RGB frame {i}")
                np.savez_compressed(out / f"input_{i:06}.npz", rgb=bgr[:, :, ::-1],
                                    depth=meters[i].astype(np.float32) * .001, K=intrinsic,
                                    control_time_sec=timeline["control_timestamp_sec"][i])
    finally:
        video.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("--camera", choices=["ext1", "ext2", "wrist"], default="ext1")
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    extract(a.episode, a.camera, a.frames, a.out)
