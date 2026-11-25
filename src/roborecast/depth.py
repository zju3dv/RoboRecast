import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

MODEL = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"

CHECKPOINT = "b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23"

REVISION = "f3a237e434ae987bc38281476d6cfb5df3e4d739"

SETTINGS = dict(resolution_level=9, apply_mask=True, use_fp16=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def worker(inputs, checkpoint, code):
    import torch
    from mdm.model.v2 import MDMModel

    git = ["git", "-C", str(code)]
    if subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() != REVISION:
        raise ValueError("Unexpected LingBot code revision")
    if subprocess.check_output(git+["status", "--porcelain", "--", "mdm"], text=True).strip():
        raise ValueError("LingBot model code has local changes")
    if digest(checkpoint) != CHECKPOINT:
        raise ValueError("Unexpected LingBot v0.5 checkpoint")
    import mdm.model.v2
    if not Path(mdm.model.v2.__file__).resolve().is_relative_to(code.resolve()):
        raise ValueError("Unexpected model import path")
    torch.manual_seed(0)
    start = time.monotonic()
    model = MDMModel.from_pretrained(str(checkpoint)).eval().cuda()
    torch.cuda.synchronize()
    record = dict(model_load_seconds=time.monotonic()-start, torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(), python=sys.executable, frames={})
    for path in sorted(inputs.glob("input_*.npz")):
        index = int(path.stem.split("_")[1])
        with np.load(path) as z:
            rgb, depth, K = z["rgb"], z["depth"], z["K"].copy()
        K[0] /= rgb.shape[1]
        K[1] /= rgb.shape[0]
        start = time.monotonic()
        with torch.inference_mode():
            output = model.infer(torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255,
                                 depth_in=torch.from_numpy(depth)[None].cuda(),
                                 intrinsics=torch.from_numpy(K)[None].cuda(), **SETTINGS)
        refined = output["depth"].squeeze().float().cpu().numpy()
        elapsed = time.monotonic()-start
        valid = np.isfinite(refined) & (refined > 0)
        common = valid & (depth > 0)
        if refined.shape != depth.shape or not common.any():
            raise ValueError("Official inference did not return a valid metric depth map")
        name = f"refined_{index:06d}.npy"
        np.save(inputs/name, refined)
        record["frames"][str(index)] = dict(file=name, sha256=digest(inputs/name),
            inference_seconds=elapsed, raw_valid_fraction=float((depth>0).mean()),
            refined_valid_fraction=float(valid.mean()),
            shared_median_abs_change_m=float(np.median(abs(refined[common]-depth[common]))))
    (inputs/"receipt.json").write_text(json.dumps(record, indent=2)+"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Refine aligned metric depth at selected RGB frames.")
    for field in ("inputs", "checkpoint", "code"):
        parser.add_argument("--" + field, required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    worker(args.inputs, args.checkpoint, args.code)
