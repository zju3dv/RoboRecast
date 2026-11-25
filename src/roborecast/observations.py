"""Extract known frames and short evidence clips from a prepared video (FFmpeg)."""
import argparse
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame", type=int, help="Zero-based frame index")
    group.add_argument("--clip", nargs=2, type=float, metavar=("START_S", "END_S"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.video)]
    if args.frame is not None:
        if args.frame < 0:
            p.error("Frame index must be nonnegative")
        cmd += ["-vf", f"select=eq(n\\,{args.frame})", "-frames:v", "1", "-f", "image2", "-update", "1"]
    else:
        start, end = args.clip
        if not 0 <= start < end:
            p.error("Specify 0 <= start < end")
        cmd += ["-ss", str(start), "-t", str(end-start), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    subprocess.run(cmd + [str(args.out)], check=True)
    if not args.out.is_file() or not args.out.stat().st_size:
        raise ValueError("No frames in the requested interval")


if __name__ == "__main__":
    main()
