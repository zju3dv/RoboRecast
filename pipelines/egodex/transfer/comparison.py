"""Pair original human footage with a Blender replay at native operation speed."""

import argparse
import json
from pathlib import Path
import subprocess


def compare(source, simulation, output, source_offset, sim_offset, fps=24):
    def length(path):
        return float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nk=1:nw=1",
                    str(path),
                ],
                text=True,
            ).strip()
        )

    delays = [
        max(0.0, sim_offset - source_offset),
        max(0.0, source_offset - sim_offset),
    ]
    span = max(length(source) + delays[0], length(simulation) + delays[1])
    filters = []
    for i, title in enumerate(("EgoDex", "RoboRecast")):
        filters.append(
            f"[{i}:v]fps={fps},scale=768:432:flags=lanczos,setsar=1,"
            f"tpad=start_mode=clone:start_duration={delays[i]:.6f}:stop_mode=clone:stop_duration={span:.6f},"
            f"trim=duration={span:.6f},drawtext=text={title}:fontsize=22:fontcolor=white:"
            f"box=1:boxcolor=black@0.6:boxborderw=5:x=12:y=12[v{i}]"
        )
    filters.append("[v0][v1]hstack=inputs=2[v]")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-i",
            str(simulation),
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return dict(
        video=str(output),
        fps=fps,
        source_start_delay=delays[0],
        simulation_start_delay=delays[1],
        speed_multiplier=1.0,
        end_padding="hold last frame of shorter stream",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("simulation", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source-start", type=float, required=True)
    p.add_argument("--sim-start", type=float, default=0.0)
    args = p.parse_args()
    print(
        json.dumps(
            compare(
                args.source,
                args.simulation,
                args.output,
                args.source_start,
                args.sim_start,
            ),
            indent=2,
        )
    )
