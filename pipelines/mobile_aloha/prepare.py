"""Prepare one Mobile ALOHA HDF5 recording before launching the episode owner."""

import argparse
from pathlib import Path

from mobile_input.readers.mobile_aloha import read_mobile_aloha
from mobile_input.writer import write_episode
from mobile_robot.bundles.mobile_aloha import install
from mobile_robot.contracts import validate_runtime_contract


def install_runtime(embodiment_id, root, initial_arm_position):
    if embodiment_id != "mobile_aloha_tracer_vx300s":
        raise ValueError(f"Unsupported robot: {embodiment_id}")
    install(root, initial_arm_position)
    runtime = root / "derived/robot_runtime"
    validate_runtime_contract(runtime / "robot_base_scene.xml", runtime / "robot_runtime_contract.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-url", default="", help="Download URL for provenance")
    args = parser.parse_args()
    episode = read_mobile_aloha(args.source, args.source.stem, args.source.parent.name, args.source_url)
    print(write_episode(episode, args.output, install_runtime))


if __name__ == "__main__":
    main()
