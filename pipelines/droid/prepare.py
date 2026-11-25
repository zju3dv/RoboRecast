"""Prepare one local DROID recording before starting the reconstruction agent."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import time

from preprocessing.util import read_json, sha256_file, write_json


def measured_observations(h5_path, episode, serials):
    import h5py
    import numpy as np

    destination = episode / "raw/observations"
    destination.mkdir()
    fields = {
        "joint_position_observed": "observation/robot_state/joint_positions",
        "joint_velocity_observed": "observation/robot_state/joint_velocities",
        "robot_read_start_ms": "observation/timestamp/robot_state/read_start",
        "robot_read_end_ms": "observation/timestamp/robot_state/read_end",
    }
    required = set(fields)
    for name in ("robot_timestamp_seconds", "robot_timestamp_nanos"):
        fields[name] = "observation/timestamp/robot_state/" + name
    for role, serial in serials.items():
        for clock in ("estimated_capture", "frame_received", "read_start", "read_end"):
            fields[f"camera_{role}_{clock}_ms"] = f"observation/timestamp/cameras/{serial}_{clock}"
    with h5py.File(h5_path) as h5:
        arrays = {name: h5[path][:] for name, path in fields.items() if path in h5}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Missing measured robot fields: {sorted(missing)}")
    np.savez_compressed(destination / "robot_camera_observations.npz", **arrays)
    write_json(destination / "observations_spec.json", {
        "fields": {name: fields[name] for name in arrays},
        "missing_optional_fields": sorted(fields.keys() - arrays.keys()),
        "camera_serials": serials, "source_hdf5_sha256": sha256_file(h5_path),
        "clock_note": "Camera and host read timestamps are milliseconds. The hardware clock can have an offset.",
    })


def robot_identity(metadata, catalog):
    serial = str(metadata["robot_serial"]).lower()
    families = {"fr3": "FR3", "panda": "Panda"}
    prefix = serial.partition("-")[0]
    if prefix in families:
        return families[prefix], None
    matches = {}
    for candidate in sorted((catalog / "candidate_metadata").glob("*.json")):
        known = str(read_json(candidate).get("robot_serial", "")).lower()
        for prefix, family in families.items():
            if known == f"{prefix}-{serial}":
                matches[family] = {"path": str(candidate), "sha256": sha256_file(candidate),
                                   "matched_serial": known, "raw_serial": serial}
    if len(matches) != 1:
        raise ValueError(f"Could not determine the robot family for serial {serial!r}")
    return next(iter(matches.items()))


def prepare(raw, catalog, robot_assets, zed_settings, output):
    from preprocessing.config import EpisodeSelection
    from preprocessing.depth import extract_episode_depth
    from preprocessing.transform import convert_episode
    from preprocessing.validate import validate_episode

    inputs = [p.resolve() for p in (raw, catalog, robot_assets, zed_settings)]
    raw, catalog, robot_assets, zed_settings = inputs
    output = output.resolve()
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError("Choose an output directory separate from the inputs")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"status": "RUNNING", "raw": str(raw), "stages": []}

    @contextmanager
    def stage(name):
        tick = time.monotonic()
        try:
            yield
        finally:
            report["stages"].append({"name": name, "seconds": time.monotonic() - tick})

    try:
        with stage("source and robot assets"):
            files = list(raw.glob("metadata_*.json"))
            if len(files) != 1:
                raise ValueError("Expected one metadata_<uuid>.json in the recording")
            metadata = read_json(files[0])
            family, identity = robot_identity(metadata, catalog)
            serials = {role: str(metadata[f"{role}_cam_serial"]) for role in ("ext1", "ext2", "wrist")}
            for serial in serials.values():
                for path in (raw / f"recordings/MP4/{serial}.mp4", raw / f"recordings/SVO/{serial}.svo",
                             zed_settings / f"SN{serial}.conf"):
                    if not path.is_file():
                        raise FileNotFoundError(path)
            staging = output / "staging"
            staging.mkdir()
            (staging / raw.name).symlink_to(raw, target_is_directory=True)
            (staging / "_catalog").symlink_to(catalog, target_is_directory=True)
            shutil.copytree(robot_assets, output / "input/shared/robot_assets" / family)
        with stage("proxies, calibration and actions"):
            selection = EpisodeSelection(metadata["uuid"], raw.name)
            episode = convert_episode(selection, staging, output / "input", robot_family_override=family)
            if identity:
                write_json(episode / "specs/robot_identity_evidence.json", identity)
            measured_observations(raw / "trajectory.h5", episode, serials)
        with stage("three-view ZED depth"):
            extract_episode_depth(episode, staging, settings_dir=zed_settings)
        with stage("prepared input validation"):
            validation = validate_episode(episode, output / "input")
            if validation["status"] != "PASS":
                raise ValueError(f"Inspect {episode / 'validation_report.json'}")
        report.update(status="PASS", standardized_episode=str(episode), validation=validation["summary"])
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        report["total_wall_seconds"] = time.monotonic() - start
        write_json(output / "preprocessing_report.json", report)
    return episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("raw", "catalog", "robot-assets", "zed-settings", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    print(prepare(**vars(parser.parse_args())))
