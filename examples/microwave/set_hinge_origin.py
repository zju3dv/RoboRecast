"""Set the microwave hinge anchor along the door's local width, measured from its reference edge."""
import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def write_bundle(source, output, inset_mm):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use separate input and output bundles")
    tree = ET.parse(source / "scene.xml")
    joint = tree.find(".//joint[@name='microwave__door_hinge']")
    if joint is None:
        raise ValueError("The microwave hinge is missing")
    # The door body's origin is the reference pivot. Keep the mesh and its rest pose fixed.
    xyz = joint.get("pos", "0 0 0").split()
    xyz[0] = format(inset_mm / 1000, ".12g")
    joint.set("pos", " ".join(xyz))
    output.mkdir(parents=True, exist_ok=True)
    for name in ("program.json", "runtime.json", "requirements.json", "success.json"):
        shutil.copyfile(source / name, output / name)
    shutil.copytree(source / "assets", output / "assets", dirs_exist_ok=True)
    tree.write(output / "scene.xml", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--inset-mm", type=float, required=True, help="Positive toward the handle")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_bundle(args.input, args.out, args.inset_mm)
