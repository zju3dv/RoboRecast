"""Assemble named MJCF units with local assets and poses into a robot or world scene."""
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco


def assemble(manifest, output):
    spec = json.loads(manifest.read_text())
    base = manifest.parent
    def load(path):
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        for item in root.findall("asset/*"):
            if item.get("file"):
                directory = ""
                if compiler is not None:
                    key = "texturedir" if item.tag == "texture" else "meshdir"
                    directory = compiler.get(key, compiler.get("assetdir", ""))
                import os
                asset = (path.parent / directory / item.get("file")).resolve()
                item.set("file", os.path.relpath(asset, output.parent))
        if compiler is not None:
            for key in ("meshdir","texturedir","assetdir"):
                compiler.attrib.pop(key, None)
        return root

    world = load(base/spec["base"])
    def compiler_semantics(root):
        compiler = root.find("compiler")
        options = {} if compiler is None else dict(compiler.attrib)
        defaults = {"angle": "degree", "eulerseq": "xyz", "autolimits": "true"}
        return {**defaults, **options}

    asset = world.find("asset")
    if asset is None:
        asset = ET.SubElement(world, "asset")
    body = world.find("worldbody")
    for unit in spec["units"]:
        part = load(base/unit["file"])
        if compiler_semantics(part) != compiler_semantics(world):
            raise ValueError("Unit and base must share compiler options, including angular units")
        for section in part:
            if section.tag not in {"compiler", "asset", "worldbody"}:
                raise ValueError("Unit supports compiler, asset and worldbody sections: "+section.tag)
        asset.extend(copy.deepcopy(part.findall("asset/*")))
        holder = ET.SubElement(body, "body", name=unit["name"],
                               pos=" ".join(map(str,unit.get("position_m",[0,0,0]))),
                               quat=" ".join(map(str,unit.get("quaternion_wxyz",[1,0,0,0]))))
        holder.extend(copy.deepcopy(part.findall("worldbody/*")))
    output.parent.mkdir(parents=True, exist_ok=True)
    xml = ET.tostring(world, encoding="unicode")
    # Compile before publishing the output. Resolve assets relative to the output directory.
    import tempfile
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".xml", mode="w") as temp:
        temp.write(xml)
        temp.flush()
        mujoco.MjModel.from_xml_path(temp.name)
    output.write_text(xml+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.manifest, args.out)
