"""Fetch the DROID robot visuals at the exact revision used by the example."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

BASE = "https://raw.githubusercontent.com/droid-dataset/droid/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/robot_ik/franka/"
HERE = Path(__file__).resolve().parent


def main():
    for name, expected in json.loads((HERE/"arm-sha256.json").read_text()).items():
        dest = HERE/"assets"/name
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
            continue
        prefix = "texture/" if name.endswith(".png") else "mesh/"
        with urlopen(BASE+prefix+name, timeout=90) as response:
            blob = response.read()
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("Unexpected content for " + name)
        dest.write_bytes(blob)
    print("DROID arm visuals ready.")


if __name__ == "__main__":
    main()
