from __future__ import annotations
import math
from typing import Any, Dict
from .errors import StructuredToolError
DENSITY_KG_M3 = {
    "plastic": (700.0, 1250.0, 950.0), "wood": (400.0, 900.0, 650.0),
    "metal": (2500.0, 8000.0, 4000.0), "ceramic": (1800.0, 2800.0, 2200.0),
    "glass": (2200.0, 2700.0, 2500.0), "rubber": (850.0, 1300.0, 1050.0),
    "fabric": (80.0, 450.0, 220.0), "produce": (650.0, 1150.0, 900.0),
    "unknown": (500.0, 2500.0, 1000.0),
}

CONTACT = {
    "plastic": (0.55, 0.01, 0.001), "wood": (0.65, 0.01, 0.001),
    "metal": (0.40, 0.008, 0.0005), "ceramic": (0.45, 0.006, 0.0005),
    "glass": (0.35, 0.004, 0.0003), "rubber": (1.0, 0.02, 0.002),
    "fabric": (0.8, 0.02, 0.003), "produce": (0.65, 0.015, 0.002),
    "unknown": (0.6, 0.01, 0.001),
}


def physical_profile_values(request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the shared material prior for both modeling backends."""
    if len(request["dimensions_m"]) != 3:
        raise ValueError("Expected three dimensions in meters")
    dimensions = [float(value) for value in request["dimensions_m"]]
    if any(not math.isfinite(value) or value <= 0 or value > 5 for value in dimensions):
        raise StructuredToolError("IMPLAUSIBLE_DIMENSIONS", "Dimensions must be finite task-scale lengths.",
                                  "$.dimensions_m", "Three values in (0, 5] meters.", dimensions,
                                  [0.1, 0.1, 0.1], True)
    material = request["material_class"]
    density_low, density_high, density = DENSITY_KG_M3[material]
    bounding_volume = dimensions[0] * dimensions[1] * dimensions[2]
    fill_fraction = {
        "support": 0.35, "container": 0.12, "handled_container": 0.16,
        "hand_tool": 0.45, "fixture": 0.30, "produce": 0.62,
        "cloth": 0.08, "granular_payload": 0.55,
    }[request["object_class"]]
    if request["mobility"] == "fixed":
        fill_fraction = max(fill_fraction, 0.5)
    mass = max(0.002, bounding_volume * fill_fraction * density)
    mass_low = max(0.001, bounding_volume * fill_fraction * density_low)
    mass_high = max(mass_low, bounding_volume * fill_fraction * density_high)
    friction = CONTACT[material]
    unit_id = request["unit_id"]
    profile_id = "physical:{}:v1".format(unit_id)
    return {
        "schema_version": "real2sim-physical-profile-v1", "physical_profile_id": profile_id,
        "unit_id": unit_id, "object_class": request["object_class"],
        "material_class": material, "mobility": request["mobility"],
        "dimensions_m": dimensions, "bounding_volume_m3": bounding_volume,
        "density_kg_m3": {"minimum": density_low, "selected": density, "maximum": density_high},
        "mass_kg": {"minimum": mass_low, "selected": mass, "maximum": mass_high},
        "contact": {"friction_slide_torsion_roll": list(friction),
                    "solref": [0.02, 1.0], "solimp": [0.9, 0.95, 0.001]},
        "damping_guidance": {"free_body_linear": 0.001,
                             "passive_joint": 0.15 if request["mobility"] == "articulated" else 0.0},
        "confidence": "low" if material == "unknown" else "medium",
        "assumptions": ["Mass uses bounding volume and a class-specific fill fraction.",
                        "Material is a semantic prior, not inferred from pixels."],
        "judgment": "Returned bounded physical priors; explicit plausible MJCF values remain authoritative.",
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Estimate physical parameters from material and shape priors.")
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    print(json.dumps(physical_profile_values(json.loads(args.request.read_text())), indent=2))
