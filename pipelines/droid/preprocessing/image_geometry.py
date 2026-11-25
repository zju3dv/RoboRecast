from __future__ import annotations

import copy
from typing import Any, Dict


def proxy_camera_geometry(source_intrinsics: Dict[str, Any], proxy_width: int = 640,
                          proxy_height: int = 360) -> Dict[str, Any]:
    """Describe the exact scale-to-fit/pad transform used by create_proxy."""
    source = copy.deepcopy(source_intrinsics)
    source_width, source_height = int(source["width"]), int(source["height"])
    if min(source_width, source_height, proxy_width, proxy_height) <= 0:
        raise ValueError("source and proxy dimensions must be positive")
    scale = min(float(proxy_width) / source_width, float(proxy_height) / source_height)
    scaled_width, scaled_height = source_width * scale, source_height * scale
    offset_x = (float(proxy_width) - scaled_width) / 2.0
    offset_y = (float(proxy_height) - scaled_height) / 2.0
    # Resize maps pixel centers, not pixel corners. Padding is then added.
    affine_x = offset_x + (scale - 1.0) / 2.0
    affine_y = offset_y + (scale - 1.0) / 2.0
    proxy = copy.deepcopy(source)
    proxy.update({
        "fx": float(source["fx"]) * scale,
        "fy": float(source["fy"]) * scale,
        "cx": float(source["cx"]) * scale + affine_x,
        "cy": float(source["cy"]) * scale + affine_y,
        "width": int(proxy_width), "height": int(proxy_height),
    })
    return {
        "source_intrinsics": source,
        "proxy_intrinsics": proxy,
        "source_to_proxy_pixel": {
            "scale_xy": [scale, scale],
            "offset_xy": [affine_x, affine_y],
            "padding_xy": [offset_x, offset_y],
            "mapping_formula": "proxy_pixel = (source_pixel + 0.5) * scale - 0.5 + padding",
            "geometry_version": "pixel_center_resize_v2",
            "pixel_center_convention": "opencv_zero_based_pixel_centers",
            "resize_policy": "scale_to_fit_then_symmetric_pad",
            "crop": False, "mirror": False,
        },
    }
