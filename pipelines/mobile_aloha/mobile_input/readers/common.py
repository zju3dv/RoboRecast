from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import h5py
import numpy as np


class HDF5FrameReader:
    """Lazy persistent HDF5 frame decoder used only during conversion."""

    def __init__(self, path: Path, key: str, decoder: Callable[[np.ndarray], np.ndarray]):
        self.path, self.key, self.decoder = Path(path), key, decoder
        self._file: h5py.File | None = None

    def __call__(self, row: int) -> np.ndarray:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self.decoder(np.asarray(self._file[self.key][row]))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()


def decode_jpeg_bgr_payload_as_source_rgb(row: np.ndarray) -> np.ndarray:
    encoded = np.asarray(row, np.uint8).tobytes().rstrip(b"\x00")
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decode failed")
    # Mobile ALOHA passed an RGB camera array directly to OpenCV's BGR JPEG
    # encoder. imdecode therefore returns the original RGB byte order already.
    return decoded


def dataset_inventory(source: h5py.File) -> dict[str, dict]:
    result: dict[str, dict] = {}

    def visit(name: str, item: object) -> None:
        if isinstance(item, h5py.Dataset):
            result[name] = {"shape": list(item.shape), "dtype": str(item.dtype),
                            "compression": item.compression}
    source.visititems(visit)
    return result
