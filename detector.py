"""
detector.py — Moondream model wrapper.

Supports two backends:
  • Cloud  — md.vl(api_key=...)  — no GPU, but has network latency + rate limits.
  • Local  — md.photon()         — private, works offline, GPU strongly recommended.

The Detector class is intentionally thin so the rest of the app
doesn't need to know which backend is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import moondream as md
from PIL import Image


@dataclass
class BoundingBox:
    """
    Normalised bounding box returned by Moondream (all values 0.0–1.0).

    Convert to pixel coords:
        x1_px = int(box.x_min * frame_width)
        y1_px = int(box.y_min * frame_height)
        x2_px = int(box.x_max * frame_width)
        y2_px = int(box.y_max * frame_height)
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detector:
    """Wraps a Moondream model and exposes a simple detect() method."""

    def __init__(
        self,
        backend: Literal["cloud", "local"] = "cloud",
        api_key: str = "",
    ) -> None:
        if backend == "local":
            print("[Detector] Loading local Moondream model (md.photon)…")
            self._model = md.photon()
            print("[Detector] Local model ready.")
        else:
            if not api_key:
                raise ValueError(
                    "Cloud backend requires a Moondream API key.\n"
                    "Set MOONDREAM_API_KEY in your .env or pass --api-key."
                )
            self._model = md.vl(api_key=api_key)
            print("[Detector] Using Moondream Cloud API.")

    def detect(self, image: Image.Image, object_label: str) -> list[BoundingBox]:
        """
        Run open-vocabulary object detection on *image*.

        Args:
            image:        PIL Image (RGB).
            object_label: Plain-English description of what to detect,
                          e.g. "person", "coffee mug", "cat".

        Returns:
            List of BoundingBox instances (may be empty).
        """
        result = self._model.detect(image, object_label)
        raw_objects: list[dict] = result.get("objects", [])
        return [
            BoundingBox(
                x_min=obj["x_min"],
                y_min=obj["y_min"],
                x_max=obj["x_max"],
                y_max=obj["y_max"],
            )
            for obj in raw_objects
        ]
