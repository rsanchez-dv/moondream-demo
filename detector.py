"""
detector.py — Moondream model wrapper.

Supports three backends:
  • cloud  — md.vl(api_key=...)   — hosted API, no GPU needed.
  • local  — md.photon()          — on-device via Moondream SDK, GPU recommended.
  • ollama — ollama.chat()        — local Ollama server running moondream model.

The Detector class is intentionally thin so the rest of the app
doesn't need to know which backend is active.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

from PIL import Image
from pydantic import BaseModel


# ── Shared output type ────────────────────────────────────────────────────────

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


# ── Pydantic schemas for Ollama structured output ─────────────────────────────

class _OllamaBBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class _OllamaDetections(BaseModel):
    objects: list[_OllamaBBox]


# ── Detector ──────────────────────────────────────────────────────────────────

class Detector:
    """Wraps a Moondream model and exposes a simple detect() method."""

    def __init__(
        self,
        backend: Literal["cloud", "local", "ollama"] = "ollama",
        api_key: str = "",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "moondream",
    ) -> None:
        self._backend = backend

        if backend == "local":
            import moondream as md
            print("[Detector] Loading local Moondream model (md.photon)…")
            self._model = md.photon()
            print("[Detector] Local model ready.")

        elif backend == "cloud":
            import moondream as md
            if not api_key:
                raise ValueError(
                    "Cloud backend requires a Moondream API key.\n"
                    "Set MOONDREAM_API_KEY in your .env or pass --api-key."
                )
            self._model = md.vl(api_key=api_key)
            print("[Detector] Using Moondream Cloud API.")

        elif backend == "ollama":
            import ollama
            self._ollama = ollama
            self._ollama_model = ollama_model

            # Point the client at the configured host
            self._ollama_client = ollama.Client(host=ollama_host)

            # Verify Ollama is reachable and the model exists
            try:
                available = [m.model for m in self._ollama_client.list().models]
            except Exception as e:
                raise RuntimeError(
                    f"Cannot reach Ollama at {ollama_host}.\n"
                    f"  Make sure 'ollama serve' is running.\n"
                    f"  Error: {e}"
                )

            # Normalise the model name for comparison (strip digest suffix)
            normalised = [m.split(":")[0] for m in available]
            model_base = ollama_model.split(":")[0]
            if model_base not in normalised:
                raise RuntimeError(
                    f"Model '{ollama_model}' not found in Ollama.\n"
                    f"  Run: ollama pull {ollama_model}\n"
                    f"  Available: {available}"
                )

            print(f"[Detector] Using Ollama — model: {ollama_model} @ {ollama_host}")

    # ── Public API ────────────────────────────────────────────────────────────

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
        if self._backend == "ollama":
            return self._detect_ollama(image, object_label)
        else:
            return self._detect_sdk(image, object_label)

    # ── Backend implementations ───────────────────────────────────────────────

    def _detect_sdk(self, image: Image.Image, object_label: str) -> list[BoundingBox]:
        """Detection via the native Moondream Python SDK (cloud or local)."""
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

    def _detect_ollama(self, image: Image.Image, object_label: str) -> list[BoundingBox]:
        """
        Detection via a local Ollama server running the moondream model.

        Uses Moondream's trained trigger phrase 'Bounding box: {object}' and
        Ollama's format= parameter to enforce structured JSON output via the
        Pydantic schema, with temperature=0 for deterministic results.
        """
        # Convert PIL Image → raw JPEG bytes (in-memory, no temp file)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        response = self._ollama_client.chat(
            model=self._ollama_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Bounding box: {object_label}",
                    "images": [img_bytes],  # SDK auto-encodes to base64
                }
            ],
            format=_OllamaDetections.model_json_schema(),
            options={"temperature": 0},
        )

        raw = _OllamaDetections.model_validate_json(response.message.content)

        # Clamp to [0, 1] to guard against occasional out-of-range model output
        return [
            BoundingBox(
                x_min=max(0.0, min(1.0, b.x_min)),
                y_min=max(0.0, min(1.0, b.y_min)),
                x_max=max(0.0, min(1.0, b.x_max)),
                y_max=max(0.0, min(1.0, b.y_max)),
            )
            for b in raw.objects
            # Filter out degenerate boxes where min >= max
            if b.x_max > b.x_min and b.y_max > b.y_min
        ]
