"""
config.py — Central defaults for moondream-demo.

All values can be overridden via CLI arguments or environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Moondream ──────────────────────────────────────────────────────────────────
MOONDREAM_API_KEY: str = os.getenv("MOONDREAM_API_KEY", "")

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "moondream")

# ── Video source ───────────────────────────────────────────────────────────────
# 0 = default webcam index; pass an RTSP URL string for IP cameras
DEFAULT_SOURCE: str | int = 0

# Tapo C100 RTSP URL (read from .env if set, else empty string → webcam mode)
TAPO_RTSP_URL: str = os.getenv("TAPO_RTSP_URL", "")

# ── Detection ──────────────────────────────────────────────────────────────────
DEFAULT_OBJECT: str = "person"

# Run model inference every N frames (higher = faster UI, staler detections)
FRAME_SKIP: int = 5

# Bounding-box colour (BGR) and thickness
BOX_COLOR: tuple[int, int, int] = (0, 255, 0)   # green
BOX_THICKNESS: int = 2

# Label font scale and thickness
LABEL_FONT_SCALE: float = 0.6
LABEL_THICKNESS: int = 2
