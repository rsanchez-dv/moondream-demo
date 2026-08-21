"""
main.py — Moondream webcam / IP-camera object detection demo.

Usage examples:
  # Cloud API, built-in webcam, detect "person"
  python main.py --api-key YOUR_KEY

  # Cloud API, Tapo C100 RTSP, detect "cat"
  python main.py --api-key YOUR_KEY --source rtsp://user:pass@192.168.1.100:554/stream2 --object cat

  # Local inference, built-in webcam
  python main.py --local

  # Override frame-skip (higher = faster UI, staler detections)
  python main.py --api-key YOUR_KEY --frame-skip 10

Keyboard shortcuts while running:
  Q  — quit
  R  — type a new detection label in the terminal (without stopping the feed)
"""

from __future__ import annotations

import argparse
import sys

import cv2
from PIL import Image

import config
from detector import Detector
from overlay import draw_boxes, draw_hud
from stream import TapoStream, WebcamStream


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Moondream real-time object detection — webcam or RTSP camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model backend
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--api-key",
        metavar="KEY",
        default=config.MOONDREAM_API_KEY or None,
        help="Moondream Cloud API key. Can also be set via MOONDREAM_API_KEY in .env.",
    )
    backend.add_argument(
        "--local",
        action="store_true",
        help="Use local inference via md.photon() instead of the cloud API.",
    )

    # Video source
    parser.add_argument(
        "--source",
        default=config.TAPO_RTSP_URL or "0",
        help=(
            "Video source: webcam index (e.g. 0) or RTSP URL "
            "(e.g. rtsp://user:pass@192.168.1.100:554/stream2). "
            "Can also be set via TAPO_RTSP_URL in .env."
        ),
    )

    # Detection config
    parser.add_argument(
        "--object",
        default=config.DEFAULT_OBJECT,
        help="Object to detect (plain English, e.g. 'person', 'coffee mug', 'cat').",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=config.FRAME_SKIP,
        help="Run model inference every N frames (higher = faster UI, staler boxes).",
    )

    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_source(source_str: str) -> str | int:
    """Convert the --source string to either an int (webcam index) or a URL."""
    try:
        return int(source_str)
    except ValueError:
        return source_str


def open_stream(source: str | int) -> WebcamStream | TapoStream:
    """Return the appropriate stream object for the given source."""
    if isinstance(source, int):
        print(f"[main] Opening webcam at index {source}…")
        return WebcamStream(index=source)
    else:
        print(f"[main] Opening RTSP stream: {source}")
        return TapoStream(rtsp_url=source)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # ── Init model ──
    if args.local:
        detector = Detector(backend="local")
    else:
        if not args.api_key:
            print(
                "[ERROR] No API key provided.\n"
                "  • Pass --api-key YOUR_KEY, or\n"
                "  • Set MOONDREAM_API_KEY in a .env file, or\n"
                "  • Use --local for on-device inference.",
                file=sys.stderr,
            )
            sys.exit(1)
        detector = Detector(backend="cloud", api_key=args.api_key)

    # ── Init stream ──
    source = resolve_source(args.source)
    try:
        stream = open_stream(source)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    object_label: str = args.object
    frame_skip: int = args.frame_skip
    frame_count: int = 0
    last_boxes: list = []

    print(f"[main] Detecting '{object_label}' every {frame_skip} frames. Press Q to quit.")

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            frame_count += 1

            # ── Run detection every N frames ──
            if frame_count % frame_skip == 0:
                # Convert BGR (OpenCV) → RGB (PIL / Moondream)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                try:
                    last_boxes = detector.detect(pil_image, object_label)
                except Exception as exc:
                    print(f"[Detector] Error: {exc}")
                    last_boxes = []

            # ── Draw overlays ──
            draw_boxes(frame, last_boxes, object_label)
            draw_hud(frame, object_label, frame_count, frame_skip, len(last_boxes))

            cv2.imshow("Moondream Object Detection", frame)

            # ── Keyboard handling ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[main] Quit requested.")
                break
            elif key == ord("r"):
                # Prompt the user for a new label without closing the window
                new_label = input(f"[main] New object to detect (current='{object_label}'): ").strip()
                if new_label:
                    object_label = new_label
                    last_boxes = []  # clear stale boxes for the old label
                    print(f"[main] Now detecting: '{object_label}'")

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("[main] Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(parse_args())
