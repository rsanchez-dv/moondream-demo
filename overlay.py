"""
overlay.py — Drawing utilities for bounding boxes on OpenCV frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from detector import BoundingBox
import config


def draw_boxes(
    frame: np.ndarray,
    boxes: list[BoundingBox],
    label: str,
    color: tuple[int, int, int] = config.BOX_COLOR,
    thickness: int = config.BOX_THICKNESS,
    font_scale: float = config.LABEL_FONT_SCALE,
    font_thickness: int = config.LABEL_THICKNESS,
) -> np.ndarray:
    """
    Draw bounding boxes on *frame* (in-place) and return the frame.

    Moondream boxes are normalised [0, 1]; we scale to pixel coords here.

    Args:
        frame:         BGR OpenCV frame.
        boxes:         List of BoundingBox from Detector.detect().
        label:         Text shown above each box.
        color:         BGR colour tuple.
        thickness:     Rectangle line thickness in pixels.
        font_scale:    cv2.putText font scale.
        font_thickness: cv2.putText thickness.

    Returns:
        The same frame (mutated in-place) for easy chaining.
    """
    h, w = frame.shape[:2]

    for box in boxes:
        x1 = int(box.x_min * w)
        y1 = int(box.y_min * h)
        x2 = int(box.x_max * w)
        y2 = int(box.y_max * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label background for readability
        label_y = max(y1 - 8, 12)  # don't draw above frame edge
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # black text on coloured background
            font_thickness,
            cv2.LINE_AA,
        )

    return frame


def draw_hud(
    frame: np.ndarray,
    object_label: str,
    frame_count: int,
    frame_skip: int,
    box_count: int,
) -> np.ndarray:
    """
    Draw a small HUD (heads-up display) in the top-left corner.

    Shows the current detection target, frame counter, and object count.
    """
    lines = [
        f"Target : {object_label}",
        f"Frame  : {frame_count}  (skip={frame_skip})",
        f"Objects: {box_count}",
        "Press Q to quit | R to relabel",
    ]
    y = 18
    for line in lines:
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18

    return frame
