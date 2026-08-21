"""
stream.py — Video source abstractions.

WebcamStream  : thin wrapper around a local USB/built-in webcam.
TapoStream    : threaded RTSP reader for Tapo C100 (or any RTSP camera).
                Uses a background thread so the buffer is always drained and
                model.detect() never blocks on stale frames.
"""

from __future__ import annotations

import cv2
from threading import Thread


class WebcamStream:
    """Simple wrapper around a local webcam (cv2.VideoCapture with an int index)."""

    def __init__(self, index: int = 0) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {index}. "
                "Is the camera connected and not in use by another process?"
            )

    def read(self) -> cv2.typing.MatLike | None:
        """Return the next frame, or None on failure."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self) -> None:
        self.cap.release()


class TapoStream:
    """
    Threaded RTSP reader for Tapo C100 (and compatible cameras).

    A daemon thread continuously calls cap.grab() to drain OpenCV's internal
    buffer, then cap.retrieve() to decode only when requested. This prevents
    frame lag that builds up when inference is slower than the camera FPS.

    Usage:
        stream = TapoStream("rtsp://user:pass@192.168.1.100:554/stream2")
        frame = stream.read()   # always the latest frame
        stream.stop()
    """

    def __init__(self, rtsp_url: str) -> None:
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open RTSP stream: {rtsp_url}\n"
                "Check:\n"
                "  • Credentials (username / password)\n"
                "  • Camera IP and port 554 are reachable\n"
                "  • Camera and PC are on the same subnet\n"
                "  • Test in VLC: Media → Open Network Stream"
            )
        self.frame: cv2.typing.MatLike | None = None
        self.stopped: bool = False
        self._thread = Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        """Background loop: drain the buffer, decode, store latest frame."""
        while not self.stopped:
            if not self.cap.grab():
                print("[TapoStream] Stream lost — grab() failed.")
                break
            _, self.frame = self.cap.retrieve()

    def read(self) -> cv2.typing.MatLike | None:
        """Return the most recently decoded frame (may be None on startup)."""
        return self.frame

    def stop(self) -> None:
        self.stopped = True
        self._thread.join(timeout=2)
        self.cap.release()
