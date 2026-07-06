"""Threaded webcam capture.

The capture loop runs on its own thread and only ever keeps the newest frame,
so consumers read the latest image without ever blocking on the camera. It
does capture and nothing else — tracking runs on its own thread (pipeline.py)
because a detect call takes as long as a whole frame interval and would drop
frames if it ran here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR, as captured (not mirrored)
    timestamp: float


class Camera:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: float = 30.0,
        manual_exposure: float | None = None,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps_request = fps
        self._manual_exposure = manual_exposure
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._running = False
        self._fps = 0.0

    def start(self) -> None:
        # CAP_DSHOW opens noticeably faster than the default MSMF backend on Windows
        capture = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Without an explicit rate the driver sometimes negotiates a 15 fps
        # low-light mode — bistably, varying between opens (measured 2026-07-06)
        capture.set(cv2.CAP_PROP_FPS, self._fps_request)
        if self._manual_exposure is not None:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DSHOW: manual mode
            capture.set(cv2.CAP_PROP_EXPOSURE, self._manual_exposure)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"could not open camera {self._index} — is another app using it?"
            )
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    @property
    def fps(self) -> float:
        return self._fps

    def _loop(self) -> None:
        assert self._capture is not None
        previous = time.perf_counter()
        while self._running:
            ok, image = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            interval = now - previous
            previous = now
            if interval > 0:
                # EMA keeps the FPS readout steady instead of flickering
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / interval) if self._fps else 1.0 / interval
            with self._lock:
                self._latest = Frame(image=image, timestamp=now)
