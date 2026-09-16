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

from fletchflow import config


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
        self._interval = 0.0   # EMA of the inter-frame gap, in seconds
        self.backend = "?"
        self.frames = 0

    def start(self) -> None:
        capture = None
        for backend in self._backends():
            capture = self._open(backend)
            if capture is not None:
                self.backend = "MSMF" if backend == cv2.CAP_MSMF else "DSHOW"
                break
        if capture is None:
            raise RuntimeError(
                f"could not open camera {self._index} — is another app using it?"
            )
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    @staticmethod
    def _backends() -> tuple[int, ...]:
        """MSMF first: it sustains 30 fps at 720p where DSHOW gives 10 on this
        machine, and opens faster once HW transforms are disabled (see
        fletchflow/__init__.py). DSHOW stays as a fallback for cameras MSMF
        cannot open at all."""
        if config.PREFER_MSMF:
            return (cv2.CAP_MSMF, cv2.CAP_DSHOW)
        return (cv2.CAP_DSHOW, cv2.CAP_MSMF)

    def _open(self, backend: int):
        capture = cv2.VideoCapture(self._index, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Without an explicit rate the driver sometimes negotiates a 15 fps
        # low-light mode — bistably, varying between opens (measured 2026-07-06)
        capture.set(cv2.CAP_PROP_FPS, self._fps_request)
        if self._manual_exposure is not None:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode
            capture.set(cv2.CAP_PROP_EXPOSURE, self._manual_exposure)
        if not capture.isOpened():
            capture.release()
            return None
        ok, _ = capture.read()  # an opened-but-unusable device fails here
        if not ok:
            capture.release()
            return None
        return capture

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
        previous: float | None = None
        while self._running:
            ok, image = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            # Smooth the INTERVAL and invert it, never the other way round.
            # Averaging 1/interval is biased high because the reciprocal is
            # convex: frames arriving in bursts (a few sub-millisecond gaps
            # among many long ones) drag the mean of the reciprocals far above
            # the reciprocal of the mean. That is why a 10 fps camera used to
            # report 100-190 fps here, which hid the DSHOW frame-rate problem
            # for two months. The first frame is skipped because `previous`
            # would otherwise be timed from before the loop started.
            if previous is not None:
                interval = now - previous
                if interval > 0:
                    self._interval = (
                        interval if self._interval == 0.0
                        else 0.9 * self._interval + 0.1 * interval
                    )
                    self._fps = 1.0 / self._interval
            previous = now
            with self._lock:
                self._latest = Frame(image=image, timestamp=now)
                self.frames += 1
