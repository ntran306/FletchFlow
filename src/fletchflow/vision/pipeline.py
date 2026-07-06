"""Tracking pipeline: detection + smoothing on a dedicated thread.

detect_for_video releases the GIL during inference (verified 2026-07-06:
camera capture held ~27 fps while a tight detect loop ran on another thread),
so this thread runs concurrently with capture and rendering. Detection costs
~19 ms with no hands and ~29 ms with two hands at 720p — slightly less than
the camera's 33 ms frame interval, so in practice every frame gets tracked.
Downscaling the model input does not help; per-hand landmark inference
dominates, not image size.
"""

from __future__ import annotations

import threading
import time
import traceback

from fletchflow.vision.camera import Camera
from fletchflow.vision.smoothing import HandSmoother
from fletchflow.vision.tracker import HandFrame, HandTracker


class TrackingPipeline:
    """Pulls the newest camera frame, runs tracker + smoother, publishes the
    latest HandFrame. Skips frames if detection falls behind — latest wins."""

    def __init__(self, camera: Camera) -> None:
        self._camera = camera
        self._tracker = HandTracker()
        self._smoother = HandSmoother()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: HandFrame | None = None
        self._running = False
        self.fps = 0.0  # tracked frames per second (EMA)
        self.ms = 0.0   # last detect+smooth cost

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._tracker.close()

    def latest(self) -> HandFrame | None:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        last_timestamp = -1.0
        last_time = 0.0
        try:
            while self._running:
                frame = self._camera.latest()
                if frame is None or frame.timestamp == last_timestamp:
                    time.sleep(0.002)  # no new frame yet
                    continue
                last_timestamp = frame.timestamp

                t0 = time.perf_counter()
                hand_frame = self._tracker.track(frame)
                if hand_frame is None:
                    continue
                hand_frame = self._smoother.smooth(hand_frame)
                now = time.perf_counter()

                self.ms = (now - t0) * 1000.0
                if last_time:
                    interval = now - last_time
                    self.fps = 0.9 * self.fps + 0.1 / interval if self.fps else 1.0 / interval
                last_time = now
                with self._lock:
                    self._latest = hand_frame
        except Exception:
            # Fail loudly — a silently dead tracking thread would just
            # freeze the hand overlay
            traceback.print_exc()
            self._running = False
