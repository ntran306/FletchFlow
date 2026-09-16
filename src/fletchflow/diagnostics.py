"""Diagnostics helpers for the --selfcheck verdict.

Two problems with the naive selfcheck implementation:

1. Saving a PNG once per second via pygame.image.save on the render thread
   holds the GIL for the whole PNG encode (~200-350 ms at 1280x720) and
   freezes the camera and tracking threads while it runs. See
   AsyncFrameWriter below for the fix and the measured numbers.
2. Reading camera.fps / pipeline.fps (EMAs) at the end of the run only
   reflects the last few frames, so whatever happened in the instant before
   the run ended decides pass/fail. mean_rate() computes a true average
   rate between two (counter, perf_counter-seconds) snapshots instead.
"""

from __future__ import annotations

import queue
import threading

import cv2
import numpy as np
import pygame


def mean_rate(count0: int, t0: float, count1: int, t1: float) -> float:
    """Events per second between two (counter, perf_counter-seconds) snapshots.

    Returns 0.0 if t1 <= t0 or count1 <= count0.
    """
    if t1 <= t0 or count1 <= count0:
        return 0.0
    return (count1 - count0) / (t1 - t0)


class AsyncFrameWriter:
    """Encodes and writes PNGs on a worker thread so the encode never holds
    the GIL on the render thread.

    pygame.image.save holds the Python GIL for the entire PNG encode
    (~200-350 ms at 1280x720). Once per second that freezes the camera and
    tracking threads. Measured over a 15 s steady-state window: without the
    saves, camera 29.0 / tracker 29.9 fps, worst gap between tracked frames
    83 ms, SELFCHECK OK. With the saves: camera 22.1 / tracker 22.4 fps,
    worst gap 357 ms, 14 gaps over 100 ms, SELFCHECK FAIL. The selfcheck was
    failing on its own instrumentation, not on the app.

    submit() runs on the render thread and stays cheap: it only copies the
    surface to raw RGB bytes and enqueues them. The actual encode
    (cv2.imwrite, which releases the GIL) happens on one daemon worker
    thread, so it never stalls the camera or tracking threads. If the
    worker falls behind, submit() drops the frame rather than block the
    render loop.
    """

    def __init__(self, max_pending: int = 4) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._closed = False
        self.failed = 0  # writes that raised or that cv2.imwrite reported False
        self._thread = threading.Thread(
            target=self._loop, name="frame-writer", daemon=True
        )
        self._thread.start()

    def submit(self, surface, path: str) -> None:
        raw = pygame.image.tobytes(surface, "RGB")
        w, h = surface.get_size()
        try:
            self._queue.put_nowait((raw, w, h, path))
        except queue.Full:
            pass  # never block the render loop — drop the frame instead

    def close(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        # Bounded, not a bare put(): this runs in main()'s `finally`, and if the
        # worker had died a blocking put on a full queue would hang the game on
        # exit instead of letting it quit.
        try:
            self._queue.put(None, timeout=timeout)  # sentinel, after pending writes
        except queue.Full:
            return
        self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            raw, w, h, path = item
            # One bad write must not kill the worker: a dead worker stops
            # draining the queue, and close() would then wait on it. imwrite
            # raises for an unknown extension but only returns False for a
            # missing directory, so both count as failures.
            try:
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(path, bgr):
                    self.failed += 1
            except Exception:
                self.failed += 1
