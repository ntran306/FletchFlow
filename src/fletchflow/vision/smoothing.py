"""One Euro filter (Casiez et al., CHI 2012) — the standard adaptive low-pass
for noisy pointing input: smooth when the hand is still, responsive when it moves.
"""

from __future__ import annotations

import math

import numpy as np

from fletchflow import config
from fletchflow.vision.tracker import HandFrame


def _alpha(cutoff, dt: float):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters an ndarray signal element-wise.

    Call the instance with the new sample and a monotonically increasing
    timestamp in seconds. The first sample (and the first after reset())
    passes through unchanged.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None or self._t_prev is None or t <= self._t_prev:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        dt = t - self._t_prev
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)  # element-wise: fast-moving components smooth less
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class HandSmoother:
    """Applies One Euro filtering to the key landmarks (config.KEY_LANDMARKS)
    of each hand in a HandFrame — only x and y; z is unused downstream.

    A side's filter resets when that hand disappears, so stale velocity state
    never bleeds into a re-acquired hand.
    """

    def __init__(self) -> None:
        self._key = list(config.KEY_LANDMARKS)
        self._filters = {
            "left": self._new_filter(),
            "right": self._new_filter(),
        }

    @staticmethod
    def _new_filter() -> OneEuroFilter:
        return OneEuroFilter(
            min_cutoff=config.ONE_EURO_MIN_CUTOFF,
            beta=config.ONE_EURO_BETA,
            d_cutoff=config.ONE_EURO_D_CUTOFF,
        )

    def smooth(self, hand_frame: HandFrame) -> HandFrame:
        t = hand_frame.timestamp_ms / 1000.0
        return HandFrame(
            timestamp_ms=hand_frame.timestamp_ms,
            left=self._smooth_side("left", hand_frame.left, t),
            right=self._smooth_side("right", hand_frame.right, t),
        )

    def _smooth_side(self, side: str, points, t: float):
        if points is None:
            self._filters[side].reset()
            return None
        smoothed = points.copy()
        smoothed[self._key, :2] = self._filters[side](points[self._key, :2], t)
        return smoothed
