"""Per-hand gesture measurements from tracked landmarks. Camera space.

pinch_ratio is scale-invariant: thumb-tip-to-index-tip distance divided by
wrist-to-middle-MCP distance (a stable proxy for hand size), so the same
thresholds work whether the player sits near the camera or far away.
Pinched reads ~0.2-0.3, an open hand ~0.8-1.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fletchflow import config
from fletchflow.vision.tracker import HandFrame


@dataclass(frozen=True)
class HandGesture:
    wrist: tuple[float, float]        # mirrored normalized coords
    pinch_point: tuple[float, float]  # midpoint of thumb/index tips (string grab)
    pinch_ratio: float
    size: float = 0.11                # wrist->MCP distance; depth-scale proxy


@dataclass(frozen=True)
class GestureFrame:
    timestamp_ms: int
    left: HandGesture | None
    right: HandGesture | None

    def get(self, side: str) -> HandGesture | None:
        return self.left if side == "left" else self.right


def _measure(points: np.ndarray) -> HandGesture:
    wrist = points[config.WRIST, :2]
    thumb = points[config.THUMB_TIP, :2]
    index = points[config.INDEX_TIP, :2]
    mcp = points[config.MIDDLE_MCP, :2]

    hand_size = float(math.hypot(*(wrist - mcp)))
    pinch = float(math.hypot(*(thumb - index)))
    mid = (thumb + index) / 2.0
    return HandGesture(
        wrist=(float(wrist[0]), float(wrist[1])),
        pinch_point=(float(mid[0]), float(mid[1])),
        pinch_ratio=pinch / hand_size if hand_size > 1e-6 else float("inf"),
        size=hand_size,
    )


def extract(hand_frame: HandFrame) -> GestureFrame:
    return GestureFrame(
        timestamp_ms=hand_frame.timestamp_ms,
        left=_measure(hand_frame.left) if hand_frame.left is not None else None,
        right=_measure(hand_frame.right) if hand_frame.right is not None else None,
    )
