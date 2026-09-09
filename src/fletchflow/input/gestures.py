"""Per-hand gesture measurements from tracked landmarks. Camera space.

Every measurement here is scale-invariant — a ratio of two distances on the
same hand — so the thresholds work whether the player sits near the camera or
far away.

- pinch_ratio: thumb-tip to index-tip over wrist to middle-MCP. Pinched reads
  ~0.2-0.3, an open hand ~0.8-1.2. Grabs the string.
- fist_ratio: fingertip-to-wrist over MCP-to-wrist, averaged over the four
  fingers. A closed fist curls the tips in toward the wrist, so it reads
  ~0.7-1.1 against ~1.9-2.3 open. Holds the bow.

palm_size is the odd one out: an absolute size, not a ratio, used as the
depth proxy for 3D draw power. It takes the max of the palm's length and its
(ratio-corrected) knuckle width because rotating the hand about its wrist axis
foreshortens one but not the other. A false size reads as false depth.
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
    pinch_point: tuple[float, float]  # midpoint of thumb/index tips (string grip)
    grip_point: tuple[float, float] = (0.0, 0.0)  # palm centre (bow grip)
    pinch_ratio: float = float("inf")
    fist_ratio: float = float("inf")
    size: float = 0.11                # wrist->MCP; legacy depth-scale proxy
    palm_size: float = 0.11           # rotation-robust size, for 3D draw depth


@dataclass(frozen=True)
class GestureFrame:
    timestamp_ms: int
    left: HandGesture | None
    right: HandGesture | None

    def get(self, side: str) -> HandGesture | None:
        return self.left if side == "left" else self.right


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(*(a - b)))


def _fist_ratio(points: np.ndarray, wrist: np.ndarray) -> float:
    """Mean fingertip-to-wrist over MCP-to-wrist across the four fingers."""
    ratios = []
    for tip_i, mcp_i in config.FINGER_PAIRS:
        span = _dist(points[mcp_i, :2], wrist)
        if span > 1e-6:
            ratios.append(_dist(points[tip_i, :2], wrist) / span)
    if not ratios:
        return float("inf")
    return sum(ratios) / len(ratios)


def _measure(points: np.ndarray) -> HandGesture:
    wrist = points[config.WRIST, :2]
    thumb = points[config.THUMB_TIP, :2]
    index = points[config.INDEX_TIP, :2]
    mcp = points[config.MIDDLE_MCP, :2]

    hand_size = _dist(wrist, mcp)
    pinch = _dist(thumb, index)
    mid = (thumb + index) / 2.0
    grip = wrist + config.GRIP_PALM_FRACTION * (mcp - wrist)

    # Palm length vs knuckle width, whichever is currently less foreshortened
    knuckles = _dist(points[config.INDEX_MCP, :2], points[config.PINKY_MCP, :2])
    palm_size = max(hand_size, knuckles / config.PALM_WIDTH_RATIO, 1e-6)

    return HandGesture(
        wrist=(float(wrist[0]), float(wrist[1])),
        pinch_point=(float(mid[0]), float(mid[1])),
        grip_point=(float(grip[0]), float(grip[1])),
        pinch_ratio=pinch / hand_size if hand_size > 1e-6 else float("inf"),
        fist_ratio=_fist_ratio(points, wrist),
        size=hand_size,
        palm_size=palm_size,
    )


def extract(hand_frame: HandFrame) -> GestureFrame:
    return GestureFrame(
        timestamp_ms=hand_frame.timestamp_ms,
        left=_measure(hand_frame.left) if hand_frame.left is not None else None,
        right=_measure(hand_frame.right) if hand_frame.right is not None else None,
    )
