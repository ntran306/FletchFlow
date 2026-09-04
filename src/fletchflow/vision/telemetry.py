"""CSV logging of per-frame tracking signals.

Exists to test one hypothesis: that an aiming pose (hands converged in 2D,
one hand clearly larger because it is nearer the camera) reliably precedes
losing the rear hand to occlusion. If it holds, hand loss becomes evidence of
intent rather than a failure. Nothing here feeds gameplay yet.
"""

from __future__ import annotations

import math

COLUMNS = (
    "t_ms", "state", "left_seen", "right_seen", "left_size", "right_size",
    "pinch_dist_2d", "size_ratio", "bow_side", "draw_side",
)


class TelemetryLogger:
    def __init__(self, path: str) -> None:
        self._file = open(path, "w", encoding="utf-8", newline="")
        self._file.write(",".join(COLUMNS) + "\n")
        self._rows = 0

    def log(self, gesture_frame, snapshot, bow_side=None, draw_side=None) -> None:
        left, right = gesture_frame.left, gesture_frame.right

        pinch_dist = ""
        size_ratio = ""
        if left is not None and right is not None:
            pinch_dist = f"{math.dist(left.pinch_point, right.pinch_point):.4f}"
            big, small = max(left.size, right.size), min(left.size, right.size)
            if small > 1e-6:
                size_ratio = f"{big / small:.3f}"

        row = (
            str(snapshot.timestamp_ms),
            snapshot.state.value,
            str(int(left is not None)),
            str(int(right is not None)),
            f"{left.size:.4f}" if left is not None else "",
            f"{right.size:.4f}" if right is not None else "",
            pinch_dist,
            size_ratio,
            bow_side or "",
            draw_side or "",
        )
        self._file.write(",".join(row) + "\n")

        self._rows += 1
        if self._rows % 30 == 0:
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
