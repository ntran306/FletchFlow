"""CSV logging of per-frame tracking signals.

Two jobs. The original one: test whether an aiming pose (hands converged in 2D,
one hand clearly larger because it is nearer the camera) reliably precedes
losing the rear hand to occlusion — if it holds, hand loss becomes evidence of
intent rather than failure.

Added for M4b: log the gesture ratios and the 3D pull directly, so a playtest
yields the distributions behind FIST_ON/FIST_OFF and CAM_FOCAL_NORM instead of
leaving them as the geometry-derived guesses they still are. Load with:

    import pandas as pd; df = pd.read_csv(path)
    df[df.state == "held"][["left_fist", "right_fist"]].describe()

Added for the release-rule follow-up: the trailing draw_grip and release_rule
columns record which grip drew the string and which release rule was active
that frame, so a playtest can compare both_open against grip_aware after the
fact instead of only live via the G key.

Nothing here feeds gameplay.
"""

from __future__ import annotations

import math

COLUMNS = (
    "t_ms", "state", "left_seen", "right_seen", "left_size", "right_size",
    "pinch_dist_2d", "size_ratio", "bow_side", "draw_side",
    # M4b: gesture ratios and the 3D draw, per hand and per frame
    "left_pinch", "right_pinch", "left_fist", "right_fist",
    "left_palm", "right_palm", "power", "pull_hw", "scale", "fired_power",
    "draw_grip", "release_rule",
)


class TelemetryLogger:
    def __init__(self, path: str) -> None:
        self._file = open(path, "w", encoding="utf-8", newline="")
        self._file.write(",".join(COLUMNS) + "\n")
        self._rows = 0

    def log(
        self,
        gesture_frame,
        snapshot,
        bow_side=None,
        draw_side=None,
        draw_grip="",
        release_rule="",
    ) -> None:
        left, right = gesture_frame.left, gesture_frame.right

        pinch_dist = ""
        size_ratio = ""
        if left is not None and right is not None:
            pinch_dist = f"{math.dist(left.pinch_point, right.pinch_point):.4f}"
            big, small = max(left.size, right.size), min(left.size, right.size)
            if small > 1e-6:
                size_ratio = f"{big / small:.3f}"

        def num(hand, attr, fmt="{:.4f}"):
            if hand is None:
                return ""
            value = getattr(hand, attr)
            return "" if value == float("inf") else fmt.format(value)

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
            num(left, "pinch_ratio", "{:.3f}"),
            num(right, "pinch_ratio", "{:.3f}"),
            num(left, "fist_ratio", "{:.3f}"),
            num(right, "fist_ratio", "{:.3f}"),
            num(left, "palm_size"),
            num(right, "palm_size"),
            f"{snapshot.power:.3f}",
            f"{snapshot.draw_power_hw:.3f}",
            f"{snapshot.scale:.3f}",
            "" if snapshot.fired_power is None else f"{snapshot.fired_power:.3f}",
            draw_grip or "",
            release_rule or "",
        )
        self._file.write(",".join(row) + "\n")

        self._rows += 1
        if self._rows % 30 == 0:
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
