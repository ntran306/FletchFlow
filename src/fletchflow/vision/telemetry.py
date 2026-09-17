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

Added for M4c phase 1: left_depth_m, right_depth_m, left_residual_px,
right_residual_px, left_inplane_deg and right_inplane_deg log the metric
Procrustes pose fit (input/hand_pose.py) per hand and per frame — empty when
that hand had no world landmarks or the fit was rejected. Optional columns:
telemetry_report.py pins its own REQUIRED_COLUMNS rather than reading this
tuple, so it keeps loading older and newer CSVs alike.

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
    # M4c phase 1: metric hand pose, per hand and per frame (§4.7.2)
    "left_depth_m", "right_depth_m", "left_residual_px", "right_residual_px",
    "left_inplane_deg", "right_inplane_deg",
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

        def pose_num(hand, attr, fmt):
            if hand is None or hand.pose is None:
                return ""
            return fmt.format(getattr(hand.pose, attr))

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
            pose_num(left, "depth_m", "{:.4f}"),
            pose_num(right, "depth_m", "{:.4f}"),
            pose_num(left, "residual_px", "{:.2f}"),
            pose_num(right, "residual_px", "{:.2f}"),
            pose_num(left, "inplane_deg", "{:.2f}"),
            pose_num(right, "inplane_deg", "{:.2f}"),
        )
        self._file.write(",".join(row) + "\n")

        self._rows += 1
        if self._rows % 30 == 0:
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
