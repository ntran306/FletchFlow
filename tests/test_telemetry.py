"""CSV telemetry logging, focused on the draw_grip / release_rule columns
added for the release-rule follow-up (see vision/telemetry.py docstring)."""

import csv

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState
from fletchflow.input.gestures import GestureFrame, HandGesture
from fletchflow.input.hand_pose import HandPose3D
from fletchflow.vision.telemetry import COLUMNS, TelemetryLogger

DOCK = config.DOCK_POS


def _hand(pose: HandPose3D | None = None) -> HandGesture:
    return HandGesture(
        wrist=DOCK, pinch_point=DOCK, grip_point=DOCK,
        pinch_ratio=0.30, fist_ratio=0.95,
        size=config.REFERENCE_HAND_SIZE, palm_size=config.REFERENCE_HAND_SIZE,
        pose=pose,
    )


def _frame(t: int = 33) -> GestureFrame:
    return GestureFrame(timestamp_ms=t, left=_hand(), right=_hand())


def _snapshot(t: int = 33) -> BowSnapshot:
    return BowSnapshot(
        timestamp_ms=t, state=BowState.DRAWN, anchor=DOCK, draw_point=DOCK,
        power=0.5, fired_power=None, scale=1.0, draw_power_hw=1.2,
    )


def test_header_ends_with_pose_columns(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(str(path))
    logger.close()

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(COLUMNS)
    # draw_grip/release_rule are still there, just no longer the tail —
    # the M4c pose columns were appended after them.
    assert COLUMNS[-10:-8] == ("draw_grip", "release_rule")
    assert COLUMNS[-8:] == (
        "left_depth_m", "right_depth_m", "left_residual_px", "right_residual_px",
        "left_inplane_deg", "right_inplane_deg", "left_pose_fit", "right_pose_fit",
    )


def test_logged_draw_grip_and_release_rule_round_trip(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(str(path))
    logger.log(
        _frame(), _snapshot(), bow_side="left", draw_side="right",
        draw_grip="fist", release_rule="grip_aware",
    )
    logger.close()

    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["draw_grip"] == "fist"
    assert rows[0]["release_rule"] == "grip_aware"


def test_omitting_grip_and_rule_writes_empty_strings(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(str(path))
    logger.log(_frame(), _snapshot(), bow_side="left", draw_side="right")
    logger.close()

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)

    assert len(row) == len(COLUMNS)
    assert row[header.index("draw_grip")] == ""
    assert row[header.index("release_rule")] == ""


def test_pose_columns_round_trip(tmp_path):
    path = tmp_path / "telemetry.csv"
    pose = HandPose3D(
        position_m=(0.01, -0.02, 0.45),
        depth_m=0.45,
        px_per_m=2400.0,
        residual_px=7.25,
        inplane_deg=-3.5,
        fit="persp",
    )
    frame = GestureFrame(timestamp_ms=33, left=_hand(pose=pose), right=_hand(pose=None))

    logger = TelemetryLogger(str(path))
    logger.log(frame, _snapshot(), bow_side="left", draw_side="right")
    logger.close()

    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["left_depth_m"] == "0.4500"
    assert row["left_residual_px"] == "7.25"
    assert row["left_inplane_deg"] == "-3.50"
    assert row["right_depth_m"] == ""
    assert row["right_residual_px"] == ""
    assert row["right_inplane_deg"] == ""
    assert row["left_pose_fit"] == "persp"
    assert row["right_pose_fit"] == ""
