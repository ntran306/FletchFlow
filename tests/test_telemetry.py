"""CSV telemetry logging, focused on the draw_grip / release_rule columns
added for the release-rule follow-up (see vision/telemetry.py docstring)."""

import csv

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState
from fletchflow.input.gestures import GestureFrame, HandGesture
from fletchflow.vision.telemetry import COLUMNS, TelemetryLogger

DOCK = config.DOCK_POS


def _hand() -> HandGesture:
    return HandGesture(
        wrist=DOCK, pinch_point=DOCK, grip_point=DOCK,
        pinch_ratio=0.30, fist_ratio=0.95,
        size=config.REFERENCE_HAND_SIZE, palm_size=config.REFERENCE_HAND_SIZE,
    )


def _frame(t: int = 33) -> GestureFrame:
    return GestureFrame(timestamp_ms=t, left=_hand(), right=_hand())


def _snapshot(t: int = 33) -> BowSnapshot:
    return BowSnapshot(
        timestamp_ms=t, state=BowState.DRAWN, anchor=DOCK, draw_point=DOCK,
        power=0.5, fired_power=None, scale=1.0, draw_power_hw=1.2,
    )


def test_header_ends_with_draw_grip_and_release_rule(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(str(path))
    logger.close()

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(COLUMNS)
    assert COLUMNS[-2:] == ("draw_grip", "release_rule")


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
