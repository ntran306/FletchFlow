"""Tests for telemetry_report.py, against synthetic playtest CSVs.

Most tests build rows directly (telemetry_report only ever sees the CSV,
so a hand-built CSV is the most direct test of its parsing/aggregation
logic) and write them with csv.DictWriter. The last section instead drives
the real BowStateMachine through the real TelemetryLogger: hand-built rows
can encode sequences the game never produces, and a generator that skipped
the release debounce once made every latency read 0 ms. REQUIRED_COLUMNS/ALL_COLUMNS below are derived from
telemetry_report.REQUIRED_COLUMNS rather than the live, evolving
fletchflow.vision.telemetry.COLUMNS, so these tests stay stable regardless
of what optional columns that module has added.
"""

from __future__ import annotations

import csv

import pytest

from fletchflow import config
from fletchflow import telemetry_report as tr

FRAME_MS = 33

REQUIRED_COLUMNS = tr.REQUIRED_COLUMNS
ALL_COLUMNS = REQUIRED_COLUMNS + ("draw_grip", "release_rule")


def make_row(t_ms, state, *, columns=ALL_COLUMNS, **fields) -> dict:
    """One CSV row, defaulted to an untracked/idle frame, overridden by fields."""
    row = {c: "" for c in columns}
    row["t_ms"] = str(t_ms)
    row["state"] = state
    row["left_seen"] = "0"
    row["right_seen"] = "0"
    row["bow_side"] = ""
    row["draw_side"] = ""
    row["power"] = "0.000"
    row["pull_hw"] = "0.000"
    row["scale"] = "1.000"
    for key, value in fields.items():
        if key not in columns:
            raise KeyError(f"not a configured column: {key}")
        row[key] = "" if value is None else value
    return row


def write_csv(path, rows, *, columns=ALL_COLUMNS) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def run_segment(rows, t, state, seconds, *, columns=ALL_COLUMNS, step_ms=FRAME_MS, **fields) -> int:
    """Append `seconds` worth of identical rows, step_ms apart. Returns the new t."""
    n = max(1, round(seconds * 1000 / step_ms))
    for _ in range(n):
        t += step_ms
        rows.append(make_row(t, state, columns=columns, **fields))
    return t


def add_cycle(rows, t, *, columns=ALL_COLUMNS, bow_side="left", draw_side="right",
              held_fields=None, drawn_rows, resolution="fire", resolution_fields=None) -> int:
    """One held row, then one drawn row per dict in drawn_rows (33ms apart),
    then either a release+fire row ("fire") or a held row (anything else,
    meaning the draw ended without firing). Returns the new t."""
    t += FRAME_MS
    hf = {"left_seen": "1", "right_seen": "0", "left_fist": "0.900", "left_pinch": "0.450"}
    hf.update(held_fields or {})
    rows.append(make_row(t, "held", columns=columns, bow_side=bow_side, **hf))

    for fields in drawn_rows:
        t += FRAME_MS
        df = {"left_seen": "1", "right_seen": "1",
              "left_fist": "0.900", "left_pinch": "0.450"}
        df.update(fields)
        rows.append(make_row(
            t, "drawn", columns=columns, bow_side=bow_side, draw_side=draw_side, **df
        ))

    t += FRAME_MS
    if resolution == "fire":
        rf = {"left_seen": "1", "right_seen": "1",
              "left_fist": "0.900", "left_pinch": "0.450",
              "right_fist": "2.000", "right_pinch": "0.900", "fired_power": "1.000"}
        rf.update(resolution_fields or {})
        rows.append(make_row(
            t, "released", columns=columns, bow_side=bow_side, draw_side=draw_side, **rf
        ))
    else:
        rf = {"left_seen": "1", "right_seen": "0",
              "left_fist": "0.900", "left_pinch": "0.450"}
        rf.update(resolution_fields or {})
        rows.append(make_row(t, "held", columns=columns, bow_side=bow_side, **rf))
    return t


# -- 1. clean session ---------------------------------------------------------


def build_clean_session() -> list:
    """3s docked (both hands open), then 5 clean grab->draw->release cycles."""
    rows = []
    t = 0
    t = run_segment(
        rows, t, "docked", 3.0,
        left_seen="1", right_seen="1",
        left_fist="2.000", right_fist="2.000",
        left_pinch="0.900", right_pinch="0.900",
    )
    for _ in range(5):
        t = run_segment(
            rows, t, "held", 1.5,
            left_seen="1", right_seen="1", bow_side="left",
            left_fist="0.900", left_pinch="0.450",
            right_fist="2.000", right_pinch="0.900",
        )
        n = max(1, round(1.2 * 1000 / FRAME_MS))
        for i in range(n):
            t += FRAME_MS
            pull = 2.4 * (i + 1) / n
            power = min(1.0, pull / 2.0)
            rows.append(make_row(
                t, "drawn",
                left_seen="1", right_seen="1",
                bow_side="left", draw_side="right",
                left_fist="0.900", left_pinch="0.450",
                right_fist="1.900", right_pinch="0.250",
                power=f"{power:.4f}", pull_hw=f"{pull:.4f}",
            ))
        t += FRAME_MS
        rows.append(make_row(
            t, "released",
            left_seen="1", right_seen="1",
            bow_side="left", draw_side="right",
            left_fist="0.900", left_pinch="0.450",
            right_fist="2.000", right_pinch="0.900",
            fired_power="1.000",
        ))
        t = run_segment(
            rows, t, "released", 0.3,
            left_seen="1", right_seen="1",
            bow_side="left", draw_side="right",
            left_fist="0.900", left_pinch="0.450",
            right_fist="2.000", right_pinch="0.900",
        )
    return rows


def test_clean_session(tmp_path):
    path = write_csv(tmp_path / "clean.csv", build_clean_session())
    result = tr.analyze(tr.load(path))

    assert result["transitions"]["fires"] == 5
    assert result["transitions"]["nocks"] == 5
    # Between-shot released->held is a cooldown, not an unexplained "other".
    assert result["transitions"]["cooldowns"] == 4
    assert result["transitions"]["other"] == 0

    assert abs(result["session"]["tracked_fps"] - 30.3) < 1.0

    assert result["shots"]["fired_power"]["p50"] == 1.0

    assert abs(result["shots"]["max_pull"]["p50"] - 2.4) < 0.1

    assert result["bow_grip"]["fist"]["p50"] == 0.9

    # All 5 draws use a pinch grip (fist stays ~1.9, well above FIST_ON).
    rr = result["release_responsiveness"]
    assert rr["by_grip"]["pinch"]["draws"] == 5
    assert rr["by_grip"]["pinch"]["fired"] == 5
    assert rr["by_grip"]["fist"]["draws"] == 0

    fist = result["suggested_thresholds"]["fist"]
    assert fist["ok"], fist
    assert 0.9 < fist["on"] < fist["off"] < 2.0

    draw = result["suggested_thresholds"]["draw_full_hw"]
    assert draw["ok"], draw
    assert abs(draw["suggested"] - 2.4) < 0.1

    # Report renders without crashing for a well-formed session too.
    report = tr.format_report(result)
    assert "5" in report


# -- 2. drop mid-draw ----------------------------------------------------------


def test_drop_mid_draw_counts_as_drop_not_fire(tmp_path):
    rows = []
    t = 0
    t = run_segment(rows, t, "docked", 0.5, left_seen="1", right_seen="0")
    t = run_segment(
        rows, t, "held", 0.5,
        left_seen="1", right_seen="0", bow_side="left",
        left_fist="0.900", left_pinch="0.450",
    )
    t = run_segment(
        rows, t, "drawn", 0.5,
        left_seen="1", right_seen="1", bow_side="left", draw_side="right",
        left_fist="0.900", left_pinch="0.450",
        right_fist="1.900", right_pinch="0.250",
        pull_hw="1.200", power="0.500",
    )
    # Bow hand lost entirely -> drop mid-draw, straight to docked, never fires.
    t = run_segment(rows, t, "docked", 0.2, left_seen="0", right_seen="0")

    path = write_csv(tmp_path / "drop_mid_draw.csv", rows)
    result = tr.analyze(tr.load(path))

    assert result["transitions"]["drops_while_drawn"] == 1
    assert result["transitions"]["fires"] == 0
    assert result["shots"]["count"] == 0


# -- 3. old-format CSV -----------------------------------------------------------


def test_old_format_csv_is_rejected(tmp_path):
    old_columns = REQUIRED_COLUMNS[: REQUIRED_COLUMNS.index("draw_side") + 1]
    path = tmp_path / "old_format.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow({
            "t_ms": "0", "state": "docked", "left_seen": "1", "right_seen": "0",
            "left_size": "0.1100", "right_size": "", "pinch_dist_2d": "",
            "size_ratio": "", "bow_side": "", "draw_side": "",
        })

    with pytest.raises(ValueError, match="left_fist"):
        tr.load(str(path))

    assert tr.main([str(path)]) == 2


# -- 4. tiny session --------------------------------------------------------------


def test_tiny_session_reports_insufficient_data(tmp_path):
    rows = []
    t = 0
    for _ in range(5):
        t += FRAME_MS
        rows.append(make_row(
            t, "docked", left_seen="1", right_seen="0",
            left_fist="2.000", left_pinch="0.900",
        ))

    path = write_csv(tmp_path / "tiny.csv", rows)
    result = tr.analyze(tr.load(path))
    report = tr.format_report(result)

    assert result["shots"]["count"] == 0
    assert "insufficient data" in report
    # Section 1 always reports the max gap, even with none over 100ms.
    assert result["session"]["gap_count"] == 0
    assert abs(result["session"]["max_gap_ms"] - FRAME_MS) < 1.0


# -- 5. gaps ------------------------------------------------------------------------


def test_gap_over_100ms_is_counted(tmp_path):
    rows = []
    t = 0
    for _ in range(10):
        t += FRAME_MS
        rows.append(make_row(t, "docked", left_seen="1", right_seen="0"))
    t += 250  # one big tracking gap
    rows.append(make_row(t, "docked", left_seen="1", right_seen="0"))
    for _ in range(10):
        t += FRAME_MS
        rows.append(make_row(t, "docked", left_seen="1", right_seen="0"))

    path = write_csv(tmp_path / "gap.csv", rows)
    result = tr.analyze(tr.load(path))

    assert result["session"]["gap_count"] == 1
    assert abs(result["session"]["max_gap_ms"] - 250) < 1.0


# -- 6. two-hand fps split -----------------------------------------------------------


def test_two_hand_fps_split(tmp_path):
    rows = []
    t = 0
    rows.append(make_row(t, "docked", left_seen="0", right_seen="0"))  # anchor
    for _ in range(50):
        t += 40
        rows.append(make_row(t, "docked", left_seen="1", right_seen="1"))
    for _ in range(50):
        t += 33
        rows.append(make_row(t, "docked", left_seen="0", right_seen="0"))

    path = write_csv(tmp_path / "fps_split.csv", rows)
    result = tr.analyze(tr.load(path))
    fps = result["session"]["fps_by_hands"]

    assert abs(fps[2] - 25.0) < 0.5
    assert abs(fps[0] - 30.3) < 0.5


# -- 7. relaxed pinch, stuck ----------------------------------------------------------


def test_relaxed_pinch_is_a_stuck_release(tmp_path):
    """Fingers relax (pinch opens) but the other three stay loosely curled, so
    fist_ratio sits at 1.3 -- below FIST_OFF the whole time. both_open never
    fires; the draw ends back in held with no shot."""
    rows = []
    closed = {"right_fist": "1.300", "right_pinch": "0.300"}
    opened = {"right_fist": "1.300", "right_pinch": "0.900"}
    drawn_rows = [dict(closed) for _ in range(5)] + [dict(opened) for _ in range(4)]

    add_cycle(rows, 0, drawn_rows=drawn_rows, resolution="held")

    path = write_csv(tmp_path / "relaxed_pinch_stuck.csv", rows)
    result = tr.analyze(tr.load(path))
    pinch = result["release_responsiveness"]["by_grip"]["pinch"]

    assert pinch["draws"] == 1
    assert pinch["unfired"] == 1
    assert pinch["stuck_releases"] == 1
    assert pinch["blocked_open_fraction"] is not None
    assert pinch["blocked_open_fraction"] > 0


# -- 8. fist-grip latency ---------------------------------------------------------------


def test_fist_grip_latency(tmp_path):
    """fist_ratio crosses FIST_OFF 3+1 rows before the fire row -> ~132ms
    latency at 33ms/row (4 row-steps from the first open row to the fire row)."""
    rows = []
    closed = {"right_fist": "0.900", "right_pinch": "0.450"}
    fist_opens = {"right_fist": "2.000", "right_pinch": "0.450"}
    both_open = {"right_fist": "2.000", "right_pinch": "0.900"}
    drawn_rows = (
        [dict(closed) for _ in range(5)]
        + [dict(fist_opens) for _ in range(3)]
        + [dict(both_open)]
    )

    add_cycle(rows, 0, drawn_rows=drawn_rows, resolution="fire")

    path = write_csv(tmp_path / "fist_latency.csv", rows)
    result = tr.analyze(tr.load(path))
    fist = result["release_responsiveness"]["by_grip"]["fist"]

    assert fist["fired"] == 1
    assert fist["latency_ms"] is not None
    assert abs(fist["latency_ms"]["p50"] - 132) < 1.0


# -- 9. draw_grip column overrides inference ---------------------------------------------


def test_draw_grip_column_overrides_inference(tmp_path):
    """fist stays 0.9 the whole draw (would infer "fist"), but draw_grip is
    explicitly "pinch" on the run's first row, which must win."""
    rows = []
    fist_like = {"right_fist": "0.900", "right_pinch": "0.300"}
    drawn_rows = [dict(fist_like, draw_grip="pinch")] + [dict(fist_like) for _ in range(5)]

    add_cycle(rows, 0, drawn_rows=drawn_rows, resolution="fire")

    path = write_csv(tmp_path / "grip_override.csv", rows)
    result = tr.analyze(tr.load(path))
    rr = result["release_responsiveness"]

    assert rr["has_draw_grip_column"] is True
    assert rr["by_grip"]["pinch"]["draws"] == 1
    assert rr["by_grip"]["fist"]["draws"] == 0


# -- 10. release_rule column controls the per-rule split ---------------------------------


def test_release_rule_column_controls_per_rule_split(tmp_path):
    pinch_vals = {"right_fist": "1.900", "right_pinch": "0.250"}

    # WITH release_rule: two pinch-grip draws tagged with different rules.
    rows = []
    t = add_cycle(
        rows, 0,
        drawn_rows=[dict(pinch_vals, release_rule="both_open")] + [dict(pinch_vals) for _ in range(5)],
        resolution="fire",
    )
    add_cycle(
        rows, t,
        drawn_rows=[dict(pinch_vals, release_rule="grip_aware")] + [dict(pinch_vals) for _ in range(5)],
        resolution="fire",
    )
    path = write_csv(tmp_path / "with_rule.csv", rows)
    result = tr.analyze(tr.load(path))
    rr = result["release_responsiveness"]

    assert rr["has_release_rule_column"] is True
    assert "both_open" in rr["by_grip_rule"]["pinch"]
    assert "grip_aware" in rr["by_grip_rule"]["pinch"]
    assert rr["by_grip_rule"]["pinch"]["both_open"]["draws"] == 1
    assert rr["by_grip_rule"]["pinch"]["grip_aware"]["draws"] == 1

    # WITHOUT release_rule (or draw_grip) at all: per-grip stats still work.
    rows2 = []
    add_cycle(
        rows2, 0, columns=REQUIRED_COLUMNS,
        drawn_rows=[dict(pinch_vals) for _ in range(6)], resolution="fire",
    )
    path2 = write_csv(tmp_path / "no_rule.csv", rows2, columns=REQUIRED_COLUMNS)
    result2 = tr.analyze(tr.load(path2))
    rr2 = result2["release_responsiveness"]

    assert rr2["has_release_rule_column"] is False
    assert rr2["by_grip"]["pinch"]["draws"] == 1


# -- 11. malformed row is skipped, not a crash ------------------------------------------


def test_malformed_row_is_skipped_and_counted(tmp_path):
    rows = []
    run_segment(rows, 0, "docked", 0.3, left_seen="1", right_seen="0")
    path = tmp_path / "malformed.csv"
    write_csv(path, rows)

    # Simulate a killed process: a truncated raw line with no comma at all, so
    # csv.DictReader fills every field after the first (t_ms) with None --
    # state ends up None, which is what load() must catch, not int(t_ms).
    with open(path, "a", encoding="utf-8") as f:
        f.write("999999\n")

    loaded = tr.load(str(path))
    assert loaded.skipped_rows == 1

    result = tr.analyze(loaded)
    assert result["session"]["skipped_rows"] == 1

    report = tr.format_report(result)  # must not crash
    assert "skipped" in report.lower()


# -- 12. released->held is a cooldown ----------------------------------------------------


def test_released_to_held_is_a_cooldown(tmp_path):
    rows = []
    t = add_cycle(
        rows, 0,
        drawn_rows=[{"right_fist": "1.900", "right_pinch": "0.250"} for _ in range(3)],
        resolution="fire",
    )
    # Cooldown elapses: released -> held, with no docked in between.
    run_segment(
        rows, t, "held", 0.1,
        bow_side="left", left_seen="1", right_seen="0",
        left_fist="0.900", left_pinch="0.450",
    )

    path = write_csv(tmp_path / "cooldown.csv", rows)
    result = tr.analyze(tr.load(path))

    assert result["transitions"]["cooldowns"] == 1
    assert result["transitions"]["other"] == 0


# -- 13. CSV without the optional columns still works ------------------------------------


def test_csv_without_optional_columns_loads_and_analyzes(tmp_path):
    rows = []
    t = run_segment(
        rows, 0, "docked", 0.3, columns=REQUIRED_COLUMNS,
        left_seen="1", right_seen="1",
        left_fist="2.000", right_fist="2.000",
        left_pinch="0.900", right_pinch="0.900",
    )
    add_cycle(
        rows, t, columns=REQUIRED_COLUMNS,
        drawn_rows=[{"right_fist": "1.900", "right_pinch": "0.250"} for _ in range(6)],
        resolution="fire",
    )

    path = write_csv(tmp_path / "no_optional.csv", rows, columns=REQUIRED_COLUMNS)
    result = tr.analyze(tr.load(path))  # must not raise

    rr = result["release_responsiveness"]
    assert rr["has_draw_grip_column"] is False
    assert rr["has_release_rule_column"] is False
    assert rr["by_grip"]["pinch"]["draws"] == 1

# -- latency ignores stale open readings ------------------------------------------


def test_latency_ignores_an_early_noise_frame(tmp_path):
    """One pinch-open noise frame early in the pull, then a genuine release:
    latency must come from the final open streak, not from the noise."""
    closed = {"right_pinch": "0.200", "right_fist": "2.000", "draw_grip": "pinch"}
    noise = {"right_pinch": "0.900", "right_fist": "2.000", "draw_grip": "pinch"}
    drawn_rows = (
        [dict(closed) for _ in range(3)]
        + [dict(noise)]
        + [dict(closed) for _ in range(20)]
        + [dict(noise)]  # the genuine opening, one debounce frame before the fire
    )
    rows = []
    add_cycle(rows, 0, drawn_rows=drawn_rows, resolution="fire")
    path = write_csv(tmp_path / "noise.csv", rows)
    pinch = tr.analyze(tr.load(path))["release_responsiveness"]["by_grip"]["pinch"]
    assert pinch["fired"] == 1
    assert abs(pinch["latency_ms"]["p50"] - FRAME_MS) < 1.0, pinch["latency_ms"]


# -- integration: the real state machine through the real logger ------------------
#
# These guard the contract between two modules written separately: that what
# input/bow_input.py actually does, as recorded by vision/telemetry.py, is
# what telemetry_report.py assumes. The hand-built-row tests above cannot.

from fletchflow.input.bow_input import BowState, BowStateMachine  # noqa: E402
from fletchflow.input.gestures import GestureFrame, HandGesture  # noqa: E402
from fletchflow.vision.telemetry import TelemetryLogger  # noqa: E402

DOCK = config.DOCK_POS


def _hand(pinch, fist, at=DOCK):
    return HandGesture(
        wrist=at, pinch_point=at, grip_point=at,
        pinch_ratio=pinch, fist_ratio=fist,
        size=config.REFERENCE_HAND_SIZE, palm_size=config.REFERENCE_HAND_SIZE,
    )


BOW_FIST = _hand(0.45, 0.90)        # fist, thumb across the fingers
PINCH_GRIP = _hand(0.20, 2.00)      # pinch with the other fingers extended
RELAXED_OPEN = _hand(0.90, 1.30)    # pinch opened, other fingers still curled
FLAT = _hand(0.90, 2.00)


def _pulled(distance):
    return _hand(0.20, 2.00, at=(DOCK[0], DOCK[1] + distance))


class _Session:
    """Drives BowStateMachine and logs every frame exactly as __main__ does."""

    def __init__(self, path, rule):
        self.machine = BowStateMachine()
        self.machine.set_release_rule(rule)
        self.logger = TelemetryLogger(str(path))
        self.path = path
        self.t = 0

    def step(self, bow, draw, n=1):
        for _ in range(n):
            self.t += FRAME_MS
            frame = GestureFrame(timestamp_ms=self.t, left=bow, right=draw)
            snap = self.machine.update(frame)
            self.logger.log(
                frame, snap, self.machine.bow_side, self.machine.draw_side,
                draw_grip=self.machine.draw_grip,
                release_rule=self.machine.release_rule,
            )

    def grab_and_nock(self):
        self.step(BOW_FIST, None, n=config.FIST_ON_FRAMES + 3)
        self.step(BOW_FIST, PINCH_GRIP, n=config.PINCH_ON_FRAMES)
        assert self.machine.state == BowState.DRAWN
        for k in range(10):
            self.step(BOW_FIST, _pulled(0.02 * (k + 1)))

    def cool_down(self, draw):
        self.step(BOW_FIST, draw, n=config.COOLDOWN_MS // FRAME_MS + 3)

    def report(self):
        self.logger.close()
        return tr.analyze(tr.load(self.path))


def test_real_machine_clean_release_latency_is_one_debounce_frame(tmp_path):
    session = _Session(tmp_path / "clean.csv", "both_open")
    for _ in range(3):
        session.grab_and_nock()
        session.step(BOW_FIST, FLAT, n=config.PINCH_OFF_FRAMES)
        assert session.machine.state == BowState.RELEASED
        session.cool_down(FLAT)
    result = session.report()

    assert result["transitions"]["fires"] == 3
    assert result["transitions"]["other"] == 0
    pinch = result["release_responsiveness"]["by_grip"]["pinch"]
    assert pinch["fired"] == 3
    # PINCH_OFF_FRAMES = 2: the hand opens on frame k (still drawn), fires on k+1
    assert pinch["latency_ms"]["p50"] == (config.PINCH_OFF_FRAMES - 1) * FRAME_MS
    assert pinch["stuck_releases"] == 0


def test_real_machine_relaxed_pinch_sticks_under_both_open(tmp_path):
    session = _Session(tmp_path / "stuck.csv", "both_open")
    session.grab_and_nock()
    session.step(BOW_FIST, RELAXED_OPEN, n=10)  # opened; nothing happens
    assert session.machine.state == BowState.DRAWN
    session.step(BOW_FIST, None, n=config.HAND_LOST_GRACE_MS // FRAME_MS + 2)
    assert session.machine.state == BowState.HELD  # the player gave up
    result = session.report()

    by_rule = result["release_responsiveness"]["by_grip_rule"]["pinch"]["both_open"]
    assert by_rule["fired"] == 0
    assert by_rule["stuck_releases"] == 1
    assert by_rule["blocked_open_fraction"] > 0


def test_real_machine_grip_aware_fires_the_same_relaxed_pinch(tmp_path):
    session = _Session(tmp_path / "aware.csv", "grip_aware")
    session.grab_and_nock()
    session.step(BOW_FIST, RELAXED_OPEN, n=config.PINCH_OFF_FRAMES)
    assert session.machine.state == BowState.RELEASED
    session.cool_down(RELAXED_OPEN)
    result = session.report()

    by_rule = result["release_responsiveness"]["by_grip_rule"]["pinch"]["grip_aware"]
    assert by_rule["fired"] == 1
    assert by_rule["stuck_releases"] == 0
    assert by_rule["latency_ms"]["p50"] == (config.PINCH_OFF_FRAMES - 1) * FRAME_MS


def test_real_machine_late_flat_hand_is_reported_as_a_held_release(tmp_path):
    """Under both_open a relaxed pinch fires only once the whole hand goes flat.
    The report must show that wait as latency, not as a clean release."""
    session = _Session(tmp_path / "late.csv", "both_open")
    session.grab_and_nock()
    session.step(BOW_FIST, RELAXED_OPEN, n=6)  # held back by the AND
    session.step(BOW_FIST, FLAT, n=config.PINCH_OFF_FRAMES)
    assert session.machine.state == BowState.RELEASED
    session.cool_down(FLAT)
    result = session.report()

    pinch = result["release_responsiveness"]["by_grip"]["pinch"]
    assert pinch["fired"] == 1
    # pinch open for 6 relaxed frames, then flat: open -> fire spans 6 + 1 frames
    assert pinch["latency_ms"]["p50"] == (6 + config.PINCH_OFF_FRAMES - 1) * FRAME_MS
    assert pinch["latency_ms"]["p50"] > 70  # past the report's "held by the rule" line
