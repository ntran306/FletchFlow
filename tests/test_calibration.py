"""Scripted runs through the measurement calibration (PLAN.md 4.6.4)."""

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState, BowStateMachine
from fletchflow.input.calibration import Calibrator, CalibrationResult
from fletchflow.input.gestures import GestureFrame, HandGesture

FRAME_MS = 33
DOCK = config.DOCK_POS


def hand(pinch: float, fist: float) -> HandGesture:
    return HandGesture(
        wrist=DOCK, pinch_point=DOCK, grip_point=DOCK,
        pinch_ratio=pinch, fist_ratio=fist,
        size=config.REFERENCE_HAND_SIZE, palm_size=config.REFERENCE_HAND_SIZE,
    )


def drawn_snapshot(t: int, pull_hw: float) -> BowSnapshot:
    return BowSnapshot(
        timestamp_ms=t, state=BowState.DRAWN, anchor=DOCK, draw_point=DOCK,
        power=0.5, fired_power=None, scale=1.0, draw_power_hw=pull_hw,
    )


def run(open_fist=2.0, open_pinch=0.9, closed_fist=0.9, closed_pinch=0.2,
        pull_hw=2.4, drawn=True):
    """Drive a full scripted calibration and return its result."""
    calib = Calibrator(0)
    result = None
    t = 0
    total_frames = int(sum(config.CALIB_STEP_S) * 1000 / FRAME_MS) + 4
    for _ in range(total_frames):
        t += FRAME_MS
        step = calib.step
        if step == 1:
            h = hand(open_pinch, open_fist)
        elif step == 2:
            h = hand(0.45, closed_fist)      # thumb across the fingers
        elif step == 3:
            h = hand(closed_pinch, open_fist)
        else:
            h = hand(closed_pinch, closed_fist)
        frame = GestureFrame(timestamp_ms=t, left=h, right=h)
        snap = drawn_snapshot(t, pull_hw) if (step == 4 and drawn) else None
        out = calib.update(frame, snap)
        if out is not None:
            result = out
    return result


def test_clean_run_produces_tighter_thresholds():
    result = run()
    assert result is not None and result.ok, result
    assert 0.9 < result.fist_on < result.fist_off < 2.0
    assert 0.2 < result.pinch_on < result.pinch_off < 0.9


def test_insufficient_separation_is_rejected():
    """Open 1.5 vs fist 1.3 is a 0.2 span — not enough to threshold reliably."""
    result = run(open_fist=1.5, closed_fist=1.3)
    assert not result.ok
    assert result.failed_step == 2
    assert result.fist_on == config.FIST_ON       # defaults returned whole
    assert result.fist_off == config.FIST_OFF
    assert result.draw_full_hw == config.DRAW_FULL_HW


def test_missing_draw_step_is_rejected():
    result = run(drawn=False)
    assert not result.ok
    assert result.failed_step == 4
    assert result.draw_full_hw == config.DRAW_FULL_HW


def test_draw_range_is_clamped():
    high = run(pull_hw=99.0)
    assert high.ok
    assert high.draw_full_hw == config.CALIB_DRAW_CLAMP[1]

    low = run(pull_hw=0.01)
    assert low.ok
    assert low.draw_full_hw == config.CALIB_DRAW_CLAMP[0]


def test_result_applies_to_the_state_machine():
    """A calibrated threshold should change what counts as a grab."""
    result = run(closed_fist=0.5, open_fist=2.4)   # a wide, clean span
    assert result.ok

    machine = BowStateMachine()
    machine.apply_calibration(result)

    # A hand between the calibrated and default thresholds
    assert result.fist_on < config.FIST_ON, (result.fist_on, config.FIST_ON)
    between = (result.fist_on + config.FIST_ON) / 2.0

    for _ in range(config.FIST_ON_FRAMES + 2):
        machine.update(GestureFrame(
            timestamp_ms=_ * FRAME_MS + FRAME_MS,
            left=hand(0.45, between), right=None,
        ))
    assert machine.state == BowState.DOCKED, "calibrated threshold should be stricter"

    # The same hand grabs under the looser config default
    uncalibrated = BowStateMachine()
    for _ in range(config.FIST_ON_FRAMES + 2):
        uncalibrated.update(GestureFrame(
            timestamp_ms=_ * FRAME_MS + FRAME_MS,
            left=hand(0.45, between), right=None,
        ))
    assert uncalibrated.state == BowState.HELD


def test_calibration_completes_within_budget():
    """Acceptance criterion 6: under 15 s."""
    assert sum(config.CALIB_STEP_S) < 15.0
    calib = Calibrator(0)
    assert calib.total_seconds < 15.0


def test_prompts_advance_through_every_step():
    calib = Calibrator(0)
    seen = []
    t = 0
    for _ in range(int(sum(config.CALIB_STEP_S) * 1000 / FRAME_MS) + 4):
        t += FRAME_MS
        if calib.step not in seen and calib.step <= len(config.CALIB_STEP_S):
            seen.append(calib.step)
        calib.update(
            GestureFrame(timestamp_ms=t, left=hand(0.9, 2.0), right=None), None
        )
    assert seen == [1, 2, 3, 4]


def test_result_is_returned_only_once():
    calib = Calibrator(0)
    t = 0
    results = []
    for _ in range(int(sum(config.CALIB_STEP_S) * 1000 / FRAME_MS) + 20):
        t += FRAME_MS
        out = calib.update(
            GestureFrame(timestamp_ms=t, left=hand(0.9, 2.0), right=None), None
        )
        if out is not None:
            results.append(out)
    assert len(results) == 1


def test_defaults_helper_is_a_complete_set():
    result = CalibrationResult.defaults(failed_step=1, message="x")
    assert not result.ok
    assert result.fist_on == config.FIST_ON
    assert result.pinch_off == config.PINCH_OFF
    assert result.draw_full_hw == config.DRAW_FULL_HW
