"""Per-player measurement: gesture thresholds, draw range, and the sight zero.

This is the **measurement** half of calibration (PLAN.md 4.6.4). It replaces
constants that were derived from hand geometry rather than from a recording —
FIST_ON/FIST_OFF, the pinch pair, and DRAW_FULL_M — with numbers taken from the
player in front of the camera. The 5-point affine **aim mapping** from the
original M6 plan is deliberately not here; M4c's step 5 below is a simpler
offset-only zero of the 3D sight (PLAN.md §4.7.9), not that fit.

Five steps, ~11 s total. The player opens both hands, closes them, pinches,
draws the bow as far as is comfortable and holds, then draws again and holds
while aiming at a centre dot so the sight can be zeroed.

The draw steps hold rather than shoot, on purpose. An arrow's impact point is
contaminated by power, gravity and release flinch — PLAN.md 4.4 measures a weak
draw at 14 m landing 1.40r wide, which is 1.26 m of error injected purely by how
hard the player pulled. Sampling the held pose measures the one thing we want.

Lives in input/ rather than game/ because it reads camera-space gesture ratios
and metric hand positions; game/ is supposed to see nothing but BowPose.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from fletchflow import config
from fletchflow.input.bow_input import BowState
from fletchflow.input.mapping import aim_angles

PROMPTS = (
    "Open both hands flat",
    "Close both hands into fists",
    "Pinch thumb and finger together",
    "Grab the bow and draw as far as is comfortable — hold",
    "Draw and aim at the centre dot — hold",
)


@dataclass(frozen=True)
class CalibrationResult:
    fist_on: float
    fist_off: float
    pinch_on: float
    pinch_off: float
    draw_full_m: float        # metres of pull for full power (PLAN.md §4.7.6)
    ok: bool
    failed_step: int | None   # 1-5, or None
    message: str
    aim_yaw0_deg: float = 0.0     # M4c §4.7.9: sight zero offsets, filled in by phase 2
    aim_pitch0_deg: float = 0.0

    @classmethod
    def defaults(cls, failed_step: int | None = None, message: str = "") -> "CalibrationResult":
        """Config defaults, used whole whenever a step fails.

        Never return a partly-calibrated mix — the caller applies the result
        unconditionally, so a half-measured set would be worse than none.
        """
        return cls(
            fist_on=config.FIST_ON,
            fist_off=config.FIST_OFF,
            pinch_on=config.PINCH_ON,
            pinch_off=config.PINCH_OFF,
            draw_full_m=config.DRAW_FULL_M,
            ok=failed_step is None,
            failed_step=failed_step,
            message=message,
        )


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; avoids a numpy import for four numbers."""
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _thresholds(open_med: float, closed_med: float) -> tuple[float, float]:
    """Hysteresis pair placed proportionally inside the player's own span."""
    span = open_med - closed_med
    on = open_med - config.CALIB_ON_FRACTION * span
    off = open_med - config.CALIB_OFF_FRACTION * span
    return on, off


def _aim_zero(
    yaws_rad: list[float], pitches_rad: list[float]
) -> tuple[float, float, str | None]:
    """Step 5 (PLAN.md §4.7.9): medians of the raw (yaw, pitch) samples, in
    degrees, become aim_yaw0_deg / aim_pitch0_deg. Rejected — 0.0, 0.0, and a
    message — if either angle's median absolute deviation exceeds
    CALIB_AIM_MAX_MAD_DEG, i.e. the aim wandered rather than holding steady on
    the dot. The caller still discards these zeros and keeps every field at
    its config default on rejection; they are never returned as a result.
    """
    yaws_deg = [math.degrees(y) for y in yaws_rad]
    pitches_deg = [math.degrees(p) for p in pitches_rad]
    yaw0 = statistics.median(yaws_deg)
    pitch0 = statistics.median(pitches_deg)
    yaw_mad = statistics.median(abs(v - yaw0) for v in yaws_deg)
    pitch_mad = statistics.median(abs(v - pitch0) for v in pitches_deg)
    if yaw_mad > config.CALIB_AIM_MAX_MAD_DEG or pitch_mad > config.CALIB_AIM_MAX_MAD_DEG:
        return 0.0, 0.0, "Aim was too unsteady while drawn — keeping defaults"
    return yaw0, pitch0, None


class Calibrator:
    """Drive with one call per tracked frame; returns the result once, at the end."""

    def __init__(self, clock_ms: int) -> None:
        self._start_ms = clock_ms
        self._now_ms = clock_ms
        self._finished = False
        self._open_fist: list[float] = []
        self._open_pinch: list[float] = []
        self._closed_fist: list[float] = []
        self._closed_pinch: list[float] = []
        self._draw_m: list[float] = []
        # Step 5 (M4c, PLAN.md §4.7.9): raw pre-gain, pre-zero (yaw, pitch) in
        # radians, sampled from aim_angles() while drawn and aiming at the dot.
        self._aim_yaw: list[float] = []
        self._aim_pitch: list[float] = []
        bounds, total = [], 0.0
        for seconds in config.CALIB_STEP_S:
            total += seconds
            bounds.append(total)
        self._bounds = bounds
        self._total_s = total

    # -- progress ---------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return (self._now_ms - self._start_ms) / 1000.0

    @property
    def step(self) -> int:
        """1-5 while running, 6 once finished."""
        elapsed = self.elapsed
        for i, bound in enumerate(self._bounds):
            if elapsed < bound:
                return i + 1
        return len(self._bounds) + 1

    @property
    def prompt(self) -> str:
        index = self.step - 1
        return PROMPTS[index] if index < len(PROMPTS) else "Done"

    @property
    def seconds_left(self) -> float:
        index = self.step - 1
        if index >= len(self._bounds):
            return 0.0
        return max(0.0, self._bounds[index] - self.elapsed)

    @property
    def total_seconds(self) -> float:
        return self._total_s

    @property
    def finished(self) -> bool:
        return self._finished

    # -- sampling ---------------------------------------------------------

    def update(self, gesture_frame, snapshot) -> CalibrationResult | None:
        """Returns None until the routine ends, then the result exactly once."""
        if self._finished:
            return None
        self._now_ms = gesture_frame.timestamp_ms

        step = self.step
        if step == 1:
            self._collect(gesture_frame, self._open_fist, self._open_pinch)
        elif step == 2:
            self._collect(gesture_frame, self._closed_fist, None)
        elif step == 3:
            self._collect(gesture_frame, None, self._closed_pinch)
        elif step == 4:
            if snapshot is not None and snapshot.state == BowState.DRAWN:
                self._draw_m.append(snapshot.pull_m)
        elif step == 5:
            self._collect_aim(snapshot)

        if self.elapsed >= self._total_s:
            self._finished = True
            return self._result()
        return None

    @staticmethod
    def _collect(gesture_frame, fist_bucket, pinch_bucket) -> None:
        for side in ("left", "right"):
            hand = gesture_frame.get(side)
            if hand is None:
                continue
            if fist_bucket is not None and hand.fist_ratio != float("inf"):
                fist_bucket.append(hand.fist_ratio)
            if pinch_bucket is not None and hand.pinch_ratio != float("inf"):
                pinch_bucket.append(hand.pinch_ratio)

    def _collect_aim(self, snapshot) -> None:
        """Step 5: raw (yaw, pitch), only once the draw is far enough past the
        nock for the aim line to mean anything — same floor the live game uses
        (AIM_BASELINE_MIN_M + AIM_BASELINE_RAMP_M is where aim_weight reaches
        1), so the samples match what the player will actually see aimed.
        """
        if (
            snapshot is None
            or snapshot.state != BowState.DRAWN
            or snapshot.bow_position_m is None
            or snapshot.draw_position_m is None
        ):
            return
        yaw, pitch, baseline = aim_angles(snapshot.bow_position_m, snapshot.draw_position_m)
        if baseline >= config.AIM_BASELINE_MIN_M + config.AIM_BASELINE_RAMP_M:
            self._aim_yaw.append(yaw)
            self._aim_pitch.append(pitch)

    # -- result -----------------------------------------------------------

    def _result(self) -> CalibrationResult:
        need = config.CALIB_MIN_SAMPLES
        for samples, step, what in (
            (self._open_fist, 1, "hands open"),
            (self._closed_fist, 2, "fists"),
            (self._closed_pinch, 3, "pinch"),
            (self._draw_m, 4, "draw"),
            (self._aim_yaw, 5, "aim"),
        ):
            if len(samples) < need:
                return CalibrationResult.defaults(
                    failed_step=step,
                    message=f"Not enough tracking during '{what}' — keeping defaults",
                )

        fist_open = statistics.median(self._open_fist)
        fist_closed = statistics.median(self._closed_fist)
        if fist_open - fist_closed < config.CALIB_MIN_SEPARATION:
            return CalibrationResult.defaults(
                failed_step=2,
                message="Fist and open hand read too alike — keeping defaults",
            )

        pinch_open = statistics.median(self._open_pinch)
        pinch_closed = statistics.median(self._closed_pinch)
        if pinch_open - pinch_closed < config.CALIB_MIN_SEPARATION:
            return CalibrationResult.defaults(
                failed_step=3,
                message="Pinch and open hand read too alike — keeping defaults",
            )

        fist_on, fist_off = _thresholds(fist_open, fist_closed)
        pinch_on, pinch_off = _thresholds(pinch_open, pinch_closed)

        low, high = config.CALIB_DRAW_CLAMP_M
        draw_full = min(
            max(_percentile(self._draw_m, config.CALIB_DRAW_PERCENTILE), low), high
        )

        aim_yaw0_deg, aim_pitch0_deg, aim_message = _aim_zero(self._aim_yaw, self._aim_pitch)
        if aim_message is not None:
            return CalibrationResult.defaults(failed_step=5, message=aim_message)

        return CalibrationResult(
            fist_on=fist_on,
            fist_off=fist_off,
            pinch_on=pinch_on,
            pinch_off=pinch_off,
            draw_full_m=draw_full,
            ok=True,
            failed_step=None,
            message="Calibrated",
            aim_yaw0_deg=aim_yaw0_deg,
            aim_pitch0_deg=aim_pitch0_deg,
        )
