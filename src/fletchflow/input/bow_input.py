"""The bow state machine. Camera space; the heart of the game feel.

The bow starts DOCKED at config.DOCK_POS ("Make a fist to grab the bow!").
Whichever hand closes a **fist** near it first becomes the bow hand, and the bow
anchors to that hand's palm grip point. The other hand grabs the string with
**either** a pinch or a fist — accepting both means the player does not have to
guess which grip the game wants — and fires when that hand goes flat.

Transition table — thresholds and durations in config.py:

| From     | To       | Condition                                                |
|----------|----------|----------------------------------------------------------|
| DOCKED   | HELD     | fist_ratio < FIST_ON for FIST_ON_FRAMES, grip point       |
|          |          | within GRAB_RADIUS of DOCK_POS -> that hand = bow hand    |
| HELD     | DRAWN    | other hand pinches OR fists (own debounce) within         |
|          |          | STRING_GRAB_RADIUS of the anchor; baselines d0, s0 kept   |
| DRAWN    | RELEASED | release_rule selects the test (config.RELEASE_RULE; G     |
|          |          | cycles it live): "both_open" needs pinch_ratio >           |
|          |          | PINCH_OFF AND fist_ratio > FIST_OFF; "grip_aware" needs    |
|          |          | just the drawn grip's own ratio open. Either way, held     |
|          |          | for PINCH_OFF_FRAMES -> fire                              |
| DRAWN    | HELD     | draw hand lost > HAND_LOST_GRACE_MS (cancel, no fire)     |
| RELEASED | HELD     | COOLDOWN_MS elapsed (DOCKED if the bow was dropped)       |
| HELD/    | DOCKED   | bow hand fist_ratio > FIST_OFF for BOW_DROP_FRAMES, or    |
| DRAWN    |          | bow hand lost > BOW_LOST_MS (drop; never fires)           |

Release is gated by a selectable rule, config.RELEASE_RULE — "both_open" (the
default) or "grip_aware" — which the player can cycle live with G. The AND in
"both_open" is load-bearing, not redundancy: in a tight fist the thumb lies
across the fingers, so pinch_ratio parks around 0.3-0.5 — below PINCH_OFF —
and a release gated on pinch_ratio alone would never fire for a player who
grabbed the string with a fist. Requiring both ratios open means "the hand is
flat", which is true of every release regardless of grip. "grip_aware" tests
only the ratio for the grip that was actually used, because a relaxed pinch
leaves the other fingers loosely curled — fist_ratio then sits near 1.3,
below FIST_OFF, so "both_open" holds it and never releases.

Power is measured in **hand-widths** and in 3D (PLAN.md 4.6.2). The draw hand
moves back toward the face, i.e. mostly in depth, so the old 2D screen distance
measured almost nothing. Depth comes from apparent hand size, since MediaPipe's
per-landmark z is wrist-relative and not comparable between hands:

    lateral_hw = (dist(anchor, draw_point) - d0) / bow_palm
    depth_hw   = CAM_FOCAL_NORM * (1/draw_palm - 1/s0)
    power      = clamp(hypot(lateral_hw, depth_hw) / DRAW_FULL_HW, 0, 1)

Both terms are in hand-widths, which makes them commensurable and independent of
how far the player sits from the camera.

Thresholds are per-instance so calibration can tighten them per player
(apply_calibration); the config values are only the defaults. Nothing here
mutates the config module at runtime.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass

from fletchflow import config
from fletchflow.input.gestures import GestureFrame, HandGesture


class BowState(enum.Enum):
    DOCKED = "docked"
    HELD = "held"
    DRAWN = "drawn"
    RELEASED = "released"


@dataclass(frozen=True)
class BowSnapshot:
    """State machine output for one tracked frame. Still camera space."""

    timestamp_ms: int
    state: BowState
    anchor: tuple[float, float]              # bow grip: dock or bow-hand palm
    draw_point: tuple[float, float] | None   # draw-hand grip while DRAWN
    power: float                             # 0..1 while DRAWN, else 0
    fired_power: float | None                # set only on the release frame
    scale: float = 1.0                       # bow-hand depth scale (EMA-smoothed)
    draw_power_hw: float = 0.0               # raw pull in hand-widths (calibration)
    # M4c (PLAN.md §4.7.10) — filled in by phase 2
    bow_position_m: tuple[float, float, float] | None = None   # smoothed, camera metres
    draw_position_m: tuple[float, float, float] | None = None  # DRAWN only; frozen while lost
    knuckle_dir: tuple[float, float] = (0.0, -1.0)             # bow hand, smoothed
    pull_m: float = 0.0
    render_scale: float = 1.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _hand_scale(hand: HandGesture) -> float:
    """Apparent-size proxy for camera distance, clamped to a sane range."""
    raw = hand.size / config.REFERENCE_HAND_SIZE
    return max(config.DEPTH_SCALE_MIN, min(config.DEPTH_SCALE_MAX, raw))


class BowStateMachine:
    def __init__(self) -> None:
        self._state = BowState.DOCKED
        self._bow_side: str | None = None
        self._draw_side: str | None = None
        self._pinch_frames = {"left": 0, "right": 0}
        self._fist_frames = {"left": 0, "right": 0}
        self._bow_open_frames = 0
        self._draw_open_frames = 0
        self._grab_baseline = 0.0
        self._bow_seen_ms: int | None = None
        self._draw_seen_ms: int | None = None
        self._released_at_ms: int | None = None
        self._anchor: tuple[float, float] = config.DOCK_POS
        self._draw_point: tuple[float, float] | None = None
        self._power_history: deque[float] = deque(maxlen=config.FIRE_POWER_WINDOW)
        self._scale = 1.0
        self._draw_uses_grip = False
        self._pull_hw = 0.0
        self._release_rule = config.RELEASE_RULE

        # palm_size EMAs, in normalized units (~0.11). Distinct from _scale,
        # which is a clamped RATIO — the power formula divides by these, not it.
        self._bow_palm = config.REFERENCE_HAND_SIZE
        self._draw_palm = config.REFERENCE_HAND_SIZE
        self._grab_palm = config.REFERENCE_HAND_SIZE

        # Per-player thresholds; calibration may tighten them.
        self._pinch_on = config.PINCH_ON
        self._pinch_off = config.PINCH_OFF
        self._fist_on = config.FIST_ON
        self._fist_off = config.FIST_OFF
        self._draw_full_hw = config.DRAW_FULL_HW

    @property
    def state(self) -> BowState:
        return self._state

    @property
    def bow_side(self) -> str | None:
        return self._bow_side

    @property
    def draw_side(self) -> str | None:
        return self._draw_side

    @property
    def draw_power_hw(self) -> float:
        """Current pull in hand-widths, before the DRAW_FULL_HW division."""
        return self._pull_hw

    @property
    def release_rule(self) -> str:
        return self._release_rule

    def set_release_rule(self, rule: str) -> None:
        if rule not in config.RELEASE_RULES:
            raise ValueError(f"unknown release rule: {rule!r}")
        self._release_rule = rule

    @property
    def draw_grip(self) -> str:
        """"fist" / "pinch" while a draw hand is assigned (DRAWN, and RELEASED
        during cooldown), else ""."""
        if self._draw_side is None:
            return ""
        return "fist" if self._draw_uses_grip else "pinch"

    def apply_calibration(self, result) -> None:
        """Adopt per-player thresholds measured by input/calibration.py."""
        self._pinch_on = result.pinch_on
        self._pinch_off = result.pinch_off
        self._fist_on = result.fist_on
        self._fist_off = result.fist_off
        self._draw_full_hw = result.draw_full_hw

    def update(self, frame: GestureFrame) -> BowSnapshot:
        now = frame.timestamp_ms
        fired: float | None = None

        if self._state == BowState.DOCKED:
            self._anchor = config.DOCK_POS
            side = self._fist_started_near(frame, config.DOCK_POS, config.GRAB_RADIUS)
            if side is not None:
                self._bow_side = side
                bow = frame.get(side)
                if bow is not None:
                    self._bow_palm = bow.palm_size
                self._to_held(now)

        elif self._state == BowState.HELD:
            if not self._bow_hand_ok(frame, now):
                self._to_docked()
            else:
                other = "right" if self._bow_side == "left" else "left"
                grabbed = self._string_grab_started(
                    frame, self._anchor, config.STRING_GRAB_RADIUS, other
                )
                if grabbed is not None:
                    side, uses_grip = grabbed
                    draw = frame.get(side)
                    self._draw_side = side
                    self._draw_uses_grip = uses_grip
                    self._draw_point = _draw_pos(draw, uses_grip)
                    self._grab_baseline = _dist(self._anchor, self._draw_point)
                    self._grab_palm = draw.palm_size
                    self._draw_palm = draw.palm_size
                    self._draw_open_frames = 0
                    self._draw_seen_ms = now
                    self._pull_hw = 0.0
                    self._power_history.clear()
                    self._power_history.append(0.0)
                    self._state = BowState.DRAWN

        elif self._state == BowState.DRAWN:
            if not self._bow_hand_ok(frame, now):
                self._to_docked()  # dropped mid-draw: no fire
            else:
                draw = frame.get(self._draw_side)
                if draw is not None:
                    self._draw_seen_ms = now
                    self._draw_point = _draw_pos(draw, self._draw_uses_grip)
                    self._draw_palm += config.DRAW_SIZE_SMOOTHING * (
                        draw.palm_size - self._draw_palm
                    )
                    self._power_history.append(self._compute_power())
                    if self._release_open(draw):
                        self._draw_open_frames += 1
                        if self._draw_open_frames >= config.PINCH_OFF_FRAMES:
                            fired = max(self._power_history)
                            self._released_at_ms = now
                            self._state = BowState.RELEASED
                    else:
                        self._draw_open_frames = 0
                elif (
                    self._draw_seen_ms is not None
                    and now - self._draw_seen_ms > config.HAND_LOST_GRACE_MS
                ):
                    self._to_held(now)  # draw cancelled, bow still in hand

        elif self._state == BowState.RELEASED:
            if not self._bow_hand_ok(frame, now):
                self._to_docked()
            elif (
                self._released_at_ms is not None
                and now - self._released_at_ms >= config.COOLDOWN_MS
            ):
                self._to_held(now)

        return BowSnapshot(
            timestamp_ms=now,
            state=self._state,
            anchor=self._anchor,
            draw_point=self._draw_point if self._state == BowState.DRAWN else None,
            power=(
                self._power_history[-1]
                if self._state == BowState.DRAWN and self._power_history
                else 0.0
            ),
            fired_power=fired,
            scale=self._scale,
            draw_power_hw=self._pull_hw if self._state == BowState.DRAWN else 0.0,
        )

    # -- power -----------------------------------------------------------

    def _compute_power(self) -> float:
        """Pull in hand-widths, combining the on-screen and depth components."""
        lateral_hw = (
            _dist(self._anchor, self._draw_point) - self._grab_baseline
        ) / max(self._bow_palm, 1e-6)
        depth_hw = config.CAM_FOCAL_NORM * (
            1.0 / max(self._draw_palm, 1e-6) - 1.0 / max(self._grab_palm, 1e-6)
        )
        depth_hw = min(max(depth_hw, 0.0), config.DEPTH_HW_MAX)
        self._pull_hw = math.hypot(max(lateral_hw, 0.0), depth_hw)
        return min(max(self._pull_hw / max(self._draw_full_hw, 1e-6), 0.0), 1.0)

    # -- release -----------------------------------------------------------

    def _release_open(self, draw: HandGesture) -> bool:
        """Whether the draw hand reads as "open" under the active release rule.

        See the module docstring for why "both_open" ANDs the two ratios, and
        why "grip_aware" testing only the used grip's own ratio can never fire
        later than "both_open".
        """
        pinch_open = draw.pinch_ratio > self._pinch_off
        fist_open = draw.fist_ratio > self._fist_off
        if self._release_rule == "grip_aware":
            return fist_open if self._draw_uses_grip else pinch_open
        return pinch_open and fist_open

    # -- grab detection ---------------------------------------------------

    def _fist_started_near(
        self,
        frame: GestureFrame,
        target: tuple[float, float],
        radius: float,
        only: str | None = None,
    ) -> str | None:
        """Debounced fist whose grip point is within radius of target."""
        for side in ("left", "right"):
            if only is not None and side != only:
                self._fist_frames[side] = 0
                continue
            hand = frame.get(side)
            if (
                hand is not None
                and hand.fist_ratio < self._fist_on
                and _dist(hand.grip_point, target) < radius * _hand_scale(hand)
            ):
                self._fist_frames[side] += 1
                if self._fist_frames[side] >= config.FIST_ON_FRAMES:
                    return side
            else:
                self._fist_frames[side] = 0
        return None

    def _pinch_started_near(
        self,
        frame: GestureFrame,
        target: tuple[float, float],
        radius: float,
        only: str | None = None,
    ) -> str | None:
        """Debounced pinch whose pinch point is within radius of target."""
        for side in ("left", "right"):
            if only is not None and side != only:
                self._pinch_frames[side] = 0
                continue
            hand = frame.get(side)
            if (
                hand is not None
                and hand.pinch_ratio < self._pinch_on
                and _dist(hand.pinch_point, target) < radius * _hand_scale(hand)
            ):
                self._pinch_frames[side] += 1
                if self._pinch_frames[side] >= config.PINCH_ON_FRAMES:
                    return side
            else:
                self._pinch_frames[side] = 0
        return None

    def _string_grab_started(
        self,
        frame: GestureFrame,
        target: tuple[float, float],
        radius: float,
        only: str,
    ) -> tuple[str, bool] | None:
        """Either grip takes the string. Returns (side, uses_grip) or None.

        Both debounces advance every frame, so a hand that starts as a pinch and
        closes into a fist still completes one of them.
        """
        pinched = self._pinch_started_near(frame, target, radius, only=only)
        fisted = self._fist_started_near(frame, target, radius, only=only)
        if pinched is not None:
            return pinched, False
        if fisted is not None:
            return fisted, True
        return None

    # -- bow hand ---------------------------------------------------------

    def _bow_hand_ok(self, frame: GestureFrame, now: int) -> bool:
        """Track the bow hand; False once the bow is dropped or lost."""
        bow = frame.get(self._bow_side)
        if bow is not None:
            self._bow_seen_ms = now
            self._anchor = bow.grip_point
            self._scale += config.DEPTH_SCALE_SMOOTHING * (_hand_scale(bow) - self._scale)
            self._bow_palm += config.DRAW_SIZE_SMOOTHING * (bow.palm_size - self._bow_palm)
            if bow.fist_ratio > self._fist_off:
                self._bow_open_frames += 1
                if self._bow_open_frames >= config.BOW_DROP_FRAMES:
                    return False
            else:
                self._bow_open_frames = 0
            return True
        return not (
            self._bow_seen_ms is not None
            and now - self._bow_seen_ms > config.BOW_LOST_MS
        )

    def _to_held(self, now: int) -> None:
        self._state = BowState.HELD
        self._draw_side = None
        self._draw_point = None
        self._draw_uses_grip = False
        self._bow_seen_ms = now
        self._bow_open_frames = 0
        self._pinch_frames = {"left": 0, "right": 0}
        self._fist_frames = {"left": 0, "right": 0}
        self._power_history.clear()
        self._pull_hw = 0.0

    def _to_docked(self) -> None:
        self._state = BowState.DOCKED
        self._bow_side = None
        self._draw_side = None
        self._draw_point = None
        self._draw_uses_grip = False
        self._anchor = config.DOCK_POS
        self._bow_open_frames = 0
        self._pinch_frames = {"left": 0, "right": 0}
        self._fist_frames = {"left": 0, "right": 0}
        self._power_history.clear()
        self._scale = 1.0
        self._pull_hw = 0.0
        self._bow_palm = config.REFERENCE_HAND_SIZE
        self._draw_palm = config.REFERENCE_HAND_SIZE
        self._grab_palm = config.REFERENCE_HAND_SIZE


def _draw_pos(hand: HandGesture, uses_grip: bool) -> tuple[float, float]:
    """The draw hand's tracked point — decided at grab, never switched mid-draw."""
    return hand.grip_point if uses_grip else hand.pinch_point
