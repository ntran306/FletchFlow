"""The bow state machine. Camera space; the heart of the game feel.

The bow starts DOCKED at config.DOCK_POS ("Grab the bow!"). Whichever hand
pinches near it first becomes the bow hand, and the bow anchors to that
hand's thumb+index pinch point. The other hand pinches near the bow to grab
the string; power is the pull distance measured from where the string was
grabbed (so full power needs only a finger-scale pull, config.DRAW_RANGE).

Transition table — thresholds and durations in config.py:

| From     | To       | Condition                                                |
|----------|----------|----------------------------------------------------------|
| DOCKED   | HELD     | pinch_ratio < PINCH_ON for PINCH_ON_FRAMES, pinch point   |
|          |          | within GRAB_RADIUS of DOCK_POS -> that hand = bow hand    |
| HELD     | DRAWN    | other hand pinches (same debounce) within                 |
|          |          | STRING_GRAB_RADIUS of the anchor; baseline d0 recorded    |
| DRAWN    | RELEASED | draw hand ratio > PINCH_OFF for PINCH_OFF_FRAMES -> fire  |
| DRAWN    | HELD     | draw hand lost > HAND_LOST_GRACE_MS (cancel, no fire)     |
| RELEASED | HELD     | COOLDOWN_MS elapsed (DOCKED if the bow was dropped)       |
| HELD/    | DOCKED   | bow hand ratio > PINCH_OFF for BOW_DROP_FRAMES, or bow    |
| DRAWN    |          | hand lost > BOW_LOST_MS (drop; never fires)               |

Power while DRAWN: clamp((dist(anchor, draw_point) - d0) / DRAW_RANGE, 0, 1),
DRAW_RANGE scaled by the bow hand's EMA-smoothed apparent size (self._scale)
so the same physical pull yields the same power at any camera distance.
"""

from __future__ import annotations

import enum
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
    anchor: tuple[float, float]              # bow grip: dock or bow-hand pinch
    draw_point: tuple[float, float] | None   # draw-hand pinch while DRAWN
    power: float                             # 0..1 while DRAWN, else 0
    fired_power: float | None                # set only on the release frame
    scale: float = 1.0                       # bow-hand depth scale (EMA-smoothed)


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
        self._pinch_frames = {"left": 0, "right": 0}  # debounced pinch-starts
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

    @property
    def state(self) -> BowState:
        return self._state

    @property
    def bow_side(self) -> str | None:
        return self._bow_side

    @property
    def draw_side(self) -> str | None:
        return self._draw_side

    def update(self, frame: GestureFrame) -> BowSnapshot:
        now = frame.timestamp_ms
        fired: float | None = None

        if self._state == BowState.DOCKED:
            self._anchor = config.DOCK_POS
            side = self._pinch_started_near(frame, config.DOCK_POS, config.GRAB_RADIUS)
            if side is not None:
                self._bow_side = side
                self._to_held(now)

        elif self._state == BowState.HELD:
            if not self._bow_hand_ok(frame, now):
                self._to_docked()
            else:
                other = "right" if self._bow_side == "left" else "left"
                side = self._pinch_started_near(
                    frame, self._anchor, config.STRING_GRAB_RADIUS, only=other
                )
                if side is not None:
                    draw = frame.get(side)
                    self._draw_side = side
                    self._draw_point = draw.pinch_point
                    self._grab_baseline = _dist(self._anchor, draw.pinch_point)
                    self._draw_open_frames = 0
                    self._draw_seen_ms = now
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
                    self._draw_point = draw.pinch_point
                    pull = _dist(self._anchor, draw.pinch_point) - self._grab_baseline
                    power = max(0.0, min(1.0, pull / (config.DRAW_RANGE * self._scale)))
                    self._power_history.append(power)
                    if draw.pinch_ratio > config.PINCH_OFF:
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
        )

    # -- helpers ---------------------------------------------------------

    def _pinch_started_near(
        self,
        frame: GestureFrame,
        target: tuple[float, float],
        radius: float,
        only: str | None = None,
    ) -> str | None:
        """Debounced pinch whose pinch point is within radius of target.
        Returns the side that completed the debounce, or None."""
        for side in ("left", "right"):
            if only is not None and side != only:
                self._pinch_frames[side] = 0
                continue
            hand = frame.get(side)
            if (
                hand is not None
                and hand.pinch_ratio < config.PINCH_ON
                and _dist(hand.pinch_point, target) < radius * _hand_scale(hand)
            ):
                self._pinch_frames[side] += 1
                if self._pinch_frames[side] >= config.PINCH_ON_FRAMES:
                    return side
            else:
                self._pinch_frames[side] = 0
        return None

    def _bow_hand_ok(self, frame: GestureFrame, now: int) -> bool:
        """Track the bow hand; False once the bow is dropped or lost."""
        bow = frame.get(self._bow_side)
        if bow is not None:
            self._bow_seen_ms = now
            self._anchor = bow.pinch_point
            self._scale += config.DEPTH_SCALE_SMOOTHING * (_hand_scale(bow) - self._scale)
            if bow.pinch_ratio > config.PINCH_OFF:
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
        self._bow_seen_ms = now
        self._bow_open_frames = 0
        self._pinch_frames = {"left": 0, "right": 0}
        self._power_history.clear()

    def _to_docked(self) -> None:
        self._state = BowState.DOCKED
        self._bow_side = None
        self._draw_side = None
        self._draw_point = None
        self._anchor = config.DOCK_POS
        self._bow_open_frames = 0
        self._pinch_frames = {"left": 0, "right": 0}
        self._power_history.clear()
        self._scale = 1.0
