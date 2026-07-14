"""The bow state machine. Camera space; the heart of the game feel.

Transition table (PLAN.md §1) — thresholds and durations in config.py:

| From     | To       | Condition                                              |
|----------|----------|--------------------------------------------------------|
| IDLE     | ARMED    | both hands tracked >= ARM_FRAMES consecutive frames    |
| ARMED    | DRAWN    | a hand's pinch_ratio < PINCH_ON for PINCH_ON_FRAMES    |
| DRAWN    | RELEASED | draw hand's ratio > PINCH_OFF for PINCH_OFF_FRAMES     |
| DRAWN    | ARMED    | either hand lost > HAND_LOST_GRACE_MS (cancel, no fire)|
| RELEASED | ARMED    | COOLDOWN_MS elapsed                                    |
| any      | IDLE     | both hands lost > IDLE_TIMEOUT_MS                      |

Roles are assigned by behaviour, never by handedness labels: whichever hand
pinches becomes the draw hand, sticky until the draw ends. Before any pinch,
the provisional bow hand is the non-dominant one (config.DOMINANT_HAND).
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass

from fletchflow import config
from fletchflow.input.gestures import GestureFrame


class BowState(enum.Enum):
    IDLE = "idle"
    ARMED = "armed"
    DRAWN = "drawn"
    RELEASED = "released"


@dataclass(frozen=True)
class BowSnapshot:
    """State machine output for one tracked frame. Still camera space."""

    timestamp_ms: int
    state: BowState
    bow_wrist: tuple[float, float] | None    # anchor hand
    draw_point: tuple[float, float] | None   # pinch midpoint of the draw hand
    power: float                             # 0..1 while DRAWN, else 0
    fired_power: float | None                # set only on the release frame


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class BowStateMachine:
    def __init__(self) -> None:
        self._state = BowState.IDLE
        self._both_hands_frames = 0
        self._pinch_on_frames = {"left": 0, "right": 0}
        self._pinch_off_frames = 0
        self._draw_side: str | None = None
        self._bow_side: str | None = None
        self._last_both_seen_ms: int | None = None
        self._draw_hands_seen_ms: int | None = None
        self._released_at_ms: int | None = None
        self._power_history: deque[float] = deque(maxlen=config.FIRE_POWER_WINDOW)
        # Last known positions during a draw — the release frame needs them
        # for the fire origin/aim even if a hand blinks out that instant
        self._last_bow_wrist: tuple[float, float] | None = None
        self._last_draw_point: tuple[float, float] | None = None

    @property
    def state(self) -> BowState:
        return self._state

    def update(self, frame: GestureFrame) -> BowSnapshot:
        now = frame.timestamp_ms
        both = frame.left is not None and frame.right is not None
        any_hand = frame.left is not None or frame.right is not None
        if both:
            self._last_both_seen_ms = now

        # any state -> IDLE when both hands are gone long enough
        if not any_hand and self._last_both_seen_ms is not None:
            if now - self._last_both_seen_ms > config.IDLE_TIMEOUT_MS:
                self._reset()
        elif self._last_both_seen_ms is None and not both:
            pass  # never armed yet; stay IDLE

        fired: float | None = None
        if self._state == BowState.IDLE:
            self._both_hands_frames = self._both_hands_frames + 1 if both else 0
            if self._both_hands_frames >= config.ARM_FRAMES:
                self._state = BowState.ARMED

        elif self._state == BowState.ARMED:
            self._update_pinch_counters(frame)
            for side in ("left", "right"):
                other = "right" if side == "left" else "left"
                if (
                    self._pinch_on_frames[side] >= config.PINCH_ON_FRAMES
                    and frame.get(other) is not None
                ):
                    self._draw_side, self._bow_side = side, other
                    self._state = BowState.DRAWN
                    self._pinch_off_frames = 0
                    self._power_history.clear()
                    self._draw_hands_seen_ms = now
                    # Seed power on the transition frame itself, so the first
                    # drawn frame is live and the fire window can't miss it
                    draw_hand, bow_hand = frame.get(side), frame.get(other)
                    self._last_bow_wrist = bow_hand.wrist
                    self._last_draw_point = draw_hand.pinch_point
                    self._power_history.append(
                        self._power(bow_hand.wrist, draw_hand.wrist)
                    )
                    break

        elif self._state == BowState.DRAWN:
            draw = frame.get(self._draw_side)
            bow = frame.get(self._bow_side)
            if draw is not None and bow is not None:
                self._draw_hands_seen_ms = now
                self._last_bow_wrist = bow.wrist
                self._last_draw_point = draw.pinch_point
                self._power_history.append(self._power(bow.wrist, draw.wrist))
                if draw.pinch_ratio > config.PINCH_OFF:
                    self._pinch_off_frames += 1
                    if self._pinch_off_frames >= config.PINCH_OFF_FRAMES:
                        fired = max(self._power_history) if self._power_history else 0.0
                        self._state = BowState.RELEASED
                        self._released_at_ms = now
                else:
                    self._pinch_off_frames = 0
            elif (
                self._draw_hands_seen_ms is not None
                and now - self._draw_hands_seen_ms > config.HAND_LOST_GRACE_MS
            ):
                self._state = BowState.ARMED  # draw cancelled, no fire
                self._end_draw()

        elif self._state == BowState.RELEASED:
            if (
                self._released_at_ms is not None
                and now - self._released_at_ms >= config.COOLDOWN_MS
            ):
                self._state = BowState.ARMED
                self._end_draw()

        return self._snapshot(frame, fired)

    # -- helpers ---------------------------------------------------------

    def _update_pinch_counters(self, frame: GestureFrame) -> None:
        for side in ("left", "right"):
            hand = frame.get(side)
            if hand is not None and hand.pinch_ratio < config.PINCH_ON:
                self._pinch_on_frames[side] += 1
            else:
                self._pinch_on_frames[side] = 0

    @staticmethod
    def _power(bow_wrist, draw_wrist) -> float:
        d = _dist(bow_wrist, draw_wrist)
        span = config.DRAW_MAX - config.DRAW_MIN
        return max(0.0, min(1.0, (d - config.DRAW_MIN) / span))

    def _end_draw(self) -> None:
        self._draw_side = None
        self._bow_side = None
        self._pinch_on_frames = {"left": 0, "right": 0}
        self._pinch_off_frames = 0
        self._power_history.clear()

    def _reset(self) -> None:
        self._state = BowState.IDLE
        self._both_hands_frames = 0
        self._end_draw()

    def _snapshot(self, frame: GestureFrame, fired: float | None) -> BowSnapshot:
        if (
            self._state in (BowState.DRAWN, BowState.RELEASED)
            and self._draw_side is not None
        ):
            bow = frame.get(self._bow_side)
            draw = frame.get(self._draw_side)
            power = self._power_history[-1] if self._power_history else 0.0
            return BowSnapshot(
                timestamp_ms=frame.timestamp_ms,
                state=self._state,
                bow_wrist=bow.wrist if bow else self._last_bow_wrist,
                draw_point=draw.pinch_point if draw else self._last_draw_point,
                power=power if self._state == BowState.DRAWN else 0.0,
                fired_power=fired,
            )

        # ARMED / RELEASED / IDLE: provisional bow hand = non-dominant one
        bow_side = "left" if config.DOMINANT_HAND == "right" else "right"
        bow = frame.get(bow_side) or frame.get(
            "right" if bow_side == "left" else "left"
        )
        return BowSnapshot(
            timestamp_ms=frame.timestamp_ms,
            state=self._state,
            bow_wrist=bow.wrist if bow and self._state != BowState.IDLE else None,
            draw_point=None,
            power=0.0,
            fired_power=fired,
        )
