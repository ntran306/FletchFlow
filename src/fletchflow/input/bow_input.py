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

Release is gated by a selectable rule, config.RELEASE_RULE — "grip_aware" (the
default since M4c; playtest telemetry showed pinch-grip fist_ratio during draws
median 1.16, so "both_open" left 12.8% of drawn frames blocked-open) or
"both_open" — which the player can cycle live with G. The AND in "both_open"
is load-bearing, not redundancy: in a tight fist the thumb lies across the
fingers, so pinch_ratio parks around 0.3-0.5 — below PINCH_OFF — and a release
gated on pinch_ratio alone would never fire for a player who grabbed the
string with a fist. Requiring both ratios open means "the hand is flat", which
is true of every release regardless of grip. "grip_aware" tests only the ratio
for the grip that was actually used, because a relaxed pinch leaves the other
fingers loosely curled — fist_ratio then sits near 1.3, below FIST_OFF, so
"both_open" holds it and never releases.

M4c robustness (PLAN.md §4.7.1/§4.7.7), from playtest telemetry: grace and
drop windows grew (HAND_LOST_GRACE_MS, BOW_LOST_MS, BOW_DROP_FRAMES) so a 3D
draw and a held bow both survive brief occlusion; power history, draw_point
and the new draw_position_m all hold their last value while the draw hand is
lost within its grace. A fist_ratio reading above FIST_RATIO_GLITCH is treated
as a tracking-glitch spike (playtest saw 9.44/5.24/4.12), not a real open
hand: it advances neither the bow-drop counter nor the release-open debounce,
for either hand, under either release rule.

M4c also derives metric 3D positions (PLAN.md §4.7.1/§4.7.10): each tracked
role (bow hand, draw hand) gets its own depth from the nearest accepted
Procrustes pose fit (input/hand_pose.py), One Euro-smoothed here, and
BowSnapshot exposes bow_position_m / draw_position_m / knuckle_dir /
render_scale for input/mapping.py's 3D aim. Draw power itself is unchanged and
still 2D/hand-widths, per PLAN.md — the 3D draw range is a later phase.

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

import numpy as np

from fletchflow import config
from fletchflow.input.gestures import GestureFrame, HandGesture
from fletchflow.vision.smoothing import OneEuroFilter

# Rough typical wrist-to-middle-MCP length, metres. Only seeds a role's depth
# filter before its first accepted Procrustes fit (input/hand_pose.py) lands —
# after that, real metric depth takes over. PLAN.md §4.7.1/4.7.10.
FALLBACK_PALM_M = 0.09


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


def _hand_position_m(
    point: tuple[float, float], depth: float
) -> tuple[float, float, float]:
    """Camera-metric (X, Y, Z) for a 2D normalized point at a known depth.

    Back-projection through the camera pinhole. The point is converted to
    pixels first (x by width, y by height), so a single focal length in pixels
    applies to both axes — this is what keeps the result isotropic, unlike the
    mixed-unit distances that made the old depth proxy rotation-sensitive
    (PLAN.md §4.7.1/§4.7.2).
    """
    W, H = config.CAPTURE_SIZE
    f_px = config.CAM_FOCAL_NORM * W
    u, v = point[0] * W, point[1] * H
    x = (u - W / 2.0) * depth / f_px
    y = (v - H / 2.0) * depth / f_px
    return (float(x), float(y), float(depth))


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

        # M4c metric pose (PLAN.md §4.7.1/4.7.3/4.7.10): one depth One Euro
        # filter per role (bow hand, draw hand), plus the smoothed value each
        # holds between updates so a lost/rejected frame can freeze it instead
        # of reaching into the filter's own private state.
        self._depth_filters = {
            "bow": self._new_depth_filter(),
            "draw": self._new_depth_filter(),
        }
        self._last_depth_m: dict[str, float | None] = {"bow": None, "draw": None}
        self._knuckle_filter = OneEuroFilter(
            min_cutoff=config.BOW_UP_MIN_CUTOFF, beta=config.BOW_UP_BETA, d_cutoff=1.0
        )
        self._knuckle_dir: tuple[float, float] = (0.0, -1.0)

    @staticmethod
    def _new_depth_filter() -> OneEuroFilter:
        return OneEuroFilter(
            min_cutoff=config.POSE_DEPTH_MIN_CUTOFF,
            beta=config.POSE_DEPTH_BETA,
            d_cutoff=config.POSE_DEPTH_D_CUTOFF,
        )

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
                    self._reset_depth("draw")  # each draw estimates depth fresh
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
                    # A reading above FIST_RATIO_GLITCH is a tracking glitch
                    # (playtest saw 9.44/5.24/4.12), not a real reading, under
                    # either release rule — so it neither advances nor resets
                    # the release-open debounce (PLAN.md §4.7.7). Gating here,
                    # in _release_open's caller, covers both_open and
                    # grip_aware alike, rather than duplicating the check
                    # inside _release_open per rule.
                    if draw.fist_ratio <= config.FIST_RATIO_GLITCH:
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

        # M4c metric pose (PLAN.md §4.7.1/4.7.3/4.7.10), unified here rather
        # than in each branch above so the very frame a hand is grabbed (bow
        # or string) already has its role's depth seeded: self._bow_side /
        # self._draw_side reflect this frame's outcome by this point, and
        # frame.get() is a pure lookup, so re-fetching here returns the same
        # HandGesture a branch above may have already used.
        t = now / 1000.0
        bow_hand_now = frame.get(self._bow_side) if self._bow_side else None
        draw_hand_now = frame.get(self._draw_side) if self._draw_side else None

        bow_depth = self._update_depth("bow", bow_hand_now, t)
        self._update_knuckle(bow_hand_now, t)
        draw_depth = self._update_depth("draw", draw_hand_now, t)

        bow_position_m = None
        draw_position_m = None
        knuckle_dir = (0.0, -1.0)
        render_scale = 1.0
        if self._state != BowState.DOCKED:
            knuckle_dir = self._knuckle_dir
            if bow_depth is not None and bow_depth > 1e-6:
                bow_position_m = _hand_position_m(self._anchor, bow_depth)
                render_scale = min(
                    max(config.REFERENCE_BOW_DEPTH_M / bow_depth, config.BOW_SCALE_RANGE[0]),
                    config.BOW_SCALE_RANGE[1],
                )
            if (
                self._state == BowState.DRAWN
                and draw_depth is not None
                and draw_depth > 1e-6
            ):
                draw_position_m = _hand_position_m(self._draw_point, draw_depth)

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
            bow_position_m=bow_position_m,
            draw_position_m=draw_position_m,
            knuckle_dir=knuckle_dir,
            render_scale=render_scale,
        )

    # -- metric pose (M4c) -------------------------------------------------

    def _update_depth(
        self, role: str, hand: HandGesture | None, t: float
    ) -> float | None:
        """Smoothed metric depth for one role ("bow" or "draw").

        A frame with an accepted Procrustes fit (`hand.pose`) feeds the
        role's One Euro filter. A tracked hand with no fit this frame (a
        rejected fit, or no world landmarks) reuses the last smoothed value
        without feeding the filter. Before any fit has ever landed for this
        role, a rough apparent-size fallback seeds it — PLAN.md §4.7.1: "only
        matters before the first good fit." An untracked hand (`hand is
        None`, e.g. lost within its grace) simply holds whatever the role's
        last smoothed value already was.
        """
        if hand is None:
            return self._last_depth_m[role]
        if hand.pose is not None:
            smoothed = self._depth_filters[role](
                np.array([hand.pose.depth_m], dtype=np.float64), t
            )
            self._last_depth_m[role] = float(smoothed[0])
        elif self._last_depth_m[role] is None:
            fallback = config.CAM_FOCAL_NORM * FALLBACK_PALM_M / max(hand.palm_size, 1e-6)
            smoothed = self._depth_filters[role](
                np.array([fallback], dtype=np.float64), t
            )
            self._last_depth_m[role] = float(smoothed[0])
        return self._last_depth_m[role]

    def _update_knuckle(self, hand: HandGesture | None, t: float) -> None:
        """Smooth the bow hand's knuckle_dir; hold the previous reading if the
        hand is untracked this frame, or if smoothing degenerates it near
        zero (PLAN.md §4.7.3)."""
        if hand is None:
            return
        smoothed = self._knuckle_filter(np.array(hand.knuckle_dir, dtype=np.float64), t)
        norm = math.hypot(float(smoothed[0]), float(smoothed[1]))
        if norm < 1e-6:
            return
        self._knuckle_dir = (float(smoothed[0]) / norm, float(smoothed[1]) / norm)

    def _reset_depth(self, role: str) -> None:
        self._depth_filters[role].reset()
        self._last_depth_m[role] = None

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
            # A reading above FIST_RATIO_GLITCH is a tracking glitch (playtest
            # saw 9.44/5.24/4.12), not a genuinely open hand: it must neither
            # advance nor reset the drop counter (PLAN.md §4.7.7).
            if bow.fist_ratio <= config.FIST_RATIO_GLITCH:
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
        self._reset_depth("bow")
        self._reset_depth("draw")
        self._knuckle_filter.reset()
        self._knuckle_dir = (0.0, -1.0)


def _draw_pos(hand: HandGesture, uses_grip: bool) -> tuple[float, float]:
    """The draw hand's tracked point — decided at grab, never switched mid-draw."""
    return hand.grip_point if uses_grip else hand.pinch_point
