"""Camera space -> screen space. The ONLY module where the two spaces meet.

Everything upstream (tracker, gestures, bow_input) thinks in mirrored
normalized [0,1] coordinates; everything downstream (game, render) thinks in
pixels. Capture and window are both 16:9, so the mapping is a plain scale.

This module also places the **sight** (M4b). The reticle used to be the bow
anchor itself, so aiming meant putting your hand literally on the target — high
in the frame, near the edge where tracking drops it. It is now a pin
CROSSHAIR_RISE_PX above the grip, with its own much heavier One Euro filter so
it reads as a steady sighted thing rather than inheriting every hand tremor.
The offset is screen-up, not along -aim: aim is precisely the vector that goes
unstable during a 3D draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState
from fletchflow.vision.smoothing import OneEuroFilter

Vec2 = tuple[float, float]


@dataclass(frozen=True)
class FireEvent:
    origin: Vec2      # px — bow anchor at release
    direction: Vec2   # unit vector, screen space
    power: float      # 0..1


@dataclass(frozen=True)
class BowPose:
    """Everything the game/render layers ever learn about the player."""

    anchor: Vec2 | None       # px — bow hand
    draw_point: Vec2 | None   # px — string hand grip point
    aim: Vec2                 # unit vector; last known when indeterminate
    power: float              # 0..1
    state: BowState
    fire: FireEvent | None    # set only on the release frame
    scale: float = 1.0        # bow-hand depth scale, passed through from BowSnapshot
    sight: Vec2 | None = None # px — the reticle: pin-offset from the anchor, smoothed
    # M4c (PLAN.md §4.7.10) — filled in by phase 2, rendered by phase 3
    bow_forward: tuple[float, float, float] = (0.0, 0.0, 1.0)  # game axes, unit
    bow_up: tuple[float, float, float] = (0.0, -1.0, 0.0)      # game axes, unit, orthogonal to forward
    aim_weight: float = 0.0                                     # crosshair alpha, 0..1
    render_scale: float = 1.0


def _to_screen(p: Vec2) -> Vec2:
    return (p[0] * config.WINDOW_SIZE[0], p[1] * config.WINDOW_SIZE[1])


UP: Vec2 = (0.0, -1.0)


class Mapper:
    def __init__(self) -> None:
        self._last_aim: Vec2 = UP
        self._sight_filter = OneEuroFilter(
            min_cutoff=config.RETICLE_MIN_CUTOFF,
            beta=config.RETICLE_BETA,
            d_cutoff=config.RETICLE_D_CUTOFF,
        )

    def apply_calibration(self, result) -> None:
        """Adopt the sight zero offsets (M4c §4.7.9). Stub until phase 2."""

    def map(self, snap: BowSnapshot) -> BowPose:
        anchor = _to_screen(snap.anchor)
        draw_point = _to_screen(snap.draw_point) if snap.draw_point else None

        if snap.state == BowState.DRAWN and draw_point is not None:
            dx = anchor[0] - draw_point[0]
            dy = anchor[1] - draw_point[1]
            length = math.hypot(dx, dy)
            # A correct 3D draw collapses the on-screen separation, and aim
            # drives bow orientation — without this floor the bow spins.
            if length > config.AIM_MIN_SEPARATION_PX:
                self._last_aim = (dx / length, dy / length)
            aim = self._last_aim
        elif snap.state == BowState.RELEASED:
            aim = self._last_aim  # hold through release so the fire uses it
        else:
            aim = self._last_aim = UP  # docked/held bows point up

        sight = self._sight(anchor, snap)

        fire = None
        if snap.fired_power is not None:
            fire = FireEvent(origin=anchor, direction=aim, power=snap.fired_power)

        return BowPose(
            anchor=anchor,
            draw_point=draw_point,
            aim=aim,
            power=snap.power,
            state=snap.state,
            fire=fire,
            scale=snap.scale,
            sight=sight,
        )

    def _sight(self, anchor: Vec2, snap: BowSnapshot) -> Vec2:
        """The reticle: a pin above the grip, low-passed so it holds still."""
        raw = (anchor[0], anchor[1] - config.CROSSHAIR_RISE_PX * snap.scale)
        if snap.state == BowState.DOCKED:
            # Start fresh on the next grab rather than sweeping in from the dock
            self._sight_filter.reset()
            return raw
        smoothed = self._sight_filter(
            np.array(raw, dtype=np.float64), snap.timestamp_ms / 1000.0
        )
        return (float(smoothed[0]), float(smoothed[1]))
