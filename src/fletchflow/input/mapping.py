"""Camera space -> screen space. The ONLY module where the two spaces meet.

Everything upstream (tracker, gestures, bow_input) thinks in mirrored
normalized [0,1] coordinates; everything downstream (game, render) thinks in
pixels. Capture and window are both 16:9, so the mapping is a plain scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState

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
    draw_point: Vec2 | None   # px — string hand pinch point
    aim: Vec2                 # unit vector; last known when indeterminate
    power: float              # 0..1
    state: BowState
    fire: FireEvent | None    # set only on the release frame


def _to_screen(p: Vec2) -> Vec2:
    return (p[0] * config.WINDOW_SIZE[0], p[1] * config.WINDOW_SIZE[1])


UP: Vec2 = (0.0, -1.0)


class Mapper:
    def __init__(self) -> None:
        self._last_aim: Vec2 = UP

    def map(self, snap: BowSnapshot) -> BowPose:
        anchor = _to_screen(snap.anchor)
        draw_point = _to_screen(snap.draw_point) if snap.draw_point else None

        if snap.state == BowState.DRAWN and draw_point is not None:
            dx = anchor[0] - draw_point[0]
            dy = anchor[1] - draw_point[1]
            length = math.hypot(dx, dy)
            if length > 1.0:  # degenerate while the hands overlap
                self._last_aim = (dx / length, dy / length)
            aim = self._last_aim
        elif snap.state == BowState.RELEASED:
            aim = self._last_aim  # hold through release so the fire uses it
        else:
            aim = self._last_aim = UP  # docked/held bows point up

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
        )
