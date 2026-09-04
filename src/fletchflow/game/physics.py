"""Fixed-timestep arrow flight and target collision. Pure math, no pygame."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fletchflow import config
from fletchflow.game.entities import Arrow, Target
from fletchflow.game.world import Vec3


@dataclass(frozen=True)
class Hit:
    target: Target
    point: Vec3       # where the arrow crossed the target plane
    fraction: float   # distance from centre, as a fraction of the radius
    points: int


def score_for(fraction: float) -> int:
    for bound, points in config.SCORE_RINGS:
        if fraction <= bound:
            return points
    return 0


def step_arrows(arrows: list[Arrow], targets: list[Target], dt: float) -> list[Hit]:
    """Advance one fixed step; return hits registered during it."""
    hits: list[Hit] = []
    for arrow in arrows:
        if not arrow.alive:
            continue
        vx, vy, vz = arrow.vel
        vy += config.GRAVITY_MS2 * dt  # semi-implicit Euler: velocity first
        arrow.vel = (vx, vy, vz)
        arrow.prev_pos = arrow.pos
        arrow.pos = (
            arrow.pos[0] + vx * dt,
            arrow.pos[1] + vy * dt,
            arrow.pos[2] + vz * dt,
        )
        hit = _check_collision(arrow, targets)
        if hit is not None:
            hits.append(hit)
            arrow.alive = False
            hit.target.alive = False
        elif not (config.NEAR_PLANE_M <= arrow.pos[2] <= config.ARROW_MAX_DEPTH_M):
            arrow.alive = False
    return hits


def _check_collision(arrow: Arrow, targets: list[Target]) -> Hit | None:
    """Plane-crossing test: exact at any speed, so arrows cannot tunnel."""
    z0, z1 = arrow.prev_pos[2], arrow.pos[2]
    if z1 <= z0:
        return None
    for target in targets:
        tz = target.pos[2]
        if not (z0 < tz <= z1):
            continue
        t = (tz - z0) / (z1 - z0)
        hx = arrow.prev_pos[0] + (arrow.pos[0] - arrow.prev_pos[0]) * t
        hy = arrow.prev_pos[1] + (arrow.pos[1] - arrow.prev_pos[1]) * t
        distance = math.hypot(hx - target.pos[0], hy - target.pos[1])
        fraction = distance / target.radius
        if fraction <= 1.0:
            return Hit(
                target=target,
                point=(hx, hy, tz),
                fraction=fraction,
                points=score_for(fraction),
            )
    return None
