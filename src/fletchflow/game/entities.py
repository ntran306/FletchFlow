"""World-space game objects and their spawning rules."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from fletchflow import config
from fletchflow.game.world import Vec3, half_width_m, normalize, unproject


@dataclass
class Target:
    pos: Vec3
    radius: float = config.TARGET_RADIUS_M
    alive: bool = True
    respawn_at: float | None = None  # session clock time to reappear


@dataclass
class Arrow:
    pos: Vec3
    vel: Vec3
    alive: bool = True
    prev_pos: Vec3 = field(init=False)

    def __post_init__(self) -> None:
        self.prev_pos = self.pos


def spawn_arrow(aim_screen: tuple[float, float], power: float) -> Arrow:
    """Launch along the view ray through the on-screen aim point.

    Both points come from the same screen position at different depths, so
    they are colinear with the eye — the arrow flies exactly where you point.
    """
    origin = unproject(aim_screen[0], aim_screen[1], config.ARROW_LAUNCH_Z_M)
    sight = unproject(aim_screen[0], aim_screen[1], config.SIGHT_DEPTH_M)
    direction = normalize(
        (sight[0] - origin[0], sight[1] - origin[1], sight[2] - origin[2])
    )
    speed = config.ARROW_SPEED_MIN_MS + power * (
        config.ARROW_SPEED_MAX_MS - config.ARROW_SPEED_MIN_MS
    )
    return Arrow(
        pos=origin,
        vel=(direction[0] * speed, direction[1] * speed, direction[2] * speed),
    )


def random_target_pos(z: float, rng: random.Random) -> Vec3:
    limit = max(0.0, half_width_m(z) * config.TARGET_X_MARGIN - config.TARGET_RADIUS_M)
    return (
        rng.uniform(-limit, limit),
        rng.uniform(*config.TARGET_Y_RANGE_M),
        z,
    )


def spawn_targets(rng: random.Random) -> list[Target]:
    return [Target(pos=random_target_pos(z, rng)) for z in config.TARGET_DEPTHS_M]
