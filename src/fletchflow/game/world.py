"""Perspective projection: the world <-> screen boundary for the gallery.

World space is metres with the eye at the origin: +x right, +y DOWN (matching
screen convention, so no sign flips downstream), +z INTO the screen. A pinhole
camera of focal length config.FOCAL_PX maps it to pixels.

Pure math, no pygame — so physics and scoring stay unit-testable.
"""

from __future__ import annotations

import math

from fletchflow import config

Vec3 = tuple[float, float, float]


def _center() -> tuple[float, float]:
    return config.WINDOW_SIZE[0] / 2.0, config.WINDOW_SIZE[1] / 2.0


def project(p: Vec3) -> tuple[float, float, float] | None:
    """World point -> (screen_x, screen_y, scale), or None if behind the eye.

    `scale` is px-per-metre at that depth: multiply a world radius by it to get
    the on-screen radius. This is the whole depth illusion in one number.
    """
    x, y, z = p
    if z < config.NEAR_PLANE_M:
        return None
    cx, cy = _center()
    s = config.FOCAL_PX / z
    return cx + x * s, cy + y * s, s


def unproject(sx: float, sy: float, z: float) -> Vec3:
    """Screen point -> the world point at depth z on that view ray."""
    cx, cy = _center()
    s = config.FOCAL_PX / z
    return (sx - cx) / s, (sy - cy) / s, z


def half_width_m(z: float) -> float:
    """Half the world width visible at depth z."""
    return (config.WINDOW_SIZE[0] / 2.0) * z / config.FOCAL_PX


def normalize(v: Vec3) -> Vec3:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-9:
        return 0.0, 0.0, 1.0
    return v[0] / length, v[1] / length, v[2] / length
