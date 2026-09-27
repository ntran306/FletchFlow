"""Bow rendering: shared geometry + 2D body fallback + string/arrow overlay.

The wooden body is drawn by a pluggable renderer — the true-3D moderngl one
(render/bow3d.py) when the GPU cooperates, else the layered-stroke 2D
fallback here. The string and arrow are always plain 2D draws on top: they
must attach to the player's actual on-screen pinch points, so compositing
them in screen space keeps them glued to the hands.

Geometry convention shared with bow3d.py: the bow spans BOW_SPAN_PX along
the axis perpendicular to aim; limbs belly out along +aim and the tips pull
back along -aim by flex (power-scaled).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from fletchflow import config
from fletchflow.input.bow_input import BowState
from fletchflow.input.mapping import BowPose
from fletchflow.render import bow_model

# Wood tones, dark to light (layered for the cylinder illusion)
LIMB_BASE = (62, 39, 22)
LIMB_MID = (110, 72, 38)
LIMB_HIGHLIGHT = (168, 118, 66)
GRIP_DARK = (38, 30, 26)
GRIP_LIGHT = (72, 58, 48)
TIP_COLOR = (222, 205, 164)
STRING_COLOR = (232, 232, 238)
STRING_SHADOW = (90, 90, 100)
SHAFT_COLOR = (196, 168, 120)
HEAD_COLOR = (200, 204, 212)
FLETCH_COLOR = (196, 60, 54)


@dataclass(frozen=True)
class BowGeometry:
    anchor: tuple[float, float]
    aim: tuple[float, float]
    perp: tuple[float, float]
    flex: float
    belly: float
    tips: tuple[tuple[float, float], tuple[float, float]]
    scale: float


def compute_geometry(pose: BowPose) -> BowGeometry:
    ax, ay = pose.anchor
    aim = pose.aim
    perp = (-aim[1], aim[0])
    half = config.BOW_SPAN_PX / 2.0 * pose.scale
    flex_unscaled = config.BOW_FLEX_MIN_PX + pose.power * (
        config.BOW_FLEX_MAX_PX - config.BOW_FLEX_MIN_PX
    )
    # Scale each length exactly once — deriving belly from the already-scaled
    # flex would make it grow quadratically, and bow3d divides scale back out
    flex = flex_unscaled * pose.scale
    belly = (26.0 + flex_unscaled * 0.35) * pose.scale
    tips = tuple(
        (
            ax + perp[0] * half * sign - aim[0] * flex,
            ay + perp[1] * half * sign - aim[1] * flex,
        )
        for sign in (1.0, -1.0)
    )
    return BowGeometry(
        anchor=(ax, ay), aim=aim, perp=perp, flex=flex, belly=belly, tips=tips,
        scale=pose.scale,
    )


def draw_bow(surface: pygame.Surface, pose: BowPose, body_renderer=None,
             asset=None) -> None:
    """body_renderer: optional object with draw(surface, pose, geom).

    `asset`: a fitted BowAsset (M4c). When one is loaded the bow is a real 3D
    model placed from the pose's own 3D frame; otherwise the procedural body
    below runs unchanged, so the game still plays with no asset file present.
    """
    if asset is not None:
        draw_bow_asset(surface, pose, asset)
        return
    geom = compute_geometry(pose)
    if body_renderer is not None:
        body_renderer.draw(surface, pose, geom)
    else:
        _draw_body_2d(surface, geom)
    _draw_string_and_arrow(surface, pose, geom)


# -- 2D fallback body -----------------------------------------------------


def _bezier(p0, p1, p2, n: int):
    points = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        points.append(
            (
                u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
            )
        )
    return points


def _draw_tapered_stroke(surface, points, color, base_width: float, tip_width: float):
    n = len(points) - 1
    for i, (x, y) in enumerate(points):
        r = base_width + (tip_width - base_width) * (i / n)
        pygame.draw.circle(surface, color, (x, y), r)


def _draw_body_2d(surface: pygame.Surface, geom: BowGeometry) -> None:
    ax, ay = geom.anchor
    aim, perp = geom.aim, geom.perp
    scale = geom.scale
    half = config.BOW_SPAN_PX / 2.0 * scale

    for sign, tip in zip((1.0, -1.0), geom.tips):
        mid = (
            ax + perp[0] * half * 0.55 * sign + aim[0] * geom.belly,
            ay + perp[1] * half * 0.55 * sign + aim[1] * geom.belly,
        )
        # Enough samples that adjacent stroke circles overlap into solid wood
        curve = _bezier((ax, ay), mid, tip, 48)
        _draw_tapered_stroke(surface, curve, LIMB_BASE, 9.0 * scale, 4.5 * scale)
        lit1 = [(x - aim[0] * 1.5 * scale, y - aim[1] * 1.5 * scale) for x, y in curve]
        _draw_tapered_stroke(surface, lit1, LIMB_MID, 6.5 * scale, 3.0 * scale)
        lit2 = [(x - aim[0] * 3.0 * scale, y - aim[1] * 3.0 * scale) for x, y in curve]
        _draw_tapered_stroke(surface, lit2, LIMB_HIGHLIGHT, 3.0 * scale, 1.2 * scale)
        pygame.draw.circle(surface, TIP_COLOR, tip, 5 * scale)
        pygame.draw.circle(surface, LIMB_BASE, tip, 5 * scale, width=2)

    # Grip wrap over the riser. pygame requires an int line width — passing
    # w * scale raised TypeError for every scale != 1.0, which meant the 2D
    # fallback body crashed on any machine without GL 3.3, as soon as the
    # depth-scale EMA left 1.0.
    for w, color in ((14, GRIP_DARK), (8, GRIP_LIGHT)):
        pygame.draw.line(
            surface, color,
            (ax - perp[0] * 26 * scale, ay - perp[1] * 26 * scale),
            (ax + perp[0] * 26 * scale, ay + perp[1] * 26 * scale),
            max(1, int(round(w * scale))),
        )


# -- String + arrow (always 2D, glued to the hands) ------------------------


def _draw_string_and_arrow(surface: pygame.Surface, pose: BowPose, geom: BowGeometry):
    tips, aim = geom.tips, geom.aim
    if pose.state == BowState.DRAWN and pose.draw_point is not None:
        nock = pose.draw_point
        for offset, color, width in ((1.5, STRING_SHADOW, 3), (0.0, STRING_COLOR, 2)):
            o = (aim[0] * offset, aim[1] * offset)
            for a, b in ((tips[0], nock), (nock, tips[1])):
                pygame.draw.line(
                    surface, color,
                    (a[0] + o[0], a[1] + o[1]),
                    (b[0] + o[0], b[1] + o[1]), width,
                )
        _draw_arrow(surface, nock, aim, geom.scale)
    else:
        pygame.draw.line(surface, STRING_SHADOW, tips[0], tips[1], 3)
        pygame.draw.line(surface, STRING_COLOR, tips[0], tips[1], 2)


def _draw_arrow(surface: pygame.Surface, nock, aim, scale: float = 1.0) -> None:
    length = config.ARROW_LENGTH_PX * scale
    tip = (nock[0] + aim[0] * length, nock[1] + aim[1] * length)
    perp = (-aim[1], aim[0])

    pygame.draw.line(surface, SHAFT_COLOR, nock, tip, max(2, round(4 * scale)))

    head_len, head_w = 16.0 * scale, 6.0 * scale
    base = (tip[0] - aim[0] * head_len, tip[1] - aim[1] * head_len)
    pygame.draw.polygon(
        surface, HEAD_COLOR,
        [
            tip,
            (base[0] + perp[0] * head_w, base[1] + perp[1] * head_w),
            (base[0] - perp[0] * head_w, base[1] - perp[1] * head_w),
        ],
    )

    for i in range(3):
        d = (10 + i * 9) * scale
        p0 = (nock[0] + aim[0] * d, nock[1] + aim[1] * d)
        for side in (1.0, -1.0):
            pygame.draw.line(
                surface, FLETCH_COLOR, p0,
                (
                    p0[0] - aim[0] * 8 * scale + perp[0] * 7 * scale * side,
                    p0[1] - aim[1] * 8 * scale + perp[1] * 7 * scale * side,
                ), 3,
            )


# -- 3D asset body (M4c, PLAN.md §4.7.8) ----------------------------------

# Key light, view space: upper-left, slightly toward the camera. World +y is
# DOWN, so the -y component is what puts the highlight on top of the limb.
_LIGHT = (-0.40, -0.70, -0.59)
_AMBIENT = 0.30

BOW_TINT = (150, 108, 60)
ARROW_TINT = (188, 184, 176)


def draw_bow_asset(surface: pygame.Surface, pose: BowPose, asset) -> None:
    """Paint a loaded 3D asset for this pose: body, string, and nocked arrow.

    Bow and arrow triangles are sorted together rather than part by part, so
    a drawn arrow passing through the riser is occluded by it correctly
    instead of always landing on top or always behind.

    The string is drawn last and unsorted. It is a line, not geometry, and
    its nock is pulled toward the archer — i.e. toward the camera — so it is
    in front of the body wherever the two meet.
    """
    placed = bow_model.place_bow(pose, asset)

    parts = [(placed.bow, BOW_TINT)]
    if placed.arrow is not None:
        parts.append((placed.arrow, ARROW_TINT))

    tris = np.concatenate([p.tris_px for p, _ in parts])
    depth = np.concatenate([p.depth_m for p, _ in parts])
    normals = np.concatenate([p.normals for p, _ in parts])
    tint = np.concatenate([
        np.tile(np.asarray(c, dtype=np.float32), (len(p.tris_px), 1))
        for p, c in parts
    ])

    light = np.asarray(_LIGHT, dtype=np.float32)
    lambert = np.clip(normals @ light, 0.0, 1.0)
    shade = (_AMBIENT + (1.0 - _AMBIENT) * lambert)[:, None]
    colors = np.clip(tint * shade, 0, 255).astype(np.uint8)

    for k in np.argsort(-depth):            # painter's algorithm, far first
        pygame.draw.polygon(
            surface, tuple(int(v) for v in colors[k]),
            [(float(x), float(y)) for x, y in tris[k]],
        )

    _draw_string_3d(surface, pose, placed)


def _draw_string_3d(surface: pygame.Surface, pose: BowPose, placed) -> None:
    top, bottom = placed.tips_px
    if pose.state == BowState.DRAWN:
        segments = ((top, placed.nock_px), (placed.nock_px, bottom))
    else:
        segments = ((top, bottom),)
    for color, width in ((STRING_SHADOW, 4), (STRING_COLOR, 2)):
        for a, b in segments:
            pygame.draw.line(surface, color, a, b, width)
