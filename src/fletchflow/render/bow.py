"""Procedurally generated pseudo-3D bow. No asset files.

The bow is drawn from geometry each frame: recurve limbs as tapered bezier
strokes with layered shading (dark base, mid tone, top highlight — a cheap
but convincing cylinder illusion), a wrapped grip, a string that follows the
draw hand, and a nocked arrow. Orientation follows the aim vector; limb flex
and string pull scale with draw power.
"""

from __future__ import annotations

import math

import pygame

from fletchflow import config
from fletchflow.input.bow_input import BowState
from fletchflow.input.mapping import BowPose

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


def _bezier(p0, p1, p2, n: int):
    """Quadratic bezier sample points."""
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


def draw_bow(surface: pygame.Surface, pose: BowPose) -> None:
    if pose.state == BowState.IDLE or pose.anchor is None:
        return

    ax, ay = pose.anchor
    aim = pose.aim
    perp = (-aim[1], aim[0])  # along the bow, tip to tip

    half = config.BOW_SPAN_PX / 2.0
    flex = config.BOW_FLEX_MIN_PX + pose.power * (
        config.BOW_FLEX_MAX_PX - config.BOW_FLEX_MIN_PX
    )

    # Tips: out along the bow axis, pulled toward the player as the bow flexes
    tips = []
    for sign in (1.0, -1.0):
        tx = ax + perp[0] * half * sign - aim[0] * flex
        ty = ay + perp[1] * half * sign - aim[1] * flex
        tips.append((tx, ty))

    # Limbs: riser -> tip, bellied toward the target (out along +aim)
    belly = 26.0 + flex * 0.35
    for sign, tip in zip((1.0, -1.0), tips):
        mid = (
            ax + perp[0] * half * 0.55 * sign + aim[0] * belly,
            ay + perp[1] * half * 0.55 * sign + aim[1] * belly,
        )
        # Enough samples that adjacent stroke circles overlap into solid wood
        # (14 was visibly beaded at this limb length)
        curve = _bezier((ax, ay), mid, tip, 48)
        # Layered strokes, each thinner and offset toward the light (-aim),
        # which reads as a lit wooden cylinder
        _draw_tapered_stroke(surface, curve, LIMB_BASE, 9.0, 4.5)
        lit1 = [(x - aim[0] * 1.5, y - aim[1] * 1.5) for x, y in curve]
        _draw_tapered_stroke(surface, lit1, LIMB_MID, 6.5, 3.0)
        lit2 = [(x - aim[0] * 3.0, y - aim[1] * 3.0) for x, y in curve]
        _draw_tapered_stroke(surface, lit2, LIMB_HIGHLIGHT, 3.0, 1.2)
        # Recurve tip nock
        pygame.draw.circle(surface, TIP_COLOR, tip, 5)
        pygame.draw.circle(surface, LIMB_BASE, tip, 5, width=2)

    # String: straight when relaxed, pulled to the draw point while DRAWN
    if pose.state == BowState.DRAWN and pose.draw_point is not None:
        nock = pose.draw_point
        for offset, color, width in ((1.5, STRING_SHADOW, 3), (0.0, STRING_COLOR, 2)):
            o = (aim[0] * offset, aim[1] * offset)
            pygame.draw.line(
                surface, color,
                (tips[0][0] + o[0], tips[0][1] + o[1]),
                (nock[0] + o[0], nock[1] + o[1]), width,
            )
            pygame.draw.line(
                surface, color,
                (nock[0] + o[0], nock[1] + o[1]),
                (tips[1][0] + o[0], tips[1][1] + o[1]), width,
            )
        _draw_arrow(surface, nock, aim)
    else:
        pygame.draw.line(surface, STRING_SHADOW, tips[0], tips[1], 3)
        pygame.draw.line(surface, STRING_COLOR, tips[0], tips[1], 2)

    # Grip: short wrapped section over the riser, drawn last so it sits on top
    for w, color in ((14, GRIP_DARK), (8, GRIP_LIGHT)):
        pygame.draw.line(
            surface, color,
            (ax - perp[0] * 26, ay - perp[1] * 26),
            (ax + perp[0] * 26, ay + perp[1] * 26), w,
        )


def _draw_arrow(surface: pygame.Surface, nock, aim) -> None:
    length = config.ARROW_LENGTH_PX
    tip = (nock[0] + aim[0] * length, nock[1] + aim[1] * length)
    perp = (-aim[1], aim[0])

    pygame.draw.line(surface, SHAFT_COLOR, nock, tip, 4)

    # Arrowhead: small triangle at the tip
    head_len, head_w = 16.0, 6.0
    base = (tip[0] - aim[0] * head_len, tip[1] - aim[1] * head_len)
    pygame.draw.polygon(
        surface, HEAD_COLOR,
        [
            tip,
            (base[0] + perp[0] * head_w, base[1] + perp[1] * head_w),
            (base[0] - perp[0] * head_w, base[1] - perp[1] * head_w),
        ],
    )

    # Fletching: three angled vanes near the nock
    for i in range(3):
        d = 10 + i * 9
        p0 = (nock[0] + aim[0] * d, nock[1] + aim[1] * d)
        p1 = (
            p0[0] - aim[0] * 8 + perp[0] * 7,
            p0[1] - aim[1] * 8 + perp[1] * 7,
        )
        p2 = (
            p0[0] - aim[0] * 8 - perp[0] * 7,
            p0[1] - aim[1] * 8 - perp[1] * 7,
        )
        pygame.draw.line(surface, FLETCH_COLOR, p0, p1, 3)
        pygame.draw.line(surface, FLETCH_COLOR, p0, p2, 3)
