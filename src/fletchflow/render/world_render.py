"""Draws the gallery world in perspective, behind the bow.

Targets and arrows live in metres; everything here is a projection plus a
painter's-algorithm sort (far to near), which is all the depth ordering a
scene of a few discs and arrows needs.
"""

from __future__ import annotations

import math

import pygame

from fletchflow import config
from fletchflow.game.session import GallerySession
from fletchflow.game.world import normalize, project

OUTLINE = (40, 40, 45)
RING_OUTER = (240, 240, 235)
RING_MID = (60, 120, 200)
RING_BULL = (245, 190, 60)
SHAFT_COLOR = (196, 168, 120)
FLETCH_COLOR = (196, 60, 54)
HIT_RING = (255, 235, 120)
TIP_COLOR = (255, 246, 214)
MIN_ARROW_PX = 28  # a receding arrow projects to almost nothing; keep it a streak


def draw_world(
    surface: pygame.Surface,
    session: GallerySession,
    font: pygame.font.Font,
    big_font: pygame.font.Font | None = None,
) -> None:
    for target in sorted(session.targets, key=lambda t: -t.pos[2]):
        if target.alive:
            _draw_target(surface, target)
    for arrow in session.arrows:
        if arrow.alive:
            _draw_arrow(surface, arrow)
    _draw_hit_feedback(surface, session, big_font or font)


def _draw_target(surface: pygame.Surface, target) -> None:
    projected = project(target.pos)
    if projected is None:
        return
    x, y, scale = projected
    radius = target.radius * scale
    if radius < 2:
        return
    w, h = config.WINDOW_SIZE
    if x + radius < 0 or x - radius > w or y + radius < 0 or y - radius > h:
        return

    for fraction, color in ((1.0, RING_OUTER), (0.60, RING_MID), (0.28, RING_BULL)):
        r = radius * fraction
        pygame.draw.circle(surface, color, (x, y), r)
        pygame.draw.circle(surface, OUTLINE, (x, y), r, max(1, round(scale * 0.012)))


def _draw_arrow(surface: pygame.Surface, arrow) -> None:
    direction = normalize(arrow.vel)
    tail_world = tuple(
        arrow.pos[i] - direction[i] * config.ARROW_LENGTH_M for i in range(3)
    )
    tip = project(arrow.pos)
    tail = project(tail_world)
    if tip is None or tail is None:
        return

    tx, ty = tip[0], tip[1]
    dx, dy = tx - tail[0], ty - tail[1]
    length = math.hypot(dx, dy)
    # Flying away from the eye, the arrow foreshortens to a dot within a few
    # metres. Hold a minimum on-screen streak so the shot stays trackable.
    if length < MIN_ARROW_PX:
        if length < 1e-3:
            dx, dy, length = 0.0, 1.0, 1.0  # dead-on: fall back to vertical
        k = MIN_ARROW_PX / length
        tail = (tx - dx * k, ty - dy * k, tail[2])

    width = max(2, round(3 * tip[2] / 90.0))
    pygame.draw.line(surface, SHAFT_COLOR, (tx, ty), tail[:2], width)
    pygame.draw.circle(surface, FLETCH_COLOR, tail[:2], max(2, width))
    pygame.draw.circle(surface, TIP_COLOR, (tx, ty), max(2, width))


def _draw_hit_feedback(
    surface: pygame.Surface, session: GallerySession, font: pygame.font.Font
) -> None:
    for hit, landed_at in session.recent_hits:
        age = session.elapsed - landed_at
        if age > config.HIT_FEEDBACK_S:
            continue
        projected = project(hit.point)
        if projected is None:
            continue
        x, y, scale = projected
        t = age / config.HIT_FEEDBACK_S
        base = max(6.0, hit.target.radius * scale)

        # White flash on impact, gone in the first quarter of the burst
        if t < 0.25:
            alpha = int(230 * (1.0 - t / 0.25))
            size = int(base * 2) + 6
            flash = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(flash, (255, 255, 255, alpha), (size // 2, size // 2), base)
            surface.blit(flash, (x - size / 2, y - size / 2))

        # Two shockwave rings, the second trailing the first
        for delay in (0.0, 0.14):
            rt = (age - delay) / config.HIT_FEEDBACK_S
            if 0.0 <= rt <= 1.0:
                pygame.draw.circle(
                    surface, HIT_RING, (x, y),
                    base * (0.4 + rt * 1.9), max(1, int(6 * (1.0 - rt))),
                )

        label = font.render(f"+{hit.points}", True, HIT_RING)
        label.set_alpha(int(255 * (1.0 - t)))
        surface.blit(label, (x - label.get_width() / 2, y - base - 18 - t * 55))
