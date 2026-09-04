"""Draws the gallery world in perspective, behind the bow.

Targets and arrows live in metres; everything here is a projection plus a
painter's-algorithm sort (far to near), which is all the depth ordering a
scene of a few discs and arrows needs.
"""

from __future__ import annotations

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


def draw_world(
    surface: pygame.Surface, session: GallerySession, font: pygame.font.Font
) -> None:
    for target in sorted(session.targets, key=lambda t: -t.pos[2]):
        if target.alive:
            _draw_target(surface, target)
    for arrow in session.arrows:
        if arrow.alive:
            _draw_arrow(surface, arrow)
    _draw_hit_feedback(surface, session, font)


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
    # Perspective foreshortening comes free: the projected length shrinks with depth
    pygame.draw.line(
        surface, SHAFT_COLOR, tip[:2], tail[:2], max(1, round(3 * tip[2] / 90.0))
    )
    pygame.draw.line(surface, FLETCH_COLOR, tail[:2], tail[:2], 2)
    pygame.draw.circle(surface, FLETCH_COLOR, tail[:2], max(1, round(tail[2] * 0.02)))


def _draw_hit_feedback(
    surface: pygame.Surface, session: GallerySession, font: pygame.font.Font
) -> None:
    for hit, landed_at in session.recent_hits:
        age = session.elapsed - landed_at
        if age > 0.6:
            continue
        projected = project(hit.point)
        if projected is None:
            continue
        x, y, scale = projected
        t = age / 0.6
        radius = hit.target.radius * scale * (0.3 + t * 1.1)
        pygame.draw.circle(surface, HIT_RING, (x, y), radius, 2)
        label = font.render(f"+{hit.points}", True, HIT_RING)
        surface.blit(label, (x - label.get_width() / 2, y - 24 - t * 30))
