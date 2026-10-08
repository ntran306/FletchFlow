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

# -- target palette ---------------------------------------------------------
# Three saturated fills (white / primary blue / primary yellow) behind a hard
# black outline is the default clip-art target, and it read as one. This is
# the same scoring geometry in a restrained palette: a warm off-white face,
# charcoal rules, and a single accent carrying the bull.
#
# Contrast is the constraint, not taste: the gallery composites over a live
# camera feed of somebody's room, so a target meets a dark wall, a bright
# window and clutter in the same session. The off-white face carries the dark
# backgrounds; the soft dark halo carries the bright ones. Either way there is
# always one high-contrast edge.
TARGET_FACE = (238, 233, 224)     # warm off-white, not pure white
TARGET_INK = (34, 37, 43)         # charcoal rules
TARGET_ACCENT = (226, 86, 62)     # the bull, and the only saturated colour here
TARGET_FACE_ALPHA = 236           # a hint of the room through the face
TARGET_HALO = (16, 17, 21)
TARGET_HALO_ALPHA = 92            # at the face edge, fading out
TARGET_HALO_SPREAD = 0.16         # how far past the face the halo reaches
TARGET_HALO_STEPS = 5
TARGET_RULE_FRACTION = 0.60       # the 5-point line; SCORE_RINGS' middle radius
TARGET_BULL_FRACTION = 0.28       # ...and its inner one
TARGET_STROKE = 0.022             # rule width as a fraction of the face radius

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

    _paint_target_face(surface, x, y, radius)


def _paint_target_face(surface: pygame.Surface, x: float, y: float, radius: float) -> None:
    """Halo, face, one rule and the bull, composited on their own layer.

    Drawn onto an SRCALPHA layer rather than straight to the screen because
    the face is slightly translucent and the halo fades: both need alpha, and
    pygame.draw has none without a surface to draw into.

    The halo is a stack of rings, not filled circles. Filled ones would
    composite over each other and the alpha would pile up into a hard dark
    disc instead of a fade.
    """
    pad = radius * TARGET_HALO_SPREAD + 4.0
    size = int((radius + pad) * 2) + 2
    layer = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2.0
    stroke = max(1, round(radius * TARGET_STROKE))

    for k in range(TARGET_HALO_STEPS, 0, -1):
        t = k / TARGET_HALO_STEPS
        alpha = int(TARGET_HALO_ALPHA * (1.0 - t) ** 1.6)
        band = max(1, round(pad / TARGET_HALO_STEPS) + 1)
        pygame.draw.circle(layer, (*TARGET_HALO, alpha), (c, c),
                           radius + pad * t, band)

    pygame.draw.circle(layer, (*TARGET_FACE, TARGET_FACE_ALPHA), (c, c), radius)
    pygame.draw.circle(layer, TARGET_INK, (c, c), radius, stroke)
    pygame.draw.circle(layer, TARGET_INK, (c, c), radius * TARGET_RULE_FRACTION, stroke)
    pygame.draw.circle(layer, TARGET_ACCENT, (c, c), radius * TARGET_BULL_FRACTION)

    surface.blit(layer, (x - c, y - c))


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
