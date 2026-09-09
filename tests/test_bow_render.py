"""The 2D fallback bow must render at every depth scale.

Regression test for a crash found while measuring the bow's extent for the
sight pin: the grip wrap passed a float pygame line width, so the fallback body
raised TypeError at any scale != 1.0. It was masked because the moderngl body
normally renders instead — but on a machine without GL 3.3 the game died as
soon as the depth-scale EMA left 1.0, which is immediately.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from fletchflow.input.bow_input import BowState  # noqa: E402
from fletchflow.input.mapping import BowPose  # noqa: E402
from fletchflow.render.bow import draw_bow  # noqa: E402

pygame.init()


def test_fallback_body_renders_at_every_scale():
    surface = pygame.Surface((1280, 720))
    anchor = (640.0, 400.0)
    for scale in (0.55, 0.8, 1.0, 1.3, 1.6):
        for power in (0.0, 0.5, 1.0):
            for state, draw_point in (
                (BowState.HELD, None),
                (BowState.DRAWN, (640.0, 520.0)),
            ):
                pose = BowPose(
                    anchor=anchor, draw_point=draw_point, aim=(0.0, -1.0),
                    power=power, state=state, fire=None, scale=scale,
                    sight=(anchor[0], anchor[1] - 110.0),
                )
                surface.fill((0, 0, 0))
                draw_bow(surface, pose, None)  # None -> 2D fallback


def test_fallback_body_actually_paints():
    """Guard against the test above passing because nothing was drawn."""
    surface = pygame.Surface((1280, 720))
    surface.fill((0, 0, 0))
    pose = BowPose(
        anchor=(640.0, 400.0), draw_point=None, aim=(0.0, -1.0), power=0.5,
        state=BowState.HELD, fire=None, scale=1.3, sight=(640.0, 290.0),
    )
    draw_bow(surface, pose, None)
    assert pygame.transform.average_color(surface)[:3] != (0, 0, 0)
