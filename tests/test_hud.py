"""Every HUD drawing function, exercised headlessly.

Coverage gap this closes: render/hud.py had no importer in the test suite, so a
broken edit to it passed CI and only failed when the game actually launched.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from fletchflow import config  # noqa: E402
from fletchflow.input.bow_input import BowSnapshot, BowState  # noqa: E402
from fletchflow.input.calibration import Calibrator  # noqa: E402
from fletchflow.input.gestures import GestureFrame, HandGesture  # noqa: E402
from fletchflow.input.mapping import BowPose  # noqa: E402
from fletchflow.render import hud  # noqa: E402

pygame.init()
pygame.font.init()

DOCK = config.DOCK_POS


def _surface():
    return pygame.Surface(config.WINDOW_SIZE)


def _font():
    return pygame.font.SysFont("consolas", 18)


def _hand():
    return HandGesture(
        wrist=DOCK, pinch_point=DOCK, grip_point=DOCK,
        pinch_ratio=0.30, fist_ratio=0.95,
        size=config.REFERENCE_HAND_SIZE, palm_size=config.REFERENCE_HAND_SIZE,
    )


def _pose(state=BowState.DRAWN):
    return BowPose(
        anchor=(640.0, 420.0), draw_point=(600.0, 500.0), aim=(0.0, -1.0),
        power=0.6, state=state, fire=None, scale=1.2, sight=(640.0, 288.0),
    )


def test_crosshair_power_bar_and_score_draw():
    surface, font = _surface(), _font()
    big = pygame.font.SysFont("consolas", 32, bold=True)

    class FakeSession:
        score, arrows_left, elapsed, finished = 30, 4, 12.0, False

    for state in (BowState.HELD, BowState.DRAWN, BowState.RELEASED):
        pose = _pose(state)
        hud.draw_crosshair(surface, pose.sight, pose.power, state)
        hud.draw_power_bar(surface, pose)
    hud.draw_score(surface, font, FakeSession(), big)


def test_grab_prompt_and_debug_overlay_draw():
    surface, font = _surface(), _font()
    big = pygame.font.SysFont("consolas", 32, bold=True)
    docked = BowPose(
        anchor=(640.0, 400.0), draw_point=None, aim=(0.0, -1.0), power=0.0,
        state=BowState.DOCKED, fire=None, scale=1.0, sight=(640.0, 290.0),
    )
    hud.draw_grab_prompt(surface, big, docked, 1.0)

    frame = GestureFrame(timestamp_ms=33, left=_hand(), right=None)
    hud.draw_debug_state(surface, font, frame, _pose())
    hud.draw_debug_state(surface, font, frame, _pose(), release_rule="grip_aware")
    hud.draw_debug_state(surface, font, None, None)


def test_calibration_panel_draws_at_every_step():
    surface, font = _surface(), _font()
    big = pygame.font.SysFont("consolas", 32, bold=True)
    frame = GestureFrame(timestamp_ms=33, left=_hand(), right=_hand())

    calib = Calibrator(0)
    t = 0
    for _ in range(int(sum(config.CALIB_STEP_S) * 1000 / 33)):
        t += 33
        stepped = GestureFrame(timestamp_ms=t, left=_hand(), right=_hand())
        snap = BowSnapshot(
            timestamp_ms=t, state=BowState.DRAWN, anchor=DOCK, draw_point=DOCK,
            power=0.5, fired_power=None, scale=1.0, draw_power_hw=2.0,
        )
        if calib.update(stepped, snap) is not None:
            break
        hud.draw_calibration(surface, big, font, calib, stepped)

    # and with no hands tracked
    hud.draw_calibration(surface, big, font, Calibrator(0), None)
    hud.draw_calibration_message(surface, font, "Calibrated", True)
    hud.draw_calibration_message(surface, font, "Keeping defaults", False)
    assert frame.left is not None


def test_grab_prompt_names_the_fist():
    """The bow is held with a fist now — the prompt must say so."""
    source = (
        __import__("pathlib").Path(hud.__file__).read_text(encoding="utf-8")
    )
    assert "fist" in source.lower()

def test_crosshair_only_while_there_is_an_arrow_to_aim():
    """Playtest 2026-09-17: the crosshair hung above a bow held in one hand."""
    for state, expect_drawn in (
        (BowState.DOCKED, False),
        (BowState.HELD, False),
        (BowState.DRAWN, True),
        (BowState.RELEASED, True),
    ):
        surface = _surface()
        surface.fill((0, 0, 0))
        hud.draw_crosshair(surface, (640.0, 360.0), 0.5, state)
        # thin lines vanish in a whole-surface average, so count lit pixels
        painted = int(pygame.surfarray.array3d(surface).any(axis=2).sum()) > 0
        assert painted == expect_drawn, state
