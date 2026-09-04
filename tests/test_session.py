import random

from fletchflow import config
from fletchflow.game.entities import Target
from fletchflow.game.session import GallerySession, aim_point
from fletchflow.input.bow_input import BowState
from fletchflow.input.mapping import BowPose, FireEvent

CENTRE = (config.WINDOW_SIZE[0] / 2, config.WINDOW_SIZE[1] / 2)
DT = 1.0 / 60.0


def firing_pose(power=1.0, at=CENTRE):
    """A fresh FireEvent each call — the session dedupes by identity."""
    return BowPose(
        anchor=at, draw_point=None, aim=(0.0, -1.0), power=0.0,
        state=BowState.RELEASED,
        fire=FireEvent(origin=at, direction=(0.0, -1.0), power=power),
    )


def held_pose(at=CENTRE, draw_point=None, power=0.0):
    return BowPose(
        anchor=at, draw_point=draw_point, aim=(0.0, -1.0), power=power,
        state=BowState.DRAWN, fire=None,
    )


def test_aim_point_is_the_bow_hand_by_default():
    assert config.AIM_LEAD_GAIN == 0.0
    assert aim_point(held_pose(at=(500.0, 300.0), draw_point=(400.0, 400.0))) == (500.0, 300.0)
    assert aim_point(None) is None


def test_firing_spends_an_arrow():
    session = GallerySession(random.Random(1))
    session.update(firing_pose(), DT)
    assert session.arrows_left == config.ROUND_ARROWS - 1
    assert len(session.arrows) == 1


def test_one_pose_reused_across_render_ticks_fires_once():
    """The 60 Hz loop reuses a pose between ~30 Hz tracking updates."""
    session = GallerySession(random.Random(1))
    pose = firing_pose()
    for _ in range(5):
        session.update(pose, DT)
    assert session.arrows_left == config.ROUND_ARROWS - 1


def test_hitting_a_target_scores_and_respawns_it():
    session = GallerySession(random.Random(2))
    target = Target(pos=(0.0, 0.0, 6.0))
    session.targets = [target]
    session.update(firing_pose(power=1.0), DT)
    for _ in range(30):
        session.update(None, DT)
    assert session.score == 10
    assert not target.alive

    for _ in range(int(config.TARGET_RESPAWN_S / DT) + 2):
        session.update(None, DT)
    assert target.alive  # a fresh target returns in the same depth band
    assert target.pos[2] == 6.0


def test_round_ends_once_every_arrow_has_landed():
    session = GallerySession(random.Random(3))
    for _ in range(config.ROUND_ARROWS):
        session.update(firing_pose(), DT)
    assert session.arrows_left == 0
    assert not session.finished  # the last arrows are still in flight

    for _ in range(240):
        session.update(None, DT)
    assert session.finished


def test_no_arrows_beyond_the_round_limit():
    session = GallerySession(random.Random(4))
    for _ in range(config.ROUND_ARROWS + 5):
        session.update(firing_pose(), DT)
    assert session.arrows_left == 0
