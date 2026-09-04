from fletchflow import config
from fletchflow.game.entities import Arrow, Target, spawn_arrow
from fletchflow.game.physics import score_for, step_arrows

CENTRE = (config.WINDOW_SIZE[0] / 2, config.WINDOW_SIZE[1] / 2)


def run(arrow, targets, steps=600):
    hits = []
    for _ in range(steps):
        hits += step_arrows([arrow], targets, config.PHYSICS_DT)
        if not arrow.alive:
            break
    return hits


def test_fast_shot_at_centre_hits_near_the_bull():
    target = Target(pos=(0.0, 0.0, 6.0))
    hits = run(spawn_arrow(CENTRE, power=1.0), [target])
    assert len(hits) == 1
    assert hits[0].fraction < 0.28  # gravity drop over 5.5 m is small at 55 m/s
    assert hits[0].points == 10


def test_plane_crossing_catches_a_tunnelling_arrow():
    """One 1.7 m step jumps clean over a flat target — the crossing test
    must still register it."""
    target = Target(pos=(0.0, 0.0, 6.0))
    arrow = Arrow(pos=(0.0, 0.0, 5.0), vel=(0.0, 0.0, 200.0))
    hits = step_arrows([arrow], [target], config.PHYSICS_DT)
    assert arrow.pos[2] > 6.0  # it did overshoot in a single step
    assert len(hits) == 1
    assert hits[0].points == 10


def test_gravity_drops_a_slow_arrow_under_a_far_target():
    target = Target(pos=(0.0, 0.0, config.TARGET_DEPTHS_M[-1]))
    hits = run(spawn_arrow(CENTRE, power=0.0), [target])
    assert hits == []


def test_arrow_dies_past_max_depth():
    arrow = spawn_arrow(CENTRE, power=1.0)
    run(arrow, [])
    assert not arrow.alive


def test_score_rings():
    assert score_for(0.10) == 10
    assert score_for(0.40) == 5
    assert score_for(0.80) == 2
    assert score_for(1.20) == 0
