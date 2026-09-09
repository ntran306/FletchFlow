"""Camera -> screen mapping, and the sight pin that sits on top of it.

PLAN.md section 3 has listed this file since the project started and it had
never existed, so the mirror/scale math was untested before M4b.
"""

import math

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState
from fletchflow.input.mapping import Mapper

FRAME_MS = 33
W, H = config.WINDOW_SIZE


def snap(
    at=(0.5, 0.5),
    state=BowState.HELD,
    t=FRAME_MS,
    scale=1.0,
    draw_point=None,
):
    return BowSnapshot(
        timestamp_ms=t,
        state=state,
        anchor=at,
        draw_point=draw_point,
        power=0.0,
        fired_power=None,
        scale=scale,
    )


def settle(mapper, frames=30, **kw):
    """Run the mapper for a while so the reticle filter converges."""
    pose = None
    for i in range(frames):
        pose = mapper.map(snap(t=(i + 1) * FRAME_MS, **kw))
    return pose


def test_mirror_and_scale():
    """Normalized coords scale straight to pixels — capture and window are 16:9."""
    pose = Mapper().map(snap(at=(0.25, 0.75)))
    assert pose.anchor == (0.25 * W, 0.75 * H)


def test_sight_sits_above_the_anchor():
    pose = settle(Mapper(), at=(0.5, 0.6))
    assert pose.sight is not None
    assert abs(pose.sight[0] - pose.anchor[0]) < 2.0
    expected_y = pose.anchor[1] - config.CROSSHAIR_RISE_PX
    assert abs(pose.sight[1] - expected_y) < 2.0


def test_sight_never_coincides_with_anchor():
    """Acceptance criterion 3: the reticle stays clear of the bow at every depth."""
    for scale in (0.55, 0.8, 1.0, 1.3, 1.6):
        pose = settle(Mapper(), at=(0.5, 0.6), scale=scale)
        gap = pose.anchor[1] - pose.sight[1]
        assert gap >= 40.0, f"scale {scale}: reticle only {gap:.1f} px above the grip"


def test_sight_scales_with_depth():
    near = settle(Mapper(), at=(0.5, 0.6), scale=1.6)
    far = settle(Mapper(), at=(0.5, 0.6), scale=0.55)
    assert (near.anchor[1] - near.sight[1]) > (far.anchor[1] - far.sight[1])


def test_sight_resets_at_the_dock():
    """A re-grab should not sweep the reticle across the screen from the dock."""
    mapper = Mapper()
    settle(mapper, at=(0.2, 0.8))
    mapper.map(snap(at=config.DOCK_POS, state=BowState.DOCKED, t=99 * FRAME_MS))
    pose = mapper.map(snap(at=(0.8, 0.3), t=100 * FRAME_MS))
    # First frame after the dock passes through unfiltered: exactly on the pin
    assert abs(pose.sight[1] - (pose.anchor[1] - config.CROSSHAIR_RISE_PX)) < 1e-6


def test_aim_holds_through_small_separation():
    """A correct 3D draw collapses the on-screen separation. Without the floor
    the aim vector jitters, and aim drives bow orientation, so the bow spins."""
    mapper = Mapper()
    # Establish a clear horizontal aim: hands 128 px apart
    pose = mapper.map(snap(state=BowState.DRAWN, draw_point=(0.4, 0.5)))
    assert abs(pose.aim[0] - 1.0) < 1e-6

    # Now only 10 px apart, and vertical — below AIM_MIN_SEPARATION_PX
    close = (0.5, 0.5 + 10.0 / H)
    pose = mapper.map(
        snap(state=BowState.DRAWN, draw_point=close, t=2 * FRAME_MS)
    )
    assert abs(pose.aim[0] - 1.0) < 1e-6, "aim should hold its last stable value"
    assert abs(pose.aim[1]) < 1e-6


def test_aim_updates_above_the_separation_floor():
    mapper = Mapper()
    mapper.map(snap(state=BowState.DRAWN, draw_point=(0.4, 0.5)))
    below = (0.5, 0.5 + 60.0 / H)  # 60 px, comfortably above the floor
    pose = mapper.map(snap(state=BowState.DRAWN, draw_point=below, t=2 * FRAME_MS))
    assert pose.aim[1] < -0.9  # now pointing up


def test_reticle_is_steadier_than_the_anchor():
    """The sight has its own much heavier filter, so it does not inherit tremor."""
    mapper = Mapper()
    jitter = [6.0 if i % 2 else -6.0 for i in range(60)]
    anchor_deltas, sight_deltas = [], []
    prev = None
    for i, dx in enumerate(jitter):
        pose = mapper.map(
            snap(at=(0.5 + dx / W, 0.6), t=(i + 1) * FRAME_MS)
        )
        if prev is not None:
            anchor_deltas.append(abs(pose.anchor[0] - prev[0]))
            sight_deltas.append(abs(pose.sight[0] - prev[1]))
        prev = (pose.anchor[0], pose.sight[0])

    anchor_mean = sum(anchor_deltas) / len(anchor_deltas)
    sight_mean = sum(sight_deltas) / len(sight_deltas)
    assert sight_mean < anchor_mean / 2.0, (anchor_mean, sight_mean)


def test_fire_event_carries_anchor_and_aim():
    mapper = Mapper()
    mapper.map(snap(state=BowState.DRAWN, draw_point=(0.4, 0.5)))
    fired = BowSnapshot(
        timestamp_ms=2 * FRAME_MS, state=BowState.RELEASED, anchor=(0.5, 0.5),
        draw_point=None, power=0.0, fired_power=0.8, scale=1.0,
    )
    pose = mapper.map(fired)
    assert pose.fire is not None
    assert pose.fire.power == 0.8
    assert pose.fire.origin == (0.5 * W, 0.5 * H)
    assert math.isclose(pose.fire.direction[0], 1.0, abs_tol=1e-6)
