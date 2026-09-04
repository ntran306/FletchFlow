"""Drive the grab-based bow state machine through every transition-table row
using scripted GestureFrames at a 33 ms cadence (30 fps camera)."""

from fletchflow import config
from fletchflow.input.bow_input import BowState, BowStateMachine
from fletchflow.input.gestures import GestureFrame, HandGesture

FRAME_MS = 33

OPEN = 0.9     # comfortably above PINCH_OFF
PINCHED = 0.2  # comfortably below PINCH_ON

DOCK = config.DOCK_POS
FAR = (0.9, 0.9)  # far from dock and from any anchor


def hand(ratio: float, at=DOCK, size: float = config.REFERENCE_HAND_SIZE) -> HandGesture:
    return HandGesture(wrist=at, pinch_point=at, pinch_ratio=ratio, size=size)


class Driver:
    def __init__(self):
        self.machine = BowStateMachine()
        self.t = 0
        self.snap = None

    def step(self, left=None, right=None, n=1):
        for _ in range(n):
            self.t += FRAME_MS
            self.snap = self.machine.update(
                GestureFrame(timestamp_ms=self.t, left=left, right=right)
            )
        return self.snap

    def grab(self, side="left"):
        """DOCKED -> HELD by pinching at the dock with `side`."""
        hands = {side: hand(PINCHED, at=DOCK)}
        self.step(hands.get("left"), hands.get("right"), n=config.PINCH_ON_FRAMES)
        assert self.machine.state == BowState.HELD
        return self

    def draw(self, draw_at=DOCK):
        """HELD (left holds bow at dock) -> DRAWN with the right hand."""
        self.step(hand(PINCHED, at=DOCK), hand(PINCHED, at=draw_at),
                  n=config.PINCH_ON_FRAMES)
        assert self.machine.state == BowState.DRAWN
        return self


def pull_point(distance: float):
    """A point `distance` below the dock (pure-y pull keeps math simple)."""
    return (DOCK[0], DOCK[1] + distance)


def test_pinch_at_dock_grabs_the_bow():
    d = Driver()
    d.step(hand(PINCHED, at=DOCK), None, n=config.PINCH_ON_FRAMES - 1)
    assert d.machine.state == BowState.DOCKED
    d.step(hand(PINCHED, at=DOCK), None)
    assert d.machine.state == BowState.HELD


def test_pinch_far_from_dock_does_not_grab():
    d = Driver()
    d.step(hand(PINCHED, at=FAR), None, n=20)
    assert d.machine.state == BowState.DOCKED


def test_anchor_follows_bow_hand_pinch_point():
    d = Driver().grab()
    held_at = (0.35, 0.6)
    snap = d.step(hand(PINCHED, at=held_at), None)
    assert snap.anchor == held_at


def test_string_grab_requires_proximity_to_anchor():
    d = Driver().grab()
    d.step(hand(PINCHED, at=DOCK), hand(PINCHED, at=FAR), n=20)
    assert d.machine.state == BowState.HELD  # pinching far away does nothing


def test_draw_starts_at_zero_power_then_pull_raises_it():
    d = Driver().grab().draw()
    assert d.snap.power == 0.0  # baseline: no pull yet at the grab point
    # pull straight down by half the range
    snap = d.step(hand(PINCHED, at=DOCK),
                  hand(PINCHED, at=pull_point(config.DRAW_RANGE / 2)))
    assert abs(snap.power - 0.5) < 0.01
    # beyond full range clamps at 1.0
    snap = d.step(hand(PINCHED, at=DOCK),
                  hand(PINCHED, at=pull_point(config.DRAW_RANGE * 2)))
    assert snap.power == 1.0


def test_release_fires_with_max_recent_power():
    d = Driver().grab().draw()
    d.step(hand(PINCHED, at=DOCK), hand(PINCHED, at=pull_point(config.DRAW_RANGE)))
    # hand creeps back toward the bow just before release
    d.step(hand(PINCHED, at=DOCK),
           hand(PINCHED, at=pull_point(config.DRAW_RANGE / 4)), n=2)
    snap = d.step(hand(PINCHED, at=DOCK),
                  hand(OPEN, at=pull_point(config.DRAW_RANGE / 4)),
                  n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    assert snap.fired_power == 1.0


def test_pinch_glitch_does_not_draw():
    d = Driver().grab()
    for _ in range(4):  # alternating single-frame pinches (tracking noise)
        d.step(hand(PINCHED, at=DOCK), hand(PINCHED, at=DOCK))
        d.step(hand(PINCHED, at=DOCK), hand(OPEN, at=DOCK))
    assert d.machine.state == BowState.HELD


def test_draw_hand_loss_cancels_without_firing():
    d = Driver().grab().draw()
    lost_frames = config.HAND_LOST_GRACE_MS // FRAME_MS + 2
    for _ in range(lost_frames):
        snap = d.step(hand(PINCHED, at=DOCK), None)
        assert snap.fired_power is None
    assert d.machine.state == BowState.HELD


def test_opening_bow_hand_drops_bow_to_dock():
    d = Driver().grab().draw()
    snap = d.step(hand(OPEN, at=DOCK), hand(PINCHED, at=pull_point(0.1)),
                  n=config.BOW_DROP_FRAMES)
    assert d.machine.state == BowState.DOCKED
    assert snap.fired_power is None  # dropping never fires
    assert snap.anchor == config.DOCK_POS


def test_bow_hand_lost_returns_to_dock():
    d = Driver().grab()
    lost_frames = config.BOW_LOST_MS // FRAME_MS + 2
    d.step(None, None, n=lost_frames)
    assert d.machine.state == BowState.DOCKED


def test_cooldown_returns_to_held_while_still_pinching():
    d = Driver().grab().draw()
    d.step(hand(PINCHED, at=DOCK), hand(OPEN, at=DOCK), n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    cooldown_frames = config.COOLDOWN_MS // FRAME_MS + 2
    d.step(hand(PINCHED, at=DOCK), hand(OPEN, at=FAR), n=cooldown_frames)
    assert d.machine.state == BowState.HELD


def test_either_hand_can_grab_the_bow():
    d = Driver()
    d.step(None, hand(PINCHED, at=DOCK), n=config.PINCH_ON_FRAMES)
    assert d.machine.state == BowState.HELD
    # ...and then the left hand draws
    d.step(hand(PINCHED, at=DOCK), hand(PINCHED, at=DOCK),
           n=config.PINCH_ON_FRAMES)
    assert d.machine.state == BowState.DRAWN


def test_power_is_distance_invariant():
    """Half the apparent hand size should need half the physical pull for
    the same power — DRAW_RANGE is normalized by the bow hand's scale."""
    ref = Driver().grab()  # default hand() size == REFERENCE_HAND_SIZE -> scale 1.0
    small = Driver()
    small_size = config.REFERENCE_HAND_SIZE / 2.0
    small.step(hand(PINCHED, at=DOCK, size=small_size), None, n=config.PINCH_ON_FRAMES)
    assert small.machine.state == BowState.HELD

    # Let the scale EMA converge while held, before grabbing the string.
    ref.step(hand(PINCHED, at=DOCK), None, n=40)
    small.step(hand(PINCHED, at=DOCK, size=small_size), None, n=40)

    ref.step(hand(PINCHED, at=DOCK), hand(PINCHED, at=DOCK), n=config.PINCH_ON_FRAMES)
    assert ref.machine.state == BowState.DRAWN
    small.step(hand(PINCHED, at=DOCK, size=small_size),
               hand(PINCHED, at=DOCK, size=small_size), n=config.PINCH_ON_FRAMES)
    assert small.machine.state == BowState.DRAWN

    ref_snap = ref.step(hand(PINCHED, at=DOCK),
                         hand(PINCHED, at=pull_point(config.DRAW_RANGE * 0.5)))
    small_snap = small.step(hand(PINCHED, at=DOCK, size=small_size),
                             hand(PINCHED, at=pull_point(config.DRAW_RANGE * 0.25),
                                  size=small_size))

    ref_expected = min(1.0, (config.DRAW_RANGE * 0.5) / (config.DRAW_RANGE * ref_snap.scale))
    small_expected = min(1.0, (config.DRAW_RANGE * 0.25) / (config.DRAW_RANGE * small_snap.scale))
    assert abs(ref_snap.power - ref_expected) < 1e-6
    assert abs(small_snap.power - small_expected) < 1e-6
    assert abs(ref_snap.power - small_snap.power) < 0.06


def test_grab_radius_scales_with_hand_size():
    far_pinch = pull_point(0.13)  # outside base GRAB_RADIUS=0.11
    big = Driver()
    big.step(hand(PINCHED, at=far_pinch, size=config.REFERENCE_HAND_SIZE * 1.5),
              None, n=config.PINCH_ON_FRAMES)
    assert big.machine.state == BowState.HELD  # scale 1.5 -> effective radius 0.165

    normal = Driver()
    normal.step(hand(PINCHED, at=far_pinch, size=config.REFERENCE_HAND_SIZE),
                None, n=20)
    assert normal.machine.state == BowState.DOCKED


def test_geometry_scales():
    from fletchflow.input.mapping import BowPose
    from fletchflow.render.bow import compute_geometry

    base = BowPose(
        anchor=(100.0, 100.0), draw_point=None, aim=(0.0, -1.0),
        power=0.5, state=BowState.HELD, fire=None, scale=1.0,
    )
    doubled = BowPose(
        anchor=(100.0, 100.0), draw_point=None, aim=(0.0, -1.0),
        power=0.5, state=BowState.HELD, fire=None, scale=2.0,
    )
    g1 = compute_geometry(base)
    g2 = compute_geometry(doubled)

    def dist_from_anchor(tip, anchor):
        return ((tip[0] - anchor[0]) ** 2 + (tip[1] - anchor[1]) ** 2) ** 0.5

    for t1, t2 in zip(g1.tips, g2.tips):
        d1 = dist_from_anchor(t1, g1.anchor)
        d2 = dist_from_anchor(t2, g2.anchor)
        assert abs(d2 - 2.0 * d1) < 1e-6
