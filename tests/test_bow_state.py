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


def hand(ratio: float, at=DOCK) -> HandGesture:
    return HandGesture(wrist=at, pinch_point=at, pinch_ratio=ratio)


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
