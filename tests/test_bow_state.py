"""Drive the bow state machine through every row of the PLAN.md transition
table using scripted GestureFrames at a 33 ms cadence (30 fps camera)."""

from fletchflow import config
from fletchflow.input.bow_input import BowState, BowStateMachine
from fletchflow.input.gestures import GestureFrame, HandGesture

FRAME_MS = 33

OPEN = 0.9    # comfortably above PINCH_OFF
PINCHED = 0.2  # comfortably below PINCH_ON


def hand(ratio: float, wrist=(0.3, 0.5)) -> HandGesture:
    return HandGesture(wrist=wrist, pinch_point=wrist, pinch_ratio=ratio)


class Driver:
    def __init__(self):
        self.machine = BowStateMachine()
        self.t = 0
        self.snap = None

    def step(self, left: HandGesture | None, right: HandGesture | None, n=1):
        for _ in range(n):
            self.t += FRAME_MS
            self.snap = self.machine.update(
                GestureFrame(timestamp_ms=self.t, left=left, right=right)
            )
        return self.snap

    def arm(self):
        """IDLE -> ARMED with both hands open."""
        self.step(hand(OPEN), hand(OPEN), n=config.ARM_FRAMES)
        assert self.machine.state == BowState.ARMED
        return self

    def draw(self, draw_wrist=(0.7, 0.5)):
        """ARMED -> DRAWN by pinching the right hand."""
        pinching = hand(PINCHED, wrist=draw_wrist)
        self.step(hand(OPEN, wrist=(0.3, 0.5)), pinching, n=config.PINCH_ON_FRAMES)
        assert self.machine.state == BowState.DRAWN
        return self


def test_idle_to_armed_needs_consecutive_frames():
    d = Driver()
    d.step(hand(OPEN), hand(OPEN), n=config.ARM_FRAMES - 1)
    assert d.machine.state == BowState.IDLE
    d.step(hand(OPEN), hand(OPEN))
    assert d.machine.state == BowState.ARMED


def test_single_hand_never_arms():
    d = Driver()
    d.step(hand(OPEN), None, n=20)
    assert d.machine.state == BowState.IDLE


def test_pinch_draws_and_release_fires():
    d = Driver().arm().draw()
    # release: open the pinch for PINCH_OFF_FRAMES
    snap = d.step(hand(OPEN), hand(OPEN, wrist=(0.7, 0.5)), n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    assert snap.fired_power is not None


def test_fired_power_is_max_of_recent_window():
    d = Driver().arm()
    # draw with hands far apart (high power)...
    d.draw(draw_wrist=(0.9, 0.5))  # dist 0.6 -> power 1.0
    # ...then hands drift together for 2 frames just before release
    d.step(hand(OPEN, wrist=(0.3, 0.5)), hand(PINCHED, wrist=(0.5, 0.5)), n=2)
    snap = d.step(hand(OPEN, wrist=(0.3, 0.5)), hand(OPEN, wrist=(0.5, 0.5)),
                  n=config.PINCH_OFF_FRAMES)
    assert snap.fired_power == 1.0  # max over window, not the weak final value


def test_pinch_glitch_does_not_draw():
    d = Driver().arm()
    for _ in range(4):  # alternating single-frame pinches (tracking noise)
        d.step(hand(OPEN), hand(PINCHED))
        d.step(hand(OPEN), hand(OPEN))
    assert d.machine.state == BowState.ARMED


def test_hand_loss_cancels_draw_without_firing():
    d = Driver().arm().draw()
    lost_frames = config.HAND_LOST_GRACE_MS // FRAME_MS + 2
    for _ in range(lost_frames):
        snap = d.step(hand(OPEN), None)
        assert snap.fired_power is None
    assert d.machine.state == BowState.ARMED


def test_cooldown_then_rearm():
    d = Driver().arm().draw()
    d.step(hand(OPEN), hand(OPEN, wrist=(0.7, 0.5)), n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    cooldown_frames = config.COOLDOWN_MS // FRAME_MS + 2
    d.step(hand(OPEN), hand(OPEN), n=cooldown_frames)
    assert d.machine.state == BowState.ARMED


def test_both_hands_lost_goes_idle():
    d = Driver().arm()
    idle_frames = config.IDLE_TIMEOUT_MS // FRAME_MS + 2
    d.step(None, None, n=idle_frames)
    assert d.machine.state == BowState.IDLE


def test_power_clamped_to_unit_range():
    d = Driver().arm()
    snap = d.draw(draw_wrist=(0.95, 0.5)).snap  # far beyond DRAW_MAX
    assert snap.power == 1.0
    d2 = Driver().arm()
    snap = d2.draw(draw_wrist=(0.35, 0.5)).snap  # dist 0.05 < DRAW_MIN
    assert snap.power == 0.0
