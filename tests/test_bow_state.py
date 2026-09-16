"""Drive the grab-based bow state machine through every transition-table row
using scripted GestureFrames at a 33 ms cadence (30 fps camera).

M4b: the bow is held with a closed FIST; the string takes a pinch OR a fist and
fires when the hand goes flat. Power is 3D, measured in hand-widths.
"""

import math

import pytest

from fletchflow import config
from fletchflow.input.bow_input import BowState, BowStateMachine
from fletchflow.input.calibration import CalibrationResult
from fletchflow.input.gestures import GestureFrame, HandGesture

FRAME_MS = 33

OPEN = 0.9        # pinch_ratio comfortably above PINCH_OFF
PINCHED = 0.2     # pinch_ratio comfortably below PINCH_ON
FIST_CLOSED = 0.9   # fist_ratio comfortably below FIST_ON
FIST_OPEN = 2.0     # fist_ratio comfortably above FIST_OFF

DOCK = config.DOCK_POS
FAR = (0.9, 0.9)  # far from dock and from any anchor

# A pinhole model of the player, for the depth tests: a hand of physical size
# HAND_M metres at depth z_m appears this big in normalized coords.
HAND_M = 0.09


def palm_at(z_m: float) -> float:
    return config.CAM_FOCAL_NORM * HAND_M / z_m


def hand(
    ratio: float = OPEN,
    at=DOCK,
    size: float = config.REFERENCE_HAND_SIZE,
    fist: float = FIST_OPEN,
    palm: float | None = None,
) -> HandGesture:
    """One hand. `at` is used as both the pinch point and the palm grip point."""
    return HandGesture(
        wrist=at,
        pinch_point=at,
        grip_point=at,
        pinch_ratio=ratio,
        fist_ratio=fist,
        size=size,
        palm_size=size if palm is None else palm,
    )


def fist_hand(at=DOCK, **kw) -> HandGesture:
    """A closed fist. pinch_ratio 0.45 mimics the thumb lying across the
    fingers — below PINCH_OFF, which is exactly why release needs both ratios."""
    kw.setdefault("ratio", 0.45)
    return hand(at=at, fist=FIST_CLOSED, **kw)


def open_hand(at=DOCK, **kw) -> HandGesture:
    return hand(ratio=OPEN, at=at, fist=FIST_OPEN, **kw)


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

    def grab(self, side="left", **kw):
        """DOCKED -> HELD by closing a fist at the dock with `side`."""
        hands = {side: fist_hand(at=DOCK, **kw)}
        self.step(hands.get("left"), hands.get("right"), n=config.FIST_ON_FRAMES)
        assert self.machine.state == BowState.HELD
        return self

    def draw(self, draw_at=DOCK, with_fist=False, **kw):
        """HELD (left holds bow at dock) -> DRAWN with the right hand."""
        grip = fist_hand(at=draw_at, **kw) if with_fist else hand(PINCHED, at=draw_at, **kw)
        frames = config.FIST_ON_FRAMES if with_fist else config.PINCH_ON_FRAMES
        self.step(fist_hand(at=DOCK, **kw), grip, n=frames)
        assert self.machine.state == BowState.DRAWN
        return self


def pull_point(distance: float):
    """A point `distance` below the dock (pure-y pull keeps math simple)."""
    return (DOCK[0], DOCK[1] + distance)


# Full-power lateral pull: DRAW_FULL_HW hand-widths of a reference-sized hand.
FULL_PULL = config.DRAW_FULL_HW * config.REFERENCE_HAND_SIZE


# -- grabbing the bow ------------------------------------------------------


def test_fist_at_dock_grabs_the_bow():
    d = Driver()
    d.step(fist_hand(at=DOCK), None, n=config.FIST_ON_FRAMES - 1)
    assert d.machine.state == BowState.DOCKED
    d.step(fist_hand(at=DOCK), None)
    assert d.machine.state == BowState.HELD


def test_pinch_alone_does_not_grab_the_bow():
    """The bow needs a fist now — a pinch at the dock must be ignored."""
    d = Driver()
    d.step(hand(PINCHED, at=DOCK, fist=FIST_OPEN), None, n=20)
    assert d.machine.state == BowState.DOCKED


def test_fist_far_from_dock_does_not_grab():
    d = Driver()
    d.step(fist_hand(at=FAR), None, n=20)
    assert d.machine.state == BowState.DOCKED


def test_anchor_follows_bow_hand_grip_point():
    d = Driver().grab()
    held_at = (0.35, 0.6)
    snap = d.step(fist_hand(at=held_at), None)
    assert snap.anchor == held_at


def test_either_hand_can_grab_the_bow():
    d = Driver()
    d.step(None, fist_hand(at=DOCK), n=config.FIST_ON_FRAMES)
    assert d.machine.state == BowState.HELD
    assert d.machine.bow_side == "right"
    d.step(hand(PINCHED, at=DOCK), fist_hand(at=DOCK), n=config.PINCH_ON_FRAMES)
    assert d.machine.state == BowState.DRAWN
    assert d.machine.draw_side == "left"


def test_grab_radius_scales_with_hand_size():
    far_grip = pull_point(0.13)  # outside base GRAB_RADIUS=0.11
    big = Driver()
    big.step(fist_hand(at=far_grip, size=config.REFERENCE_HAND_SIZE * 1.5),
             None, n=config.FIST_ON_FRAMES)
    assert big.machine.state == BowState.HELD  # scale 1.5 -> effective radius 0.165

    normal = Driver()
    normal.step(fist_hand(at=far_grip, size=config.REFERENCE_HAND_SIZE), None, n=20)
    assert normal.machine.state == BowState.DOCKED


# -- grabbing the string ---------------------------------------------------


def test_string_grabs_with_pinch_or_fist():
    """Either grip takes the string — the player should not have to guess."""
    with_pinch = Driver().grab().draw()
    assert with_pinch.machine.state == BowState.DRAWN

    with_fist = Driver().grab().draw(with_fist=True)
    assert with_fist.machine.state == BowState.DRAWN


def test_string_grab_requires_proximity_to_anchor():
    d = Driver().grab()
    d.step(fist_hand(at=DOCK), hand(PINCHED, at=FAR), n=20)
    assert d.machine.state == BowState.HELD  # grabbing far away does nothing


def test_pinch_glitch_does_not_draw():
    d = Driver().grab()
    for _ in range(4):  # alternating single-frame pinches (tracking noise)
        d.step(fist_hand(at=DOCK), hand(PINCHED, at=DOCK))
        d.step(fist_hand(at=DOCK), open_hand(at=DOCK))
    assert d.machine.state == BowState.HELD


# -- release ---------------------------------------------------------------


def test_fist_grip_release_requires_both_ratios_open():
    """Regression test for the load-bearing AND (PLAN.md 4.6.1).

    A hand released from a fist passes through a pose where the fingers have
    opened but the thumb still reads as a pinch. Firing on fist_ratio alone
    would loose the arrow early; firing on pinch_ratio alone would never loose
    it at all, because a tight fist parks pinch_ratio below PINCH_OFF.
    """
    d = Driver().grab().draw(with_fist=True)
    d.step(fist_hand(at=DOCK), fist_hand(at=pull_point(FULL_PULL)))

    # Fingers open, thumb still across them: pinch_ratio 0.45 < PINCH_OFF
    half_open = hand(ratio=0.45, at=pull_point(FULL_PULL), fist=FIST_OPEN)
    d.step(fist_hand(at=DOCK), half_open, n=config.PINCH_OFF_FRAMES + 3)
    assert d.machine.state == BowState.DRAWN, "must not fire on a half-open hand"

    # Now genuinely flat: both ratios open
    snap = d.step(fist_hand(at=DOCK), open_hand(at=pull_point(FULL_PULL)),
                  n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    assert snap.fired_power is not None


def test_release_fires_with_max_recent_power():
    d = Driver().grab().draw()
    d.step(fist_hand(at=DOCK), hand(PINCHED, at=pull_point(FULL_PULL)))
    # hand creeps back toward the bow just before release
    d.step(fist_hand(at=DOCK), hand(PINCHED, at=pull_point(FULL_PULL / 4)), n=2)
    snap = d.step(fist_hand(at=DOCK), open_hand(at=pull_point(FULL_PULL / 4)),
                  n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    assert snap.fired_power == 1.0


def test_cooldown_returns_to_held_while_still_holding():
    d = Driver().grab().draw()
    d.step(fist_hand(at=DOCK), open_hand(at=DOCK), n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    cooldown_frames = config.COOLDOWN_MS // FRAME_MS + 2
    d.step(fist_hand(at=DOCK), open_hand(at=FAR), n=cooldown_frames)
    assert d.machine.state == BowState.HELD


# -- release rule (both_open vs grip_aware) ---------------------------------


def test_default_release_rule_is_both_open():
    assert config.RELEASE_RULE == "both_open"
    assert BowStateMachine().release_rule == "both_open"


def test_both_open_holds_a_relaxed_pinch_release():
    """Documents the problem: a relaxed pinch leaves the other fingers curled,
    which parks fist_ratio around 1.3 -- below FIST_OFF -- so both_open never
    reads the hand as open and the shot never releases."""
    d = Driver().grab().draw()
    relaxed = hand(ratio=0.9, at=DOCK, fist=1.3)
    d.step(fist_hand(at=DOCK), relaxed, n=config.PINCH_OFF_FRAMES + 3)
    assert d.machine.state == BowState.DRAWN


def test_grip_aware_releases_a_relaxed_pinch():
    d = Driver().grab().draw()
    d.machine.set_release_rule("grip_aware")
    relaxed = hand(ratio=0.9, at=DOCK, fist=1.3)
    snap = d.step(fist_hand(at=DOCK), relaxed, n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    assert snap.fired_power is not None


def test_grip_aware_fist_grip_fires_when_fingers_open():
    d = Driver().grab().draw(with_fist=True)
    d.machine.set_release_rule("grip_aware")
    fingers_open = hand(ratio=0.45, at=DOCK, fist=2.0)
    snap = d.step(fist_hand(at=DOCK), fingers_open, n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    assert snap.fired_power is not None


def test_grip_aware_pinch_grip_ignores_fist_noise():
    d = Driver().grab().draw()
    d.machine.set_release_rule("grip_aware")
    noisy = hand(ratio=0.25, at=DOCK, fist=2.0)
    d.step(fist_hand(at=DOCK), noisy, n=5)
    assert d.machine.state == BowState.DRAWN


def test_grip_aware_fist_grip_ignores_the_thumb():
    d = Driver().grab().draw(with_fist=True)
    d.machine.set_release_rule("grip_aware")
    noisy = hand(ratio=0.9, at=DOCK, fist=0.9)
    d.step(fist_hand(at=DOCK), noisy, n=5)
    assert d.machine.state == BowState.DRAWN


def test_unknown_release_rule_is_rejected():
    machine = BowStateMachine()
    with pytest.raises(ValueError):
        machine.set_release_rule("nonsense")


def test_draw_grip_reports_the_grip():
    held = Driver().grab()
    assert held.machine.draw_grip == ""

    pinch_draw = Driver().grab().draw()
    assert pinch_draw.machine.draw_grip == "pinch"

    fist_draw = Driver().grab().draw(with_fist=True)
    assert fist_draw.machine.draw_grip == "fist"


def test_calibration_does_not_change_the_release_rule():
    machine = BowStateMachine()
    machine.set_release_rule("grip_aware")
    machine.apply_calibration(CalibrationResult.defaults())
    assert machine.release_rule == "grip_aware"


# -- losing the bow --------------------------------------------------------


def test_draw_hand_loss_cancels_without_firing():
    d = Driver().grab().draw()
    lost_frames = config.HAND_LOST_GRACE_MS // FRAME_MS + 2
    for _ in range(lost_frames):
        snap = d.step(fist_hand(at=DOCK), None)
        assert snap.fired_power is None
    assert d.machine.state == BowState.HELD


def test_opening_bow_fist_drops_bow_to_dock():
    d = Driver().grab().draw()
    snap = d.step(open_hand(at=DOCK), hand(PINCHED, at=pull_point(0.1)),
                  n=config.BOW_DROP_FRAMES)
    assert d.machine.state == BowState.DOCKED
    assert snap.fired_power is None  # dropping never fires
    assert snap.anchor == config.DOCK_POS


def test_bow_hand_lost_returns_to_dock():
    d = Driver().grab()
    lost_frames = config.BOW_LOST_MS // FRAME_MS + 2
    d.step(None, None, n=lost_frames)
    assert d.machine.state == BowState.DOCKED


# -- power -----------------------------------------------------------------


def test_draw_starts_at_zero_power_then_pull_raises_it():
    d = Driver().grab().draw()
    assert d.snap.power == 0.0  # baseline: no pull yet at the grab point
    snap = d.step(fist_hand(at=DOCK), hand(PINCHED, at=pull_point(FULL_PULL / 2)))
    assert abs(snap.power - 0.5) < 0.01
    snap = d.step(fist_hand(at=DOCK), hand(PINCHED, at=pull_point(FULL_PULL * 2)))
    assert snap.power == 1.0


def test_depth_only_draw_builds_full_power():
    """The whole point of 4b(c): a draw straight back toward the face moves the
    hand almost nowhere on screen, and used to build no power at all."""
    z0 = 0.60
    s0 = palm_at(z0)
    d = Driver().grab(palm=s0)
    d.step(fist_hand(at=DOCK, palm=s0), hand(PINCHED, at=DOCK, palm=s0),
           n=config.PINCH_ON_FRAMES)
    assert d.machine.state == BowState.DRAWN
    assert d.snap.power == 0.0

    # Draw 0.20 m back toward the face — the on-screen point does not move
    s1 = palm_at(z0 + 0.20)
    snap = d.step(fist_hand(at=DOCK, palm=s0), hand(PINCHED, at=DOCK, palm=s1), n=40)
    assert snap.power >= 0.95, f"depth-only draw gave {snap.power:.3f}"


def test_power_is_seating_distance_invariant():
    """The same physical 0.20 m draw must read the same near and far."""
    powers = []
    for z0 in (0.60, 1.20):
        s0, s1 = palm_at(z0), palm_at(z0 + 0.20)
        d = Driver().grab(palm=s0)
        d.step(fist_hand(at=DOCK, palm=s0), hand(PINCHED, at=DOCK, palm=s0),
               n=config.PINCH_ON_FRAMES)
        snap = d.step(fist_hand(at=DOCK, palm=s0), hand(PINCHED, at=DOCK, palm=s1),
                      n=40)
        powers.append(snap.power)
    assert abs(powers[0] - powers[1]) < 0.10, powers


def test_lateral_power_is_hand_size_invariant():
    """A smaller apparent hand needs a proportionally smaller on-screen pull."""
    powers = []
    for palm in (config.REFERENCE_HAND_SIZE, config.REFERENCE_HAND_SIZE / 2.0):
        pull = config.DRAW_FULL_HW * palm * 0.5  # half power, in hand-widths
        d = Driver().grab(palm=palm)
        d.step(fist_hand(at=DOCK, palm=palm), hand(PINCHED, at=DOCK, palm=palm),
               n=config.PINCH_ON_FRAMES)
        snap = d.step(fist_hand(at=DOCK, palm=palm),
                      hand(PINCHED, at=pull_point(pull), palm=palm))
        powers.append(snap.power)
    assert abs(powers[0] - 0.5) < 0.02
    assert abs(powers[0] - powers[1]) < 0.06, powers


def test_draw_power_hw_is_exposed_for_calibration():
    d = Driver().grab().draw()
    d.step(fist_hand(at=DOCK), hand(PINCHED, at=pull_point(FULL_PULL)))
    assert abs(d.machine.draw_power_hw - config.DRAW_FULL_HW) < 0.05
    assert abs(d.snap.draw_power_hw - config.DRAW_FULL_HW) < 0.05


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
        return math.hypot(tip[0] - anchor[0], tip[1] - anchor[1])

    for t1, t2 in zip(g1.tips, g2.tips):
        d1 = dist_from_anchor(t1, g1.anchor)
        d2 = dist_from_anchor(t2, g2.anchor)
        assert abs(d2 - 2.0 * d1) < 1e-6
