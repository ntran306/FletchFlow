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
from fletchflow.input.hand_pose import HandPose3D

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
    pose: HandPose3D | None = None,
    knuckle: tuple[float, float] = (0.0, -1.0),
    knuckle_px: float = 150.0,   # a hand seen face on at play distance
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
        pose=pose,
        knuckle_dir=knuckle,
        knuckle_px=knuckle_px,
    )


def pose_at(z_m: float) -> HandPose3D:
    """A metric pose fit reporting depth z_m, camera metres. The other fields
    (px_per_m, residual_px, inplane_deg) never feed bow_input.py's depth path,
    so their exact values don't matter for these tests."""
    return HandPose3D(
        position_m=(0.0, 0.0, z_m), depth_m=z_m, px_per_m=100.0,
        residual_px=1.0, inplane_deg=0.0,
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

    This is specifically a "both_open" property — M4c makes "grip_aware" the
    default, so the rule is set explicitly here (PLAN.md acceptance 12).
    """
    d = Driver().grab().draw(with_fist=True)
    d.machine.set_release_rule("both_open")
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
    """Fire takes the max power over the recent window, not the value at the
    instant of release — the hand always creeps forward a little as it opens.

    Drawn sideways rather than in depth: the on-screen point is used as read,
    while depth is One Euro smoothed, so the creep-back shows up immediately
    and the window is actually exercised.
    """
    bow_z = 0.55
    W, H = config.CAPTURE_SIZE
    f = config.CAM_FOCAL_NORM * W

    def pull_at(metres):
        """A draw point whose sideways offset is `metres` at this depth."""
        return (DOCK[0], DOCK[1] + metres * f / (H * bow_z))

    bow = fist_hand(at=DOCK, pose=pose_at(bow_z))
    d, d0 = drawn_at(bow_z=bow_z)

    full = pull_at(config.DRAW_FULL_M * 1.2)
    d.step(bow, hand(PINCHED, at=full, pose=pose_at(bow_z)))
    assert d.snap.power >= 0.95

    crept = pull_at(config.DRAW_FULL_M * 0.3)
    d.step(bow, hand(PINCHED, at=crept, pose=pose_at(bow_z)), n=2)
    assert d.snap.power < 0.6, "the creep-back must actually drop live power"

    snap = d.step(bow, open_hand(at=crept, pose=pose_at(bow_z)),
                  n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    assert snap.fired_power >= 0.95, snap.fired_power


def test_cooldown_returns_to_held_while_still_holding():
    d = Driver().grab().draw()
    d.step(fist_hand(at=DOCK), open_hand(at=DOCK), n=config.PINCH_OFF_FRAMES)
    assert d.machine.state == BowState.RELEASED
    cooldown_frames = config.COOLDOWN_MS // FRAME_MS + 2
    d.step(fist_hand(at=DOCK), open_hand(at=FAR), n=cooldown_frames)
    assert d.machine.state == BowState.HELD


# -- release rule (both_open vs grip_aware) ---------------------------------


def test_default_release_rule_is_grip_aware():
    """M4c (PLAN.md acceptance 12): playtest telemetry showed both_open left
    12.8% of drawn frames blocked-open, so grip_aware is now the default."""
    assert config.RELEASE_RULE == "grip_aware"
    assert BowStateMachine().release_rule == "grip_aware"


def test_both_open_holds_a_relaxed_pinch_release():
    """Documents the problem: a relaxed pinch leaves the other fingers curled,
    which parks fist_ratio around 1.3 -- below FIST_OFF -- so both_open never
    reads the hand as open and the shot never releases.

    A "both_open" property — set explicitly since grip_aware is now the
    default (PLAN.md acceptance 12)."""
    d = Driver().grab().draw()
    d.machine.set_release_rule("both_open")
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


# -- power (metric, PLAN.md §4.7.6) ------------------------------------------


def hand_xyz(point, depth):
    """Back-project a normalized point at a known depth to camera metres.

    Written out here rather than imported from bow_input so these tests check
    the formula independently instead of re-running the code under test.
    """
    W, H = config.CAPTURE_SIZE
    f = config.CAM_FOCAL_NORM * W
    return (
        (point[0] * W - W / 2.0) * depth / f,
        (point[1] * H - H / 2.0) * depth / f,
        depth,
    )


def sep(point_a, depth_a, point_b, depth_b):
    """Metric 3D separation between two tracked points."""
    a, b = hand_xyz(point_a, depth_a), hand_xyz(point_b, depth_b)
    return math.dist(a, b)


def expected_power(bow_pt, bow_z, draw_pt, draw_z, d0):
    pull = max(0.0, sep(bow_pt, bow_z, draw_pt, draw_z) - d0)
    return min(1.0, pull / config.DRAW_FULL_M)


def drawn_at(bow_z=0.55, draw_z=0.55, draw_pt=DOCK):
    """Grab the bow and string with known metric depths; return (driver, d0)."""
    d = Driver().grab(pose=pose_at(bow_z))
    d.step(fist_hand(at=DOCK, pose=pose_at(bow_z)),
           hand(PINCHED, at=draw_pt, pose=pose_at(draw_z)),
           n=config.PINCH_ON_FRAMES)
    assert d.machine.state == BowState.DRAWN
    return d, sep(DOCK, bow_z, draw_pt, draw_z)


def test_draw_starts_at_zero_power_then_pull_raises_it():
    """power = clamp((|P_bow - P_draw| - d0) / DRAW_FULL_M, 0, 1)."""
    bow_z = 0.55
    d, d0 = drawn_at(bow_z=bow_z)
    assert d.snap.power == 0.0  # baseline: no pull yet at the grab point

    for draw_z in (0.60, 0.65, 0.75, 1.00):
        snap = d.step(fist_hand(at=DOCK, pose=pose_at(bow_z)),
                      hand(PINCHED, at=DOCK, pose=pose_at(draw_z)), n=40)
        want = expected_power(DOCK, bow_z, DOCK, draw_z, d0)
        assert abs(snap.power - want) < 0.02, (draw_z, snap.power, want)


def test_depth_only_draw_builds_full_power():
    """A draw straight back toward the face moves the hand almost nowhere on
    screen. Before 4b(c) it built no power at all; now it is just distance."""
    bow_z = 0.60
    d, d0 = drawn_at(bow_z=bow_z, draw_z=bow_z)
    assert d.snap.power == 0.0

    # Back far enough that the separation clears DRAW_FULL_M
    draw_z = bow_z + config.DRAW_FULL_M * 1.2
    snap = d.step(fist_hand(at=DOCK, pose=pose_at(bow_z)),
                  hand(PINCHED, at=DOCK, pose=pose_at(draw_z)), n=40)
    assert snap.power >= 0.95, f"depth-only draw gave {snap.power:.3f}"


def test_power_is_seating_distance_invariant():
    """The same physical 0.20 m draw must read the same near and far. In
    metres this is structural rather than a scale-invariance trick."""
    powers = []
    for bow_z in (0.60, 1.20):
        d, d0 = drawn_at(bow_z=bow_z, draw_z=bow_z)
        snap = d.step(fist_hand(at=DOCK, pose=pose_at(bow_z)),
                      hand(PINCHED, at=DOCK, pose=pose_at(bow_z + 0.20)), n=40)
        powers.append(snap.power)
    assert abs(powers[0] - powers[1]) < 0.02, powers


def test_the_same_physical_sideways_pull_reads_the_same_at_any_depth():
    """A lateral pull is metres too. The further away the player sits, the
    fewer pixels the same real pull covers, and power must not care."""
    powers = []
    for bow_z in (0.60, 1.20):
        W, H = config.CAPTURE_SIZE
        f = config.CAM_FOCAL_NORM * W
        # normalized-y offset whose metric length at this depth is 0.10 m
        dv = 0.10 * f / (H * bow_z)
        draw_pt = (DOCK[0], DOCK[1] + dv)
        d, d0 = drawn_at(bow_z=bow_z, draw_z=bow_z)  # grab at the dock
        snap = d.step(fist_hand(at=DOCK, pose=pose_at(bow_z)),
                      hand(PINCHED, at=draw_pt, pose=pose_at(bow_z)), n=40)
        powers.append(snap.power)
        want = expected_power(DOCK, bow_z, draw_pt, bow_z, d0)
        assert abs(snap.power - want) < 0.02, (bow_z, snap.power, want)
    assert abs(powers[0] - powers[1]) < 0.02, powers


def test_pull_m_is_exposed_for_calibration():
    """Calibration step 4 measures DRAW_FULL_M from this, so it has to be the
    raw pull in metres, before the DRAW_FULL_M division."""
    bow_z = 0.55
    d, d0 = drawn_at(bow_z=bow_z)
    draw_z = bow_z + 0.12
    snap = d.step(fist_hand(at=DOCK, pose=pose_at(bow_z)),
                  hand(PINCHED, at=DOCK, pose=pose_at(draw_z)), n=40)
    want = sep(DOCK, bow_z, DOCK, draw_z) - d0
    assert abs(d.machine.pull_m - want) < 0.005, (d.machine.pull_m, want)
    assert abs(snap.pull_m - want) < 0.005


def test_pull_m_holds_its_last_value_while_the_draw_hand_is_lost():
    """PLAN.md §4.7.7: within the grace window a lost draw hand freezes the
    pull. Reading zero instead would collapse the power bar mid-draw."""
    bow_z = 0.55
    d, _d0 = drawn_at(bow_z=bow_z)
    d.step(fist_hand(at=DOCK, pose=pose_at(bow_z)),
           hand(PINCHED, at=DOCK, pose=pose_at(bow_z + 0.12)), n=40)
    held = d.machine.pull_m
    assert held > 0.05

    d.step(fist_hand(at=DOCK, pose=pose_at(bow_z)), None, n=10)
    assert d.machine.state == BowState.DRAWN
    assert abs(d.machine.pull_m - held) < 1e-9


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


# -- M4c metric pose & robustness (PLAN.md §4.7.1/§4.7.7, acceptance 9-11) --


def test_draw_hand_loss_500ms_holds_position_700ms_cancels():
    """Acceptance 9: HAND_LOST_GRACE_MS = 600. A loss shorter than that must
    not cancel the draw, and draw_position_m freezes at its last value rather
    than clearing; a loss past it cancels (back to HELD, no fire)."""
    d = Driver().grab()
    p = pose_at(0.80)
    snap = d.step(
        fist_hand(at=DOCK), hand(PINCHED, at=DOCK, pose=p), n=config.PINCH_ON_FRAMES
    )
    assert d.machine.state == BowState.DRAWN
    assert snap.draw_position_m is not None
    frozen = snap.draw_position_m

    start_lost = d.t
    while d.t - start_lost < 500:
        snap = d.step(fist_hand(at=DOCK), None)
    assert d.machine.state == BowState.DRAWN, "500 ms loss must not cancel"
    assert snap.draw_position_m == frozen, "draw_position_m must freeze, not clear"

    while d.t - start_lost < 700:
        snap = d.step(fist_hand(at=DOCK), None)
    assert d.machine.state == BowState.HELD, "700 ms loss must cancel"
    assert snap.fired_power is None


def test_eleven_open_frames_do_not_drop_twelve_do():
    """Acceptance 10: BOW_DROP_FRAMES = 12. fist_ratio 1.7 is a genuine open
    reading (between FIST_OFF and FIST_RATIO_GLITCH), so it counts."""
    d = Driver().grab()
    d.step(hand(at=DOCK, fist=1.7), None, n=config.BOW_DROP_FRAMES - 1)
    assert d.machine.state == BowState.HELD, "11 open frames must not drop the bow"
    snap = d.step(hand(at=DOCK, fist=1.7), None)
    assert d.machine.state == BowState.DOCKED, "the 12th open frame must drop it"
    assert snap.fired_power is None


def test_glitch_frames_interleaved_in_an_open_streak_do_not_count_or_reset():
    """Acceptance 10: a FIST_RATIO_GLITCH-range reading (playtest saw 9.44) is
    a tracking glitch, not a real reading -- it must neither advance nor reset
    the drop counter, even interleaved mid-streak."""
    d = Driver().grab()
    for _ in range(config.BOW_DROP_FRAMES - 1):
        d.step(hand(at=DOCK, fist=1.7), None)
        d.step(hand(at=DOCK, fist=9.44), None)  # glitch, interleaved
    assert d.machine.state == BowState.HELD, "glitches must not count toward the drop"

    # ...and the 12th genuine open frame still drops it. Without this the test
    # could not tell "the glitch did not count" from "the glitch reset the
    # counter" -- both leave the bow held above.
    d.step(hand(at=DOCK, fist=1.7), None)
    assert d.machine.state == BowState.DOCKED, "glitches must not reset the counter"


def test_fist_grip_grip_aware_does_not_fire_on_a_single_glitch_frame():
    """Acceptance 10: under grip_aware (the default), a fist grip fires on
    fist_ratio > FIST_OFF alone -- so a single glitch-range reading (9.44)
    must not be read as that "open" signal, or tracking noise would fire a
    shot."""
    d = Driver().grab().draw(with_fist=True)
    assert d.machine.release_rule == "grip_aware"
    d.step(fist_hand(at=DOCK), fist_hand(at=pull_point(FULL_PULL)))
    snap = d.step(fist_hand(at=DOCK), hand(at=pull_point(FULL_PULL), fist=9.44))
    assert d.machine.state == BowState.DRAWN
    assert snap.fired_power is None


def test_render_scale_converges_and_clamps_and_holds_on_lost_pose():
    """Acceptance 11: render_scale = clamp(REFERENCE_BOW_DEPTH_M / depth,
    *BOW_SCALE_RANGE). Feeding the pose from the very first tracked frame
    (the grab itself) means the One Euro filter's first-sample passthrough
    makes the depth exact rather than merely asymptotically convergent."""
    d = Driver()
    p045 = pose_at(0.45)
    d.step(fist_hand(at=DOCK, pose=p045), None, n=config.FIST_ON_FRAMES)
    assert d.machine.state == BowState.HELD
    snap = d.step(fist_hand(at=DOCK, pose=p045), None, n=10)
    expected = config.REFERENCE_BOW_DEPTH_M / 0.45
    assert abs(snap.render_scale - expected) < 1e-6
    assert abs(snap.render_scale - 1.2222) < 1e-3

    p020 = pose_at(0.20)
    snap = d.step(fist_hand(at=DOCK, pose=p020), None, n=10)
    assert abs(snap.render_scale - 1.25) < 1e-6  # clamped to BOW_SCALE_RANGE[1]

    held_scale = snap.render_scale
    snap = d.step(fist_hand(at=DOCK, pose=None), None)  # rejected fit this frame
    assert snap.render_scale == held_scale, "must hold the last depth, not fall back"


# -- the string grab spans the whole limb (PLAN.md §4.7.5, acceptance 11) ----


def _string_geometry():
    """(anchor_iso, half_span, aspect) for a bow drawn at render_scale 1.0.

    Mirrors bow_input._string_zone so the tests below state the geometry they
    depend on instead of hard-coding pixel numbers that a config change would
    silently invalidate.
    """
    W, H = config.CAPTURE_SIZE
    aspect = H / W
    half = (config.BOW_SPAN_PX / 2.0) / config.WINDOW_SIZE[0]
    return (DOCK[0], DOCK[1] * aspect), half, aspect


def test_string_grabs_along_the_whole_limb_not_just_at_the_grip():
    """Acceptance 11: a point 0.20 from the anchor but only ~0.05 from the
    string segment grabs the string.

    Under the old anchor-disc test (STRING_GRAB_RADIUS from the grip point) a
    point that far out was unreachable, which is why making the bow bigger did
    not make its string any easier to grab.
    """
    (cx, cy), half, aspect = _string_geometry()
    depth = config.REFERENCE_BOW_DEPTH_M  # puts render_scale at 1.0
    d = Driver().grab(pose=pose_at(depth))

    off = 0.05                                   # sideways, off the string
    along = math.sqrt(0.20 ** 2 - off ** 2)      # ...and down the limb
    p_iso = (cx + off, cy + along)
    assert abs(math.hypot(off, along) - 0.20) < 1e-9, "point must be 0.20 from the anchor"

    # Beyond the lower limb tip, so the nearest point on the segment is the tip
    seg_dist = math.hypot(off, max(0.0, p_iso[1] - (cy + half)))
    assert abs(seg_dist - 0.05) < 0.005, f"should sit ~0.05 off the string, got {seg_dist}"
    assert seg_dist < config.STRING_GRAB_RADIUS

    point = (p_iso[0], p_iso[1] / aspect)        # back to normalized coords
    d.step(fist_hand(at=DOCK, pose=pose_at(depth)), hand(PINCHED, at=point),
           n=config.PINCH_ON_FRAMES)
    assert d.machine.state == BowState.DRAWN


def test_the_string_zone_is_a_segment_not_a_bigger_disc():
    """The same 0.20 from the anchor, but straight out sideways — far from the
    string — must NOT grab. Without this the widened zone could just be a
    larger circle, which would grab the air beside the bow."""
    (cx, cy), _half, aspect = _string_geometry()
    depth = config.REFERENCE_BOW_DEPTH_M
    d = Driver().grab(pose=pose_at(depth))

    p_iso = (cx + 0.20, cy)                      # perpendicular to the limb
    assert 0.20 > config.STRING_GRAB_RADIUS
    point = (p_iso[0], p_iso[1] / aspect)
    d.step(fist_hand(at=DOCK, pose=pose_at(depth)), hand(PINCHED, at=point), n=20)
    assert d.machine.state == BowState.HELD


def test_the_string_zone_follows_the_knuckles():
    """The segment is oriented by the bow hand's knuckle_dir, so rolling the
    bow rolls its string. A point along the rolled limb grabs; the same
    distance along where the limb *used* to be does not."""
    (cx, cy), half, aspect = _string_geometry()
    depth = config.REFERENCE_BOW_DEPTH_M
    rolled = (1.0, 0.0)   # bow rolled 90 deg: limbs now run left-right

    def bow_hand():
        return fist_hand(at=DOCK, pose=pose_at(depth), knuckle=rolled)

    d = Driver()
    d.step(bow_hand(), None, n=config.FIST_ON_FRAMES)
    assert d.machine.state == BowState.HELD
    d.step(bow_hand(), None, n=10)  # let the knuckle filter settle on the roll

    along_new = (cx + half * 0.8, cy)                     # on the rolled limb
    d.step(bow_hand(), hand(PINCHED, at=(along_new[0], along_new[1] / aspect)),
           n=config.PINCH_ON_FRAMES)
    assert d.machine.state == BowState.DRAWN, "should grab along the rolled limb"

    d2 = Driver()
    d2.step(bow_hand(), None, n=config.FIST_ON_FRAMES + 10)
    along_old = (cx, cy + half * 0.8)                     # where the limb was
    d2.step(bow_hand(), hand(PINCHED, at=(along_old[0], along_old[1] / aspect)), n=20)
    assert d2.machine.state == BowState.HELD, "must not grab off the rolled limb"


def test_a_rejected_fit_at_the_grab_does_not_invent_power():
    """d0 and the live hand separation must share one depth basis.

    A grab frame whose Procrustes fit was rejected seeds d0 from the
    apparent-size fallback. If real fits then start landing, the depth filter
    converging from that guess moves the separation on its own — and if the
    guess happened to sit near the bow hand's depth while the draw hand is
    really 20 cm further back, that reads as a full-power draw from a
    completely still hand. Measured at 1.000 before the fix (PLAN.md §4.7.6).
    """
    from fletchflow.input.bow_input import FALLBACK_PALM_M

    bow_z, true_draw_z = 0.55, 0.75
    fallback_z = 0.56  # the guess lands right next to the bow hand
    palm = config.CAM_FOCAL_NORM * FALLBACK_PALM_M / fallback_z

    d = Driver().grab(pose=pose_at(bow_z))
    bow = fist_hand(at=DOCK, pose=pose_at(bow_z))
    d.step(bow, hand(PINCHED, at=DOCK, pose=None, palm=palm),
           n=config.PINCH_ON_FRAMES)
    assert d.machine.state == BowState.DRAWN

    # Hold perfectly still for 3 s while real fits arrive
    for _ in range(90):
        snap = d.step(bow, hand(PINCHED, at=DOCK, pose=pose_at(true_draw_z)))
        assert snap.power < 0.01, f"a still hand invented {snap.power:.3f} power"

    # ...and a genuine pull afterwards still builds power normally
    snap = d.step(bow, hand(PINCHED, at=DOCK,
                            pose=pose_at(true_draw_z + config.DRAW_FULL_M * 1.2)),
                  n=40)
    assert snap.power >= 0.95, snap.power


def test_a_foreshortened_fist_does_not_roll_the_bow():
    """Playtest 2026-10-07: "moving your knuckle around makes the bow stuck at
    an angle instead of straight."

    A fist pointed at the camera — which is how the bow is held — foreshortens
    the index and pinky knuckles onto nearly the same pixel. knuckle_dir is
    normalized regardless, so a pixel of landmark jitter arrived as a
    full-strength roll and parked the bow at whatever angle the noise last
    said. Below BOW_UP_MIN_KNUCKLE_PX the reading is ignored and the last
    trustworthy one holds.
    """
    d = Driver().grab()

    # a good, square reading establishes upright
    d.step(fist_hand(at=DOCK, knuckle=(0.0, -1.0), knuckle_px=150.0), None, n=12)
    assert d.snap.knuckle_dir[1] < -0.9, d.snap.knuckle_dir

    # now the fist turns toward the camera and the direction becomes noise
    noise = [(0.8, -0.6), (-0.7, -0.7), (0.9, 0.4), (-0.95, 0.3)]
    for k in noise * 4:
        d.step(fist_hand(at=DOCK, knuckle=k, knuckle_px=6.0), None)
    assert d.snap.knuckle_dir[1] < -0.9, (
        f"noise rolled the bow to {d.snap.knuckle_dir}")

    # a genuine roll, properly seen, still gets through
    d.step(fist_hand(at=DOCK, knuckle=(1.0, 0.0), knuckle_px=150.0), None, n=25)
    assert d.snap.knuckle_dir[0] > 0.9, d.snap.knuckle_dir
