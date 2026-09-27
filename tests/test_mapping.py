"""Camera -> screen mapping, and the 3D sight that sits on top of it.

PLAN.md section 3 has listed this file since the project started. M4c
(PLAN.md §4.7.4) replaces the 4b sight pin (a fixed screen offset above the
grip) with a crosshair aimed along the real 3D line from the draw hand
through the bow hand, so this file's sight tests are rewritten around that:
`aim_angles()` and `Mapper.map()`'s DRAWN/RELEASED/HELD/DOCKED behaviour,
weight ramp, calibration zeroing, clamping, and the derived bow_forward /
bow_up axes. The mirror/scale math and the 2D aim vector (unchanged since
M4b, still driving the fallback 2D bow renderer) keep their original tests.
"""

import math

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState
from fletchflow.input.mapping import Mapper, aim_angles

FRAME_MS = 33
W, H = config.WINDOW_SIZE
CX, CY = W / 2.0, H / 2.0


def snap(
    at=(0.5, 0.5),
    state=BowState.HELD,
    t=FRAME_MS,
    scale=1.0,
    draw_point=None,
    bow_m=None,
    draw_m=None,
    knuckle=(0.0, -1.0),
    render_scale=1.0,
):
    return BowSnapshot(
        timestamp_ms=t,
        state=state,
        anchor=at,
        draw_point=draw_point,
        power=0.0,
        fired_power=None,
        scale=scale,
        bow_position_m=bow_m,
        draw_position_m=draw_m,
        knuckle_dir=knuckle,
        render_scale=render_scale,
    )


def settle(mapper, frames=30, **kw):
    """Run the mapper for a while so a One Euro filter converges."""
    pose = None
    for i in range(frames):
        pose = mapper.map(snap(t=(i + 1) * FRAME_MS, **kw))
    return pose


def _aim_hands(yaw_deg: float, baseline: float = 0.40, bow=(0.0, 0.10, 0.45)):
    """A (bow_m, draw_m) pair whose 3D line reads as `yaw_deg` right, at the
    given baseline (PLAN.md §4.7.4's own worked example)."""
    a = (math.sin(math.radians(yaw_deg)), 0.0, math.cos(math.radians(yaw_deg)))
    d = (baseline * a[0], 0.0, -baseline * a[2])
    draw = (bow[0] - d[0], bow[1] - d[1], bow[2] - d[2])
    return bow, draw


def test_mirror_and_scale():
    """Normalized coords scale straight to pixels — capture and window are 16:9."""
    pose = Mapper().map(snap(at=(0.25, 0.75)))
    assert pose.anchor == (0.25 * W, 0.75 * H)


# -- 2D aim vector (unchanged since M4b) -------------------------------------


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


# -- aim_angles() -------------------------------------------------------------


def test_aim_angles_sign_conventions():
    """Bow hand nearer the camera (d.z < 0) gives a.z > 0; bow to the right of
    the draw hand gives yaw > 0; bow above the draw hand gives pitch < 0."""
    bow, draw = _aim_hands(10.0)
    yaw, pitch, baseline = aim_angles(bow, draw)
    assert yaw > 0
    assert abs(math.degrees(yaw) - 10.0) < 1e-6
    assert abs(pitch) < 1e-9
    assert abs(baseline - 0.40) < 1e-9

    bow_above = (0.0, 0.0, 0.45)
    draw_below = (0.0, 0.10, 0.85)  # farther from the camera too, as bow_input gives it
    yaw2, pitch2, _ = aim_angles(bow_above, draw_below)
    assert pitch2 < 0
    assert abs(yaw2) < 1e-9


# -- 3D sight along the arrow (§4.7.4) — acceptance criteria 1, 2, 3, 4, 5, 6, 7


def test_sight_follows_a_ten_degree_aim():
    """Acceptance 1: bow at (0.0, 0.10, 0.45) m, draw hand placed so the
    game-space aim is 10 deg right, 60 DRAWN snapshots 33 ms apart ->
    sight.x = CX + 900*tan(1.5*10deg) +/- 2 px, sight.y = CY +/- 2 px."""
    bow_m, draw_m = _aim_hands(10.0)
    mapper = Mapper()
    pose = None
    for i in range(60):
        pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                t=(i + 1) * FRAME_MS))
    expected_x = CX + config.FOCAL_PX * math.tan(math.radians(config.AIM_GAIN * 10.0))
    assert pose.sight is not None
    assert abs(pose.sight[0] - expected_x) < 2.0
    assert abs(pose.sight[1] - CY) < 2.0


def test_sight_moves_up_when_the_bow_hand_is_above_the_draw_hand():
    """Acceptance 2: the same aim, but with the bow hand above the draw hand
    -> sight.y < CY (up on screen)."""
    a = (math.sin(math.radians(10.0)), 0.0, math.cos(math.radians(10.0)))
    bow_m = (0.0, 0.10, 0.45)
    d = (0.40 * a[0], -0.10, -0.40 * a[2])  # bow.y < draw.y: bow sits above
    draw_m = (bow_m[0] - d[0], bow_m[1] - d[1], bow_m[2] - d[2])
    mapper = Mapper()
    pose = None
    for i in range(60):
        pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                t=(i + 1) * FRAME_MS))
    assert pose.sight is not None
    assert pose.sight[1] < CY


def test_no_sight_while_held_full_weight_at_ten_cm_baseline():
    """Acceptance 3: no sight while HELD; DRAWN at a 0.10 m baseline -> weight 1."""
    held = Mapper().map(snap(state=BowState.HELD))
    assert held.sight is None
    assert held.aim_weight == 0.0

    bow_m = (0.0, 0.0, 0.45)
    draw_m = (0.0, 0.0, 0.45 + 0.10)
    mapper = Mapper()
    pose = None
    for i in range(30):
        pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                t=(i + 1) * FRAME_MS))
    assert abs(pose.aim_weight - 1.0) < 1e-6


def test_weight_ramps_between_the_baseline_floor_and_full_trust():
    """Acceptance 3: a 0.07 m baseline (AIM_BASELINE_MIN_M=0.05, RAMP=0.05) is
    (0.07-0.05)/0.05 = 0.4 of the way up the ramp."""
    bow_m = (0.0, 0.0, 0.45)
    draw_m = (0.0, 0.0, 0.45 + 0.07)
    mapper = Mapper()
    pose = None
    for i in range(30):
        pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                t=(i + 1) * FRAME_MS))
    assert abs(pose.aim_weight - 0.4) < 0.02


def test_released_holds_sight_then_held_clears_it():
    """Acceptance 4: RELEASED holds the previous sight; returning to HELD
    clears it."""
    bow_m, draw_m = _aim_hands(15.0)
    mapper = Mapper()
    pose = None
    for i in range(30):
        pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                t=(i + 1) * FRAME_MS))
    drawn_sight, drawn_weight = pose.sight, pose.aim_weight
    assert drawn_sight is not None and drawn_weight > 0.0

    released = mapper.map(snap(state=BowState.RELEASED, t=31 * FRAME_MS))
    assert released.sight == drawn_sight
    assert released.aim_weight == drawn_weight

    held = mapper.map(snap(state=BowState.HELD, t=32 * FRAME_MS))
    assert held.sight is None
    assert held.aim_weight == 0.0


def test_a_sixty_degree_aim_pins_the_sight_at_the_clamp():
    """Acceptance 5: a 60 deg aim (gained to 90 deg) pins at the screen-margin
    clamp rather than flying off past the window edge."""
    bow_m, draw_m = _aim_hands(60.0)
    mapper = Mapper()
    pose = None
    for i in range(60):
        pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                t=(i + 1) * FRAME_MS))
    yaw_clamp = math.atan(config.AIM_SCREEN_MARGIN * (W / 2.0) / config.FOCAL_PX)
    expected = CX + config.FOCAL_PX * math.tan(yaw_clamp)
    assert abs(pose.sight[0] - expected) < 1.0


def test_extreme_aim_in_both_axes_stays_inside_the_window():
    """The vertical clamp has to account for yaw. The sight projects to
    cy + f*tan(pitch)/cos(yaw), so clamping pitch on its own let the crosshair
    leave the window by up to 53 px once yaw was also extreme — drawn
    off-screen, i.e. invisible exactly when the player aimed hardest."""
    for yaw_deg, pitch_deg in ((60.0, 40.0), (60.0, -40.0), (-60.0, 40.0), (-60.0, -40.0)):
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        baseline = 0.40
        d = (
            baseline * math.cos(pitch) * math.sin(yaw),
            baseline * math.sin(pitch),
            -baseline * math.cos(pitch) * math.cos(yaw),
        )
        bow_m = (0.0, 0.10, 0.45)
        draw_m = (bow_m[0] - d[0], bow_m[1] - d[1], bow_m[2] - d[2])
        mapper = Mapper()
        pose = None
        for i in range(60):
            pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                    t=(i + 1) * FRAME_MS))
        sx, sy = pose.sight
        assert 0.0 <= sx <= W, f"yaw={yaw_deg} pitch={pitch_deg}: sight.x={sx}"
        assert 0.0 <= sy <= H, f"yaw={yaw_deg} pitch={pitch_deg}: sight.y={sy}"


def test_apply_calibration_zeroes_a_known_raw_aim():
    """Acceptance 6: apply_calibration with aim_yaw0_deg equal to a known raw
    aim puts the sight at CX +/- 2 px."""
    bow_m, draw_m = _aim_hands(7.0)
    yaw_known, pitch_known, _ = aim_angles(bow_m, draw_m)

    class _Result:
        aim_yaw0_deg = math.degrees(yaw_known)
        aim_pitch0_deg = math.degrees(pitch_known)

    mapper = Mapper()
    mapper.apply_calibration(_Result())
    pose = None
    for i in range(60):
        pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                t=(i + 1) * FRAME_MS))
    assert abs(pose.sight[0] - CX) < 2.0
    assert abs(pose.sight[1] - CY) < 2.0


def test_bow_forward_and_up():
    """Acceptance 7: bow_forward is (0,0,1) while HELD, converges to the
    gained aim direction while DRAWN, and bow_up stays orthogonal to it;
    knuckle_dir (0,-1) gives bow_up (0,-1,0) while HELD."""
    held = settle(Mapper(), state=BowState.HELD)
    assert all(abs(a - b) < 1e-6 for a, b in zip(held.bow_forward, (0.0, 0.0, 1.0)))
    assert all(abs(a - b) < 1e-6 for a, b in zip(held.bow_up, (0.0, -1.0, 0.0)))

    bow_m, draw_m = _aim_hands(20.0)
    mapper = Mapper()
    pose = None
    for i in range(90):
        pose = mapper.map(snap(state=BowState.DRAWN, bow_m=bow_m, draw_m=draw_m,
                                t=(i + 1) * FRAME_MS, knuckle=(1.0, 0.3)))
    yaw_g = config.AIM_GAIN * math.radians(20.0)
    expected_forward = (math.sin(yaw_g), 0.0, math.cos(yaw_g))
    assert all(abs(a - b) < 1e-3 for a, b in zip(pose.bow_forward, expected_forward))
    dot = sum(a * b for a, b in zip(pose.bow_forward, pose.bow_up))
    assert abs(dot) < 1e-6
