"""Tests for input/hand_pose.py — the M4c phase-1 metric pose fit (PLAN.md §4.7.2).

Ports the synthetic-hand generator from the phase-1 prototype
(scratchpad/proto_pose.py) that produced PLAN.md §4.7.2's accuracy numbers:
a canonical right hand's 5 palm points, rotated and projected with FULL
perspective, compared against the weak-perspective Procrustes fit under test.
Reusing that generator is what lets the noiseless numbers below reproduce the
prototype's almost exactly (see test_accuracy_grid_matches_prototype).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fletchflow import config
from fletchflow.input.gestures import extract
from fletchflow.input.hand_pose import estimate_hand_pose
from fletchflow.vision.tracker import HandFrame

W, H = config.CAPTURE_SIZE
F = config.CAM_FOCAL_NORM * W
PALM = config.POSE_PALM_POINTS  # (0, 5, 9, 13, 17): wrist + the four MCPs

# Canonical right hand, metres, hand frame: +y toward fingers is -y (up),
# knuckles spread along x, slight arch in z. Identical to the prototype's
# LOCAL so the measured accuracy numbers reproduce.
LOCAL = {
    config.WRIST: (0.000, 0.000, 0.000),
    config.INDEX_MCP: (-0.026, -0.086, -0.006),
    config.MIDDLE_MCP: (0.000, -0.090, -0.009),
    config.RING_MCP: (0.020, -0.086, -0.007),
    config.PINKY_MCP: (0.038, -0.077, -0.003),
}

Z_GRID = (0.35, 0.45, 0.6, 0.8, 1.0)
ROLLS = range(-90, 91, 15)
PITCHES_YAWS = (-20, 0, 20)


def rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = map(math.radians, (roll, pitch, yaw))
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return Rz @ Rx @ Ry


def synth(
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    x_off: float = 0.05,
    y_off: float = 0.08,
    img_noise: float = 0.0,
    world_noise: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """Full-perspective synthetic hand.

    Returns (img_px, world_m, z_true): the 5 palm points' projected pixel
    coordinates, their hand-centred camera-aligned world coordinates
    (metres — what hand_world_landmarks would report), and the true camera
    depth of their centroid.
    """
    R = rot(roll, pitch, yaw)
    local = np.array([LOCAL[i] for i in PALM])
    centred = local - local.mean(axis=0)
    world = centred @ R.T                                     # camera-aligned, hand-centred
    cam = world + np.array([x_off, y_off, z])                 # camera space, metres
    img = np.column_stack([
        W / 2 + F * cam[:, 0] / cam[:, 2],
        H / 2 + F * cam[:, 1] / cam[:, 2],
    ])                                                         # full perspective, px
    if img_noise:
        img = img + rng.normal(0, img_noise, img.shape)
    if world_noise:
        world = world + rng.normal(0, world_noise, world.shape)
    return img, world, float(cam.mean(axis=0)[2])


def _arrays(img_px: np.ndarray, world_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack the 5 palm points into (21,3) arrays: image normalized to [0,1],
    world in metres. The rest of the 21 landmarks are left zero — the fit
    only reads config.POSE_PALM_POINTS."""
    image_pts = np.zeros((21, 3), dtype=np.float64)
    world_pts = np.zeros((21, 3), dtype=np.float64)
    for row, idx in enumerate(PALM):
        image_pts[idx, 0] = img_px[row, 0] / W
        image_pts[idx, 1] = img_px[row, 1] / H
        world_pts[idx, :] = world_m[row, :]
    return image_pts, world_pts


def _wrist_grip_norm(img_px: np.ndarray) -> tuple[float, float]:
    """PALM[0] is the wrist; its projected position stands in for grip_norm.

    Depth, residual and theta never depend on grip_norm, only position_m
    does (tested separately), so any point works here.
    """
    return (float(img_px[0, 0] / W), float(img_px[0, 1] / H))


# ---------------------------------------------------------------------------
# 1. Accuracy grid (acceptance 1)
# ---------------------------------------------------------------------------

def test_accuracy_grid_matches_prototype():
    """Noiseless, |pitch|,|yaw| <= 20 deg, roll +-90 deg, across z.

    Prototype measured worst 13.6/10.4/7.7/5.8/4.6% and median
    4.1/3.2/2.4/1.8/1.4% at z = 0.35/0.45/0.6/0.8/1.0 m.
    """
    worst_cap = {0.35: 0.15, 0.45: 0.15, 0.6: 0.09, 0.8: 0.09, 1.0: 0.09}
    median_cap = 0.05

    for z in Z_GRID:
        errors = []
        for roll in ROLLS:
            for pitch in PITCHES_YAWS:
                for yaw in PITCHES_YAWS:
                    img_px, world_m, z_true = synth(z, roll, pitch, yaw)
                    image_pts, world_pts = _arrays(img_px, world_m)
                    grip = _wrist_grip_norm(img_px)
                    pose = estimate_hand_pose(image_pts, world_pts, grip)
                    assert pose is not None, (z, roll, pitch, yaw)
                    errors.append(abs(pose.depth_m - z_true) / z_true)
        errors.sort()
        worst = errors[-1]
        median = errors[len(errors) // 2]
        print(f"z={z:.2f}  worst={worst * 100:.1f}%  median={median * 100:.1f}%  n={len(errors)}")
        assert worst <= worst_cap[z], f"z={z}: worst {worst:.3%} > cap {worst_cap[z]:.0%}"
        assert median <= median_cap, f"z={z}: median {median:.3%} > cap {median_cap:.0%}"


# ---------------------------------------------------------------------------
# 2. Rotation swing at a fixed depth (acceptance 2)
# ---------------------------------------------------------------------------

def test_rotation_swing_at_fixed_depth():
    """Same grid as above, fixed z=0.45 m: max/min estimated depth <= x1.25
    (prototype x1.19)."""
    depths = []
    for roll in ROLLS:
        for pitch in PITCHES_YAWS:
            for yaw in PITCHES_YAWS:
                img_px, world_m, _ = synth(0.45, roll, pitch, yaw)
                image_pts, world_pts = _arrays(img_px, world_m)
                grip = _wrist_grip_norm(img_px)
                pose = estimate_hand_pose(image_pts, world_pts, grip)
                assert pose is not None
                depths.append(pose.depth_m)
    swing = max(depths) / min(depths)
    print(f"rotation swing x{swing:.3f}")
    assert swing <= 1.25, f"rotation swing x{swing:.3f} > x1.25 cap"


# ---------------------------------------------------------------------------
# 3. Noise (acceptance 3)
# ---------------------------------------------------------------------------

def test_noise_median_depth_error():
    """1.5 px image + 4 mm world noise, 400 random poses per z, seeded.

    A residual-gate rejection is an expected outcome of noise at close range
    (the weak-perspective approximation is worst there — see test 1), not a
    bug, so rejected frames are excluded from the median the same way the
    caller would just hold the last accepted pose. Prototype (which has no
    rejection gate at all) measured median 6.1/5.1/3.8% at z=0.35/0.6/1.0.
    """
    caps = {0.35: 0.07, 0.6: 0.07, 1.0: 0.07}
    for z, cap in caps.items():
        rng = np.random.default_rng(5)
        errors = []
        n_reject = 0
        for _ in range(400):
            roll = rng.uniform(-90, 90)
            pitch = rng.uniform(-40, 40)
            yaw = rng.uniform(-40, 40)
            img_px, world_m, z_true = synth(
                z, roll, pitch, yaw, img_noise=1.5, world_noise=0.004, rng=rng
            )
            image_pts, world_pts = _arrays(img_px, world_m)
            grip = _wrist_grip_norm(img_px)
            pose = estimate_hand_pose(image_pts, world_pts, grip)
            if pose is None:
                n_reject += 1
                continue
            errors.append(abs(pose.depth_m - z_true) / z_true)
        errors.sort()
        median = errors[len(errors) // 2]
        print(f"z={z:.2f}  median={median * 100:.1f}%  rejects={n_reject}/400")
        assert median <= cap, f"z={z}: median {median:.3%} > cap {cap:.0%}"


# ---------------------------------------------------------------------------
# 4. knuckle_dir (acceptance 4)
# ---------------------------------------------------------------------------

def test_knuckle_dir_roll_error_via_extract():
    """Through gestures.extract() on a full 21-landmark HandFrame: roll error
    (true vs. estimated in-image knuckle-line angle) p95 <= 4 deg at 1.5 px
    image noise, 500 seeded poses. Prototype measured p95 3.2 deg.
    """
    index_row = PALM.index(config.INDEX_MCP)
    pinky_row = PALM.index(config.PINKY_MCP)

    rng = np.random.default_rng(7)
    errors = []
    for _ in range(500):
        z = rng.uniform(0.35, 1.0)
        roll = rng.uniform(-90, 90)
        pitch = rng.uniform(-40, 40)
        yaw = rng.uniform(-40, 40)

        img_px, world_m, _ = synth(z, roll, pitch, yaw, img_noise=1.5, rng=rng)
        image_pts, world_pts = _arrays(img_px, world_m)
        frame = HandFrame(
            timestamp_ms=0,
            left=image_pts.astype(np.float32),
            right=None,
            left_world=world_pts.astype(np.float32),
            right_world=None,
        )
        kd = extract(frame).left.knuckle_dir

        # True (noiseless) in-image knuckle-line angle, same pose.
        img_true, _, _ = synth(z, roll, pitch, yaw)
        v_true = (
            img_true[index_row, 0] - img_true[pinky_row, 0],
            img_true[index_row, 1] - img_true[pinky_row, 1],
        )
        true_angle = math.degrees(math.atan2(v_true[1], v_true[0]))
        est_angle = math.degrees(math.atan2(kd[1], kd[0]))
        err = abs(((est_angle - true_angle + 180) % 360) - 180)
        errors.append(err)

    errors.sort()
    p95 = errors[int(0.95 * len(errors))]
    print(f"knuckle_dir roll error p50={errors[len(errors) // 2]:.2f} deg  p95={p95:.2f} deg")
    assert p95 <= 4.0, f"knuckle_dir roll error p95 {p95:.2f} deg > 4 deg cap"


def test_knuckle_dir_thumb_up_reads_up():
    """A hand with the index knuckle directly above the pinky knuckle in the
    image (thumb-up fist) gives knuckle_dir approx (0, -1): up."""
    points = np.zeros((21, 3), dtype=np.float32)
    points[config.WRIST] = (0.50, 0.60, 0.0)
    points[config.INDEX_MCP] = (0.50, 0.45, 0.0)   # smaller y = higher in image
    points[config.PINKY_MCP] = (0.50, 0.55, 0.0)
    frame = HandFrame(timestamp_ms=0, left=points, right=None)

    kd = extract(frame).left.knuckle_dir
    assert kd[0] == pytest.approx(0.0, abs=1e-6)
    assert kd[1] == pytest.approx(-1.0, abs=1e-6)


def test_knuckle_dir_degenerate_falls_back_to_up():
    points = np.zeros((21, 3), dtype=np.float32)
    points[config.INDEX_MCP] = (0.5, 0.5, 0.0)
    points[config.PINKY_MCP] = (0.5, 0.5, 0.0)
    frame = HandFrame(timestamp_ms=0, left=points, right=None)

    assert extract(frame).left.knuckle_dir == (0.0, -1.0)


# ---------------------------------------------------------------------------
# 5. Mirroring (acceptance 5)
# ---------------------------------------------------------------------------

def test_mirroring_gives_same_depth():
    img_px, world_m, _ = synth(0.6, 25.0, -10.0, 15.0)
    image_pts, world_pts = _arrays(img_px, world_m)
    grip = _wrist_grip_norm(img_px)
    pose = estimate_hand_pose(image_pts, world_pts, grip)
    assert pose is not None

    mirrored_image = image_pts.copy()
    mirrored_image[:, 0] = 1.0 - mirrored_image[:, 0]
    mirrored_world = world_pts.copy()
    mirrored_world[:, 0] = -mirrored_world[:, 0]
    mirrored_grip = (1.0 - grip[0], grip[1])

    mirrored_pose = estimate_hand_pose(mirrored_image, mirrored_world, mirrored_grip)
    assert mirrored_pose is not None
    assert abs(mirrored_pose.depth_m - pose.depth_m) < 1e-6


# ---------------------------------------------------------------------------
# 6. Rejections (acceptance 6)
# ---------------------------------------------------------------------------

def _valid_arrays():
    img_px, world_m, _ = synth(0.5, 10.0, 5.0, -5.0)
    return _arrays(img_px, world_m)


def test_rejects_world_none():
    image_pts, _ = _valid_arrays()
    assert estimate_hand_pose(image_pts, None, (0.5, 0.5)) is None


def test_rejects_nan_input():
    image_pts, world_pts = _valid_arrays()
    bad = image_pts.copy()
    bad[config.WRIST, 0] = float("nan")
    assert estimate_hand_pose(bad, world_pts, (0.5, 0.5)) is None


def test_rejects_all_coincident_points():
    image_pts = np.zeros((21, 3), dtype=np.float64)
    world_pts = np.zeros((21, 3), dtype=np.float64)
    for idx in PALM:
        image_pts[idx, 0] = 0.5
        image_pts[idx, 1] = 0.5
        # world_pts stays all zero: every palm point at the same 3D point too
    assert estimate_hand_pose(image_pts, world_pts, (0.5, 0.5)) is None


def test_rejects_depth_outside_range():
    lo, hi = config.POSE_DEPTH_RANGE_M
    img_px, world_m, z_true = synth(hi + 1.0, 0.0, 0.0, 0.0)
    assert z_true > hi  # sanity: this pose really is out of range
    image_pts, world_pts = _arrays(img_px, world_m)
    grip = _wrist_grip_norm(img_px)
    assert estimate_hand_pose(image_pts, world_pts, grip) is None


def test_rejects_high_residual():
    """Scramble the world points' correspondence among the 5 palm indices:
    the fit tries to explain the (unrelated) shapes with one similarity
    transform and the residual blows well past POSE_MAX_RESIDUAL_PX."""
    img_px, world_m, _ = synth(0.5, 10.0, 0.0, 0.0)
    image_pts, world_pts = _arrays(img_px, world_m)

    scrambled = world_pts.copy()
    order = list(PALM)
    reversed_order = list(reversed(order))
    for src, dst in zip(order, reversed_order):
        scrambled[dst] = world_pts[src]

    grip = _wrist_grip_norm(img_px)
    assert estimate_hand_pose(image_pts, scrambled, grip) is None


# ---------------------------------------------------------------------------
# 7. Position (acceptance 7)
# ---------------------------------------------------------------------------

def test_position_grip_at_image_centre():
    img_px, world_m, _ = synth(0.5, 5.0, 0.0, 0.0)
    image_pts, world_pts = _arrays(img_px, world_m)
    pose = estimate_hand_pose(image_pts, world_pts, (0.5, 0.5))
    assert pose is not None
    assert pose.position_m[0] == pytest.approx(0.0, abs=1e-6)
    assert pose.position_m[1] == pytest.approx(0.0, abs=1e-6)


def test_position_grip_offset_right():
    img_px, world_m, _ = synth(0.5, 5.0, 0.0, 0.0)
    image_pts, world_pts = _arrays(img_px, world_m)
    pose = estimate_hand_pose(image_pts, world_pts, (0.75, 0.5))
    assert pose is not None
    assert pose.position_m[0] > 0

    f_px = config.CAM_FOCAL_NORM * W
    expected_x = (0.25 * W) * pose.depth_m / f_px
    assert pose.position_m[0] == pytest.approx(expected_x, rel=1e-9)


# ---------------------------------------------------------------------------
# 8. gestures.extract integration (acceptance 8)
# ---------------------------------------------------------------------------

def test_extract_pose_present_with_world_and_none_without():
    img_px, world_m, _ = synth(0.5, 10.0, 0.0, 0.0)
    image_pts, world_pts = _arrays(img_px, world_m)
    image_pts32 = image_pts.astype(np.float32)
    world_pts32 = world_pts.astype(np.float32)

    with_world = extract(HandFrame(
        timestamp_ms=0, left=image_pts32, right=None,
        left_world=world_pts32, right_world=None,
    )).left
    without_world = extract(HandFrame(
        timestamp_ms=0, left=image_pts32, right=None,
    )).left

    assert with_world.pose is not None
    assert without_world.pose is None

    # Every pre-existing HandGesture field is unchanged whether or not world
    # landmarks were supplied.
    assert with_world.wrist == without_world.wrist
    assert with_world.pinch_point == without_world.pinch_point
    assert with_world.grip_point == without_world.grip_point
    assert with_world.pinch_ratio == without_world.pinch_ratio
    assert with_world.fist_ratio == without_world.fist_ratio
    assert with_world.size == without_world.size
    assert with_world.palm_size == without_world.palm_size
    assert with_world.knuckle_dir == without_world.knuckle_dir
