import numpy as np

from fletchflow import config
from fletchflow.input.gestures import extract
from fletchflow.vision.tracker import HandFrame

WRIST_XY = (0.50, 0.60)
PALM_LEN = 0.10   # wrist -> middle MCP


def synthetic_hand(pinch_dist: float = 0.02, curl: float = 2.0) -> np.ndarray:
    """21 landmarks laid out so every derived measurement is exact.

    Each fingertip sits at `wrist + curl * (mcp - wrist)`, so fist_ratio is
    exactly `curl` — 2.0 for an open hand, ~0.9 for a closed fist. The thumb tip
    is then placed `pinch_dist` from the index tip, giving a pinch_ratio of
    pinch_dist / PALM_LEN.
    """
    points = np.zeros((21, 3), dtype=np.float32)
    wx, wy = WRIST_XY
    points[config.WRIST] = (wx, wy, 0)

    # Knuckle row, PALM_LEN above the wrist. Index->pinky span is
    # PALM_WIDTH_RATIO * PALM_LEN so palm_size agrees with the palm length.
    half = PALM_LEN * config.PALM_WIDTH_RATIO / 2.0
    mcp_y = wy - PALM_LEN
    for idx, offset in (
        (config.INDEX_MCP, -half),
        (config.MIDDLE_MCP, 0.0),
        (config.RING_MCP, half * 0.5),
        (config.PINKY_MCP, half),
    ):
        points[idx] = (wx + offset, mcp_y, 0)
    # MIDDLE_MCP must sit exactly PALM_LEN from the wrist
    points[config.MIDDLE_MCP] = (wx, mcp_y, 0)

    for tip, mcp in config.FINGER_PAIRS:
        mx, my = points[mcp, 0], points[mcp, 1]
        points[tip] = (wx + curl * (mx - wx), wy + curl * (my - wy), 0)

    ix, iy = points[config.INDEX_TIP, 0], points[config.INDEX_TIP, 1]
    points[config.THUMB_TIP] = (ix - pinch_dist, iy, 0)
    return points


def test_pinch_ratio_scale_invariant():
    near = extract(HandFrame(0, left=synthetic_hand(0.02), right=None))
    assert abs(near.left.pinch_ratio - 0.2) < 1e-5

    # Same hand twice as far from the camera: all distances halved
    far_points = synthetic_hand(0.02)
    far_points[:, :2] = 0.5 + (far_points[:, :2] - 0.5) * 0.5
    far = extract(HandFrame(0, left=far_points, right=None))
    assert abs(far.left.pinch_ratio - near.left.pinch_ratio) < 1e-5


def test_pinch_point_is_tip_midpoint():
    points = synthetic_hand(0.02)
    frame = extract(HandFrame(0, left=points, right=None))
    expected_x = (points[config.THUMB_TIP, 0] + points[config.INDEX_TIP, 0]) / 2
    assert abs(frame.left.pinch_point[0] - expected_x) < 1e-5


def test_missing_hands_pass_through():
    frame = extract(HandFrame(0, left=None, right=synthetic_hand(0.09)))
    assert frame.left is None
    assert frame.right.pinch_ratio > config.PINCH_OFF  # open hand


def test_extract_populates_size():
    frame = extract(HandFrame(0, left=synthetic_hand(0.02), right=None))
    assert abs(frame.left.size - PALM_LEN) < 1e-5


def test_fist_ratio_open_vs_closed():
    """fist_ratio equals the curl factor by construction."""
    open_hand = extract(HandFrame(0, left=synthetic_hand(curl=2.0), right=None))
    assert abs(open_hand.left.fist_ratio - 2.0) < 1e-4
    assert open_hand.left.fist_ratio > config.FIST_OFF

    fist = extract(HandFrame(0, left=synthetic_hand(curl=0.9), right=None))
    assert abs(fist.left.fist_ratio - 0.9) < 1e-4
    assert fist.left.fist_ratio < config.FIST_ON


def test_fist_ratio_is_scale_invariant():
    near = extract(HandFrame(0, left=synthetic_hand(curl=0.9), right=None))
    far_points = synthetic_hand(curl=0.9)
    far_points[:, :2] = 0.5 + (far_points[:, :2] - 0.5) * 0.4
    far = extract(HandFrame(0, left=far_points, right=None))
    assert abs(far.left.fist_ratio - near.left.fist_ratio) < 1e-4


def test_palm_size_survives_foreshortening():
    """Tilting the hand foreshortens wrist->MCP but not the knuckle row.

    `size` collapses; `palm_size` holds, which is what the 3D draw depends on —
    a false size would read as false depth and move the power bar on its own.
    """
    points = synthetic_hand()
    baseline = extract(HandFrame(0, left=points, right=None)).left

    tilted = points.copy()
    wy = WRIST_XY[1]
    for idx in (config.INDEX_MCP, config.MIDDLE_MCP, config.RING_MCP,
                config.PINKY_MCP):
        tilted[idx, 1] = wy - PALM_LEN * 0.5  # palm length halved
    shortened = extract(HandFrame(0, left=tilted, right=None)).left

    assert abs(shortened.size - PALM_LEN * 0.5) < 1e-5      # size collapses
    assert abs(shortened.palm_size - baseline.palm_size) < 5e-3  # palm_size holds


def test_grip_point_sits_between_wrist_and_knuckles():
    frame = extract(HandFrame(0, left=synthetic_hand(), right=None))
    grip = frame.left.grip_point
    wx, wy = WRIST_XY
    assert abs(grip[0] - wx) < 1e-5
    assert abs(grip[1] - (wy - PALM_LEN * config.GRIP_PALM_FRACTION)) < 1e-5
