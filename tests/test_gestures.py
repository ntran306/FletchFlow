import numpy as np

from fletchflow import config
from fletchflow.input.gestures import extract
from fletchflow.vision.tracker import HandFrame


def synthetic_hand(pinch_dist: float) -> np.ndarray:
    """21 landmarks; wrist->MCP distance fixed at 0.10, tips pinch_dist apart."""
    points = np.zeros((21, 3), dtype=np.float32)
    points[config.WRIST] = (0.50, 0.60, 0)
    points[config.MIDDLE_MCP] = (0.50, 0.50, 0)  # hand size = 0.10
    points[config.THUMB_TIP] = (0.45, 0.45, 0)
    points[config.INDEX_TIP] = (0.45 + pinch_dist, 0.45, 0)
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
    frame = extract(HandFrame(0, left=synthetic_hand(0.02), right=None))
    assert abs(frame.left.pinch_point[0] - 0.46) < 1e-5
    assert abs(frame.left.pinch_point[1] - 0.45) < 1e-5


def test_missing_hands_pass_through():
    frame = extract(HandFrame(0, left=None, right=synthetic_hand(0.09)))
    assert frame.left is None
    assert frame.right.pinch_ratio > 0.55  # open hand
