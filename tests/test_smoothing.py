import numpy as np

from fletchflow.vision.smoothing import HandSmoother, OneEuroFilter
from fletchflow.vision.tracker import HandFrame


def make_filter() -> OneEuroFilter:
    return OneEuroFilter(min_cutoff=1.5, beta=0.3, d_cutoff=1.0)


def test_constant_signal_passes_through():
    f = make_filter()
    x = np.array([0.5, 0.5])
    out = x
    for i in range(30):
        out = f(x, t=i / 30)
    assert np.allclose(out, x, atol=1e-9)


def test_noise_is_attenuated_at_rest():
    rng = np.random.default_rng(0)
    f = make_filter()
    raw, filtered = [], []
    for i in range(300):
        x = np.array([0.5 + rng.normal(0.0, 0.01)])
        raw.append(x[0])
        filtered.append(f(x, t=i / 30)[0])
    raw = np.array(raw[30:])       # skip warm-up
    filtered = np.array(filtered[30:])
    # At rest the filter should cut noise std at least in half
    assert filtered.std() < raw.std() * 0.5


def test_bounded_lag_during_steady_motion():
    f = make_filter()
    out = np.array([0.0])
    for i in range(61):
        t = i / 30
        out = f(np.array([t]), t)  # position moves 1.0 units/s
    # Expected lag ≈ v * tau = 1/(2π(min_cutoff + beta·v)) ≈ 0.09 units
    assert abs(out[0] - 2.0) < 0.15


def test_reset_clears_state():
    f = make_filter()
    f(np.array([0.0]), 0.0)
    f(np.array([0.1]), 1 / 30)
    f.reset()
    out = f(np.array([5.0]), 1.0)
    assert out[0] == 5.0  # first sample after reset passes through unfiltered


def test_world_landmarks_pass_through_smooth_unchanged():
    """HandSmoother.smooth() rebuilds HandFrame from its smoothed image
    landmarks; left_world/right_world must survive that rebuild untouched —
    there is no smoothing on world landmarks in this phase (M4c phase 1)."""
    left_world = np.arange(63, dtype=np.float32).reshape(21, 3)
    right_world = -np.arange(63, dtype=np.float32).reshape(21, 3)
    left = np.full((21, 3), 0.5, dtype=np.float32)
    right = np.full((21, 3), 0.4, dtype=np.float32)

    smoother = HandSmoother()
    frame = HandFrame(
        timestamp_ms=0, left=left, right=right,
        left_world=left_world, right_world=right_world,
    )
    out = smoother.smooth(frame)

    assert out.left_world is not None and out.right_world is not None
    np.testing.assert_array_equal(out.left_world, left_world)
    np.testing.assert_array_equal(out.right_world, right_world)

    # A second call (now with prior state) still leaves world untouched, and
    # a missing hand's world array comes through as None.
    frame2 = HandFrame(
        timestamp_ms=33, left=left, right=None,
        left_world=left_world, right_world=None,
    )
    out2 = smoother.smooth(frame2)
    np.testing.assert_array_equal(out2.left_world, left_world)
    assert out2.right_world is None
