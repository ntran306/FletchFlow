import numpy as np

from fletchflow.vision.smoothing import OneEuroFilter


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
