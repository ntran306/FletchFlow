import pytest

from fletchflow import config
from fletchflow.game.world import half_width_m, project, unproject


@pytest.mark.parametrize("point", [(0.0, 0.0, 5.0), (1.5, -0.8, 9.0), (-3.0, 2.0, 20.0)])
def test_project_unproject_round_trip(point):
    projected = project(point)
    assert projected is not None
    back = unproject(projected[0], projected[1], point[2])
    for a, b in zip(back, point):
        assert abs(a - b) < 1e-6


def test_centre_of_screen_is_the_forward_axis():
    cx, cy = config.WINDOW_SIZE[0] / 2, config.WINDOW_SIZE[1] / 2
    x, y, z = unproject(cx, cy, 7.0)
    assert (abs(x), abs(y), z) == (0.0, 0.0, 7.0)


def test_behind_near_plane_does_not_project():
    assert project((0.0, 0.0, config.NEAR_PLANE_M - 0.01)) is None


def test_twice_the_depth_is_half_the_size():
    near = project((0.0, 0.0, 6.0))
    far = project((0.0, 0.0, 12.0))
    assert abs(near[2] / far[2] - 2.0) < 1e-9


def test_visible_width_grows_with_depth():
    assert half_width_m(20.0) > half_width_m(10.0) > 0
