"""AsyncFrameWriter, mean_rate, and the new frame_to_surface — the pieces
behind the --selfcheck verdict fix.

Coverage gap this closes: nothing exercised the PNG-writer's color handling
(an RGB/BGR swap would silently corrupt every selfcheck screenshot) or the
mean-rate math the verdict now depends on for pass/fail.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from fletchflow.diagnostics import AsyncFrameWriter, mean_rate  # noqa: E402
from fletchflow.__main__ import frame_to_surface  # noqa: E402

pygame.init()


def test_mean_rate():
    assert mean_rate(10, 1.0, 40, 2.0) == 30.0
    assert mean_rate(10, 1.0, 40, 1.0) == 0.0    # zero interval
    assert mean_rate(10, 2.0, 40, 1.0) == 0.0    # negative interval
    assert mean_rate(40, 1.0, 40, 2.0) == 0.0    # non-increasing count (equal)
    assert mean_rate(40, 1.0, 10, 2.0) == 0.0    # non-increasing count (decreasing)


def test_async_writer_writes_correct_colors(tmp_path):
    surface = pygame.Surface((64, 32))
    surface.fill((255, 0, 0), pygame.Rect(0, 0, 32, 32))   # left half pure red
    surface.fill((0, 0, 255), pygame.Rect(32, 0, 32, 32))  # right half pure blue

    writer = AsyncFrameWriter()
    out_path = tmp_path / "frame.png"
    writer.submit(surface, str(out_path))
    writer.close()

    img = cv2.imread(str(out_path))
    assert img is not None
    assert tuple(int(c) for c in img[16, 8]) == (0, 0, 255)   # red pixel -> BGR
    assert tuple(int(c) for c in img[16, 56]) == (255, 0, 0)  # blue pixel -> BGR


def test_async_writer_drops_instead_of_blocking(tmp_path):
    writer = AsyncFrameWriter(max_pending=1)
    surface = pygame.Surface((64, 32))

    start = time.perf_counter()
    for i in range(50):
        writer.submit(surface, str(tmp_path / f"frame_{i:02d}.png"))
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0

    writer.close()  # must not hang


def test_close_is_idempotent(tmp_path):
    writer = AsyncFrameWriter()
    writer.submit(pygame.Surface((8, 8)), str(tmp_path / "a.png"))
    writer.close()
    writer.close()  # must not raise or hang


def test_worker_survives_a_failed_write_and_close_returns(tmp_path):
    """A write that raises must not kill the worker — a dead worker stops
    draining the queue, and close() runs in main()'s finally block, so the game
    would hang on exit instead of quitting."""
    writer = AsyncFrameWriter()
    surface = pygame.Surface((16, 8))
    surface.fill((0, 255, 0))
    writer.submit(surface, str(tmp_path / "bad.notanimage"))  # cv2 raises
    writer.submit(surface, str(tmp_path / "missing" / "x.png"))  # cv2 returns False
    good = tmp_path / "good.png"
    writer.submit(surface, str(good))

    start = time.perf_counter()
    writer.close(timeout=5.0)
    assert time.perf_counter() - start < 5.0
    assert writer.failed == 2
    assert cv2.imread(str(good)) is not None, "worker died before the good write"


def test_frame_to_surface_mirrors_and_keeps_colors():
    pygame.display.set_mode((1280, 720))
    image_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    image_bgr[:, :640] = (255, 0, 0)  # pure blue in BGR, left half
    image_bgr[:, 640:] = (0, 0, 255)  # pure red in BGR, right half

    surf = frame_to_surface(image_bgr, (1280, 720), mirror=True)

    # Mirrored: the original right (red) half is now on the left, and
    # the original left (blue) half is now on the right.
    assert surf.get_at((100, 360))[:3] == (255, 0, 0)
    assert surf.get_at((1180, 360))[:3] == (0, 0, 255)
    assert surf.get_bitsize() == pygame.display.get_surface().get_bitsize()
