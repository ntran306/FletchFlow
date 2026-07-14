"""FletchFlow entry point.

Milestone 1: mirrored webcam feed with both hands' landmarks drawn on top,
One-Euro-smoothed, with handedness labels (L cyan / R orange).
Run with `fletchflow` (after `pip install -e .`) or `python -m fletchflow`.
Press ESC or close the window to quit.

Self-check mode (`fletchflow --selfcheck [seconds]`) runs the identical
pipeline headless: no window opens, rendered frames are saved as PNGs once
per second, and a stats verdict prints at the end. This lets the pipeline be
verified end-to-end — including what actually got drawn — without a display.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import pygame

from fletchflow import config
from fletchflow.render.hud import draw_hands
from fletchflow.vision.camera import Camera
from fletchflow.vision.pipeline import TrackingPipeline


def frame_to_surface(image_bgr, size: tuple[int, int], mirror: bool) -> pygame.Surface:
    if mirror:
        image_bgr = cv2.flip(image_bgr, 1)
    image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(image_rgb.tobytes(), size, "RGB")


class BackgroundCache:
    """Convert each camera frame to a pygame surface only once.

    The render loop runs at 60 fps but frames arrive at ~30-40; without this
    the flip/resize/convert/copy chain (several ms at 720p) runs twice per
    frame, stealing CPU that detection needs.
    """

    def __init__(self) -> None:
        self._timestamp = -1.0
        self._surface: pygame.Surface | None = None

    def get(self, frame) -> pygame.Surface | None:
        if frame is None:
            return None
        if frame.timestamp != self._timestamp:
            self._timestamp = frame.timestamp
            self._surface = frame_to_surface(
                frame.image, config.WINDOW_SIZE, config.MIRROR
            )
        return self._surface


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fletchflow")
    parser.add_argument(
        "--selfcheck",
        nargs="?",
        const=6.0,
        default=None,
        type=float,
        metavar="SECONDS",
        help="run headless for N seconds, saving one rendered frame per second",
    )
    parser.add_argument(
        "--out",
        default="selfcheck_frames",
        help="directory for --selfcheck frame PNGs",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    if not config.MODEL_PATH.exists():
        print(
            f"error: model not found at {config.MODEL_PATH}\n"
            "Download it first — see README.md, Setup section.",
            file=sys.stderr,
        )
        return 1

    camera = Camera(
        config.CAMERA_INDEX,
        *config.CAPTURE_SIZE,
        fps=config.CAPTURE_FPS,
        manual_exposure=config.MANUAL_EXPOSURE,
    )
    try:
        camera.start()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    pipeline = TrackingPipeline(camera)
    pipeline.start()

    pygame.init()
    screen = pygame.display.set_mode(config.WINDOW_SIZE)
    pygame.display.set_caption("FletchFlow")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    background_cache = BackgroundCache()

    start_time = time.perf_counter()
    next_save = start_time + 1.0
    saved = 0
    hands_seen = 0
    ticks = 0

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False

            background = background_cache.get(camera.latest())
            if background is not None:
                screen.blit(background, (0, 0))
            else:
                screen.fill((20, 20, 20))

            hand_frame = pipeline.latest()
            draw_hands(screen, hand_frame, font)

            if config.SHOW_FPS:
                hands = (
                    int(hand_frame.left is not None) + int(hand_frame.right is not None)
                    if hand_frame
                    else 0
                )
                text = (
                    f"render {clock.get_fps():5.1f} fps | camera {camera.fps:5.1f} fps"
                    f" | tracker {pipeline.fps:5.1f} fps @ {pipeline.ms:4.1f} ms"
                    f" | hands {hands}"
                )
                screen.blit(font.render(text, True, (0, 255, 128)), (10, 10))
                if camera.fps and camera.fps < 20:
                    warning = "low camera fps — improve lighting for responsive tracking"
                    screen.blit(font.render(warning, True, (255, 210, 0)), (10, 34))

            pygame.display.flip()
            clock.tick(config.TARGET_FPS)

            if args.selfcheck:
                ticks += 1
                if hand_frame and (
                    hand_frame.left is not None or hand_frame.right is not None
                ):
                    hands_seen += 1
                now = time.perf_counter()
                if now >= next_save:
                    pygame.image.save(screen, str(out_dir / f"frame_{saved:02d}.png"))
                    saved += 1
                    next_save += 1.0
                if now - start_time >= args.selfcheck:
                    running = False
    finally:
        pipeline.stop()
        camera.stop()
        pygame.quit()

    if args.selfcheck:
        ok = camera.fps >= 25 and pipeline.fps >= 25
        print(
            f"selfcheck: render {clock.get_fps():.1f} fps | camera {camera.fps:.1f} fps"
            f" | tracker {pipeline.fps:.1f} fps @ {pipeline.ms:.1f} ms"
            f" | ticks with hands {hands_seen}/{ticks}"
            f" | {saved} frames -> {out_dir}"
        )
        print("SELFCHECK OK" if ok else "SELFCHECK FAIL: camera or tracker below 25 fps")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
