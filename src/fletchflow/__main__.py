"""FletchFlow entry point.

Milestone 1: mirrored webcam feed with both hands' landmarks drawn on top,
One-Euro-smoothed, with handedness labels (L cyan / R orange).
Run with `fletchflow` (after `pip install -e .`) or `python -m fletchflow`.
Press ESC or close the window to quit.
"""

from __future__ import annotations

import sys

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


def main() -> int:
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

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False

            frame = camera.latest()
            if frame is not None:
                screen.blit(
                    frame_to_surface(frame.image, config.WINDOW_SIZE, config.MIRROR),
                    (0, 0),
                )
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
    finally:
        pipeline.stop()
        camera.stop()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
