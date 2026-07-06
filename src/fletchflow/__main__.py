"""FletchFlow entry point.

Milestone 0: mirrored webcam feed rendered in a pygame window with FPS readouts.
Run with `fletchflow` (after `pip install -e .`) or `python -m fletchflow`.
Press ESC or close the window to quit.
"""

from __future__ import annotations

import sys

import cv2
import pygame

from fletchflow import config
from fletchflow.vision.camera import Camera


def frame_to_surface(image_bgr, size: tuple[int, int], mirror: bool) -> pygame.Surface:
    if mirror:
        image_bgr = cv2.flip(image_bgr, 1)
    image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(image_rgb.tobytes(), size, "RGB")


def main() -> int:
    camera = Camera(config.CAMERA_INDEX, *config.CAPTURE_SIZE)
    try:
        camera.start()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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

            if config.SHOW_FPS:
                text = f"render {clock.get_fps():5.1f} fps | camera {camera.fps:5.1f} fps"
                screen.blit(font.render(text, True, (0, 255, 128)), (10, 10))

            pygame.display.flip()
            clock.tick(config.TARGET_FPS)
    finally:
        camera.stop()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
