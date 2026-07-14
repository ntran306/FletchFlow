"""HUD and debug drawing. Milestone 1: hand landmark overlay."""

from __future__ import annotations

import pygame

from fletchflow import config
from fletchflow.vision.tracker import HandFrame

# Standard MediaPipe 21-landmark hand skeleton
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
)

LEFT_COLOR = (80, 200, 255)   # cyan — player's left hand
RIGHT_COLOR = (255, 170, 60)  # orange — player's right hand


def draw_hands(
    surface: pygame.Surface, hand_frame: HandFrame | None, font: pygame.font.Font
) -> None:
    if hand_frame is None:
        return
    for points, color, label in (
        (hand_frame.left, LEFT_COLOR, "L"),
        (hand_frame.right, RIGHT_COLOR, "R"),
    ):
        if points is not None:
            _draw_hand(surface, points, color, label, font)


def _draw_hand(surface, points, color, label, font) -> None:
    w, h = config.WINDOW_SIZE
    # pygame rejects numpy scalars as coordinates — convert to Python floats
    px = (points[:, 0] * w).tolist()
    py = (points[:, 1] * h).tolist()

    dim = tuple(c // 2 for c in color)
    for a, b in HAND_CONNECTIONS:
        pygame.draw.line(surface, dim, (px[a], py[a]), (px[b], py[b]), 2)
    for i in range(21):
        # The four landmarks the game actually uses get bigger, brighter dots
        radius = 7 if i in config.KEY_LANDMARKS else 3
        pygame.draw.circle(surface, color, (px[i], py[i]), radius)

    wrist = config.WRIST
    surface.blit(font.render(label, True, color), (px[wrist] + 12, py[wrist] + 12))
