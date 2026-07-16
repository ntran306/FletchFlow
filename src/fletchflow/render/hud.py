"""HUD and debug drawing: landmark overlay, power bar, state readout."""

from __future__ import annotations

import pygame

from fletchflow import config
from fletchflow.input.bow_input import BowState
from fletchflow.input.gestures import GestureFrame
from fletchflow.input.mapping import BowPose
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


def draw_grab_prompt(
    surface: pygame.Surface, font: pygame.font.Font, pose: BowPose | None, t: float
) -> None:
    """Pulsing 'Grab the bow!' while the bow waits at its dock."""
    if pose is not None and pose.state != BowState.DOCKED:
        return
    import math

    dock_x = config.DOCK_POS[0] * config.WINDOW_SIZE[0]
    dock_y = config.DOCK_POS[1] * config.WINDOW_SIZE[1]
    text = font.render("Grab the bow!", True, (255, 240, 180))
    text.set_alpha(int(170 + 85 * math.sin(t * 4.0)))
    surface.blit(
        text, (dock_x - text.get_width() // 2, dock_y - 150)
    )


def draw_power_bar(surface: pygame.Surface, pose: BowPose | None) -> None:
    if pose is None or pose.state == BowState.DOCKED:
        return
    w, h = config.WINDOW_SIZE
    bar_w, bar_h = 320, 16
    x, y = (w - bar_w) // 2, h - 44

    pygame.draw.rect(surface, (24, 24, 28), (x, y, bar_w, bar_h), border_radius=8)
    if pose.power > 0:
        # green -> amber -> red as power rises
        t = pose.power
        color = (
            int(80 + 175 * t),
            int(200 - 90 * t),
            60,
        )
        pygame.draw.rect(
            surface, color,
            (x + 2, y + 2, int((bar_w - 4) * t), bar_h - 4),
            border_radius=6,
        )
    pygame.draw.rect(surface, (200, 200, 210), (x, y, bar_w, bar_h), 2, border_radius=8)


def draw_fire_flash(
    surface: pygame.Surface, font: pygame.font.Font, power: float, age_ms: float
) -> None:
    """Placeholder release feedback until arrows exist (milestone 4)."""
    if age_ms > 600:
        return
    w = config.WINDOW_SIZE[0]
    text = font.render(f"FIRE!  power {power * 100:3.0f}%", True, (255, 235, 120))
    surface.blit(text, (w // 2 - text.get_width() // 2, 80))


def draw_debug_state(
    surface: pygame.Surface,
    font: pygame.font.Font,
    gesture_frame: GestureFrame | None,
    pose: BowPose | None,
) -> None:
    """F1 overlay: live numbers for tuning thresholds in config.py."""
    lines = []
    if gesture_frame is not None:
        for side in ("left", "right"):
            hand = gesture_frame.get(side)
            lines.append(
                f"pinch {side:5s}: {hand.pinch_ratio:5.2f}" if hand
                else f"pinch {side:5s}:   ---"
            )
    if pose is not None:
        lines.append(f"state: {pose.state.value.upper()}")
        lines.append(f"power: {pose.power:4.2f}")
        lines.append(f"aim  : ({pose.aim[0]:+.2f}, {pose.aim[1]:+.2f})")
    for i, line in enumerate(lines):
        surface.blit(font.render(line, True, (255, 220, 90)), (10, 58 + i * 22))
