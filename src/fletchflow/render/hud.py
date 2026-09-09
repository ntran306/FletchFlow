"""HUD and debug drawing: landmark overlay, power bar, state readout."""

from __future__ import annotations

import math

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
    """Pulsing grab prompt while the bow waits at its dock."""
    if pose is not None and pose.state != BowState.DOCKED:
        return
    dock_x = config.DOCK_POS[0] * config.WINDOW_SIZE[0]
    dock_y = config.DOCK_POS[1] * config.WINDOW_SIZE[1]
    text = font.render("Make a fist to grab the bow!", True, (255, 240, 180))
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
                f"{side:5s}: pinch {hand.pinch_ratio:5.2f}  fist {hand.fist_ratio:5.2f}"
                if hand else f"{side:5s}:   ---"
            )
    if pose is not None:
        lines.append(f"state: {pose.state.value.upper()}")
        lines.append(f"power: {pose.power:4.2f}")
        lines.append(f"aim  : ({pose.aim[0]:+.2f}, {pose.aim[1]:+.2f})")
        lines.append(f"scale: {pose.scale:4.2f}")
        if pose.sight is not None:
            lines.append(f"sight: ({pose.sight[0]:6.1f}, {pose.sight[1]:6.1f})")
    for i, line in enumerate(lines):
        surface.blit(font.render(line, True, (255, 220, 90)), (10, 58 + i * 22))


def draw_crosshair(
    surface: pygame.Surface,
    aim_point: tuple[float, float] | None,
    power: float,
    state: BowState,
) -> None:
    """Four diagonal arms with an open centre; the gap closes as you pull back,
    so the sight visibly tightens with draw power."""
    if aim_point is None or state == BowState.DOCKED:
        return
    x, y = aim_point
    gap = config.CROSSHAIR_GAP_MAX_PX - power * (
        config.CROSSHAIR_GAP_MAX_PX - config.CROSSHAIR_GAP_MIN_PX
    )
    light = (
        int(235 + 20 * power),
        int(235 - 10 * power),
        int(240 - 120 * power),
    )
    diag = math.sqrt(0.5)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        dx, dy = sx * diag, sy * diag
        start = (x + dx * gap, y + dy * gap)
        end = (
            x + dx * (gap + config.CROSSHAIR_ARM_PX),
            y + dy * (gap + config.CROSSHAIR_ARM_PX),
        )
        pygame.draw.line(surface, (20, 20, 25), start, end, 4)
        pygame.draw.line(surface, light, start, end, 2)


def draw_score(
    surface: pygame.Surface, font: pygame.font.Font, session, big_font: pygame.font.Font
) -> None:
    w, _ = config.WINDOW_SIZE
    lines = [f"score {session.score}", f"arrows {session.arrows_left}"]
    if not session.finished:
        lines.append(f"time {max(0.0, config.ROUND_SECONDS - session.elapsed):4.0f}s")
    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 240, 200))
        surface.blit(text, (w - text.get_width() - 14, 12 + i * 22))

    if session.finished:
        over = big_font.render(f"ROUND OVER - score {session.score}", True, (255, 235, 140))
        hint = font.render("press R for a new round", True, (235, 235, 240))
        cx, cy = w // 2, config.WINDOW_SIZE[1] // 2
        surface.blit(over, (cx - over.get_width() // 2, cy - 40))
        surface.blit(hint, (cx - hint.get_width() // 2, cy + 6))

def draw_calibration(
    surface: pygame.Surface,
    big_font: pygame.font.Font,
    font: pygame.font.Font,
    calibrator,
    gesture_frame: GestureFrame | None,
) -> None:
    """Prompt, countdown, and the live ratio being measured.

    Showing the number the routine is actually collecting makes a failed step
    self-explanatory — you can see your fist and open hand reading alike.
    """
    w, h = config.WINDOW_SIZE
    panel = pygame.Surface((w, 150), pygame.SRCALPHA)
    panel.fill((12, 12, 18, 205))
    surface.blit(panel, (0, h // 2 - 75))

    step = min(calibrator.step, len(config.CALIB_STEP_S))
    header = font.render(
        f"CALIBRATION  step {step} of {len(config.CALIB_STEP_S)}", True, (150, 220, 255)
    )
    surface.blit(header, (w // 2 - header.get_width() // 2, h // 2 - 62))

    prompt = big_font.render(calibrator.prompt, True, (255, 245, 210))
    surface.blit(prompt, (w // 2 - prompt.get_width() // 2, h // 2 - 34))

    # Countdown bar for the current step
    total = config.CALIB_STEP_S[step - 1]
    frac = max(0.0, min(1.0, calibrator.seconds_left / total)) if total else 0.0
    bar_w, bar_h = 420, 10
    bx, by = (w - bar_w) // 2, h // 2 + 22
    pygame.draw.rect(surface, (40, 40, 50), (bx, by, bar_w, bar_h), border_radius=5)
    pygame.draw.rect(
        surface, (120, 210, 255),
        (bx, by, int(bar_w * frac), bar_h), border_radius=5,
    )

    readout = _calibration_readout(calibrator, gesture_frame)
    if readout:
        text = font.render(readout, True, (200, 230, 200))
        surface.blit(text, (w // 2 - text.get_width() // 2, h // 2 + 44))


def _calibration_readout(calibrator, gesture_frame: GestureFrame | None) -> str:
    if calibrator.step == 4:
        return "draw and hold"
    if gesture_frame is None:
        return "no hands tracked"
    parts = []
    for side in ("left", "right"):
        hand = gesture_frame.get(side)
        if hand is None:
            continue
        value = hand.pinch_ratio if calibrator.step == 3 else hand.fist_ratio
        label = "pinch" if calibrator.step == 3 else "fist"
        parts.append(f"{side} {label} {value:4.2f}")
    return "   ".join(parts) if parts else "no hands tracked"


def draw_calibration_message(
    surface: pygame.Surface, font: pygame.font.Font, message: str, ok: bool
) -> None:
    w, h = config.WINDOW_SIZE
    color = (170, 255, 190) if ok else (255, 200, 140)
    text = font.render(message, True, color)
    surface.blit(text, (w // 2 - text.get_width() // 2, h // 2 - 10))
