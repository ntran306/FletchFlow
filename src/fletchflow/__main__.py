"""FletchFlow entry point.

Milestone 4: a first-person shooting gallery. Pinch to grab the bow, pinch
near it to nock, pull to build power, open your fingers to loose. Arrows fly
INTO the screen and shrink with distance; targets sit at three depths. F1
toggles the debug overlay, R starts a new round, ESC quits.

Self-check mode (`fletchflow --selfcheck [seconds]`) runs the identical
pipeline headless: no window opens, rendered frames are saved as PNGs once
per second, and a stats verdict prints at the end. `--fake-bow` additionally
injects a synthetic sweeping BowPose so the bow rendering can be inspected
without anyone in front of the camera.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import pygame

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState, BowStateMachine
from fletchflow.input.gestures import extract as extract_gestures
from fletchflow.input.mapping import BowPose, Mapper
from fletchflow.game.session import GallerySession, aim_point
from fletchflow.render.bow import draw_bow
from fletchflow.render.hud import (
    draw_crosshair,
    draw_debug_state,
    draw_grab_prompt,
    draw_hands,
    draw_power_bar,
    draw_score,
)
from fletchflow.render.world_render import draw_world
from fletchflow.vision.camera import Camera
from fletchflow.vision.pipeline import TrackingPipeline
from fletchflow.vision.telemetry import TelemetryLogger


def fake_bow_pose(elapsed: float) -> BowPose:
    """Synthetic DRAWN pose sweeping aim and power, for --fake-bow."""
    w, h = config.WINDOW_SIZE
    angle = math.radians(-60 + 50 * math.sin(elapsed * 0.9))
    aim = (math.cos(angle), math.sin(angle))
    power = 0.5 + 0.5 * math.sin(elapsed * 1.7)
    anchor = (w * 0.42, h * 0.52)
    draw_dist = 120 + 140 * power
    scale = 1.0 + 0.45 * math.sin(elapsed * 0.6)
    return BowPose(
        anchor=anchor,
        draw_point=(anchor[0] - aim[0] * draw_dist, anchor[1] - aim[1] * draw_dist),
        aim=aim,
        power=power,
        state=BowState.DRAWN,
        fire=None,
        scale=scale,
    )


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
    parser.add_argument(
        "--fake-bow",
        action="store_true",
        help="render a synthetic sweeping bow (visual iteration without hands)",
    )
    parser.add_argument(
        "--telemetry",
        metavar="PATH",
        default=None,
        help="log per-frame hand-tracking signals to a CSV for aim-pose analysis",
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

    state_machine = BowStateMachine()
    mapper = Mapper()
    session = GallerySession()
    telemetry = TelemetryLogger(args.telemetry) if args.telemetry else None

    body_renderer = None
    try:
        from fletchflow.render.bow3d import BowBodyRenderer3D

        body_renderer = BowBodyRenderer3D()
    except Exception as exc:  # no GL 3.3 / driver issue -> 2D fallback
        print(f"3D bow unavailable ({exc}); using 2D fallback", file=sys.stderr)

    pygame.init()
    screen = pygame.display.set_mode(config.WINDOW_SIZE)
    pygame.display.set_caption("FletchFlow")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    big_font = pygame.font.SysFont("consolas", 32, bold=True)
    background_cache = BackgroundCache()

    debug_overlay = config.DEBUG_OVERLAY
    start_time = time.perf_counter()
    next_save = start_time + 1.0
    saved = 0
    hands_seen = 0
    ticks = 0
    fires = 0
    last_frame_time = start_time
    gesture_frame = None
    # Docked bow renders immediately, before any hands are tracked
    pose: BowPose | None = mapper.map(
        BowSnapshot(0, BowState.DOCKED, config.DOCK_POS, None, 0.0, None)
    )
    last_processed_ms = -1

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                    debug_overlay = not debug_overlay
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    session = GallerySession()

            background = background_cache.get(camera.latest())
            if background is not None:
                screen.blit(background, (0, 0))
            else:
                screen.fill((20, 20, 20))

            # Input chain: run once per NEW tracked frame, not per render tick
            hand_frame = pipeline.latest()
            if hand_frame is not None and hand_frame.timestamp_ms != last_processed_ms:
                last_processed_ms = hand_frame.timestamp_ms
                gesture_frame = extract_gestures(hand_frame)
                snapshot = state_machine.update(gesture_frame)
                pose = mapper.map(snapshot)
                if pose.fire is not None:
                    fires += 1
                if telemetry is not None:
                    telemetry.log(
                        gesture_frame, snapshot,
                        state_machine.bow_side, state_machine.draw_side,
                    )

            now = time.perf_counter()
            if args.fake_bow:
                pose = fake_bow_pose(now - start_time)

            dt = now - last_frame_time
            last_frame_time = now
            session.update(pose, dt)

            # World first: targets and arrows sit behind the bow you hold
            draw_world(screen, session, font, big_font)
            if pose is not None:
                draw_bow(screen, pose, body_renderer)
                draw_crosshair(screen, aim_point(pose), pose.power, pose.state)
            draw_grab_prompt(screen, big_font, pose, now - start_time)
            draw_power_bar(screen, pose)
            draw_score(screen, font, session, big_font)

            if debug_overlay:
                draw_hands(screen, hand_frame, font)
                draw_debug_state(screen, font, gesture_frame, pose)

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
        if telemetry is not None:
            telemetry.close()
        pipeline.stop()
        camera.stop()
        pygame.quit()

    if args.selfcheck:
        ok = camera.fps >= 25 and pipeline.fps >= 25
        print(
            f"selfcheck: render {clock.get_fps():.1f} fps | camera {camera.fps:.1f} fps"
            f" | tracker {pipeline.fps:.1f} fps @ {pipeline.ms:.1f} ms"
            f" | ticks with hands {hands_seen}/{ticks} | fires {fires}"
            f" | score {session.score}"
            f" | {saved} frames -> {out_dir}"
        )
        print("SELFCHECK OK" if ok else "SELFCHECK FAIL: camera or tracker below 25 fps")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
