"""Camera space -> screen space. The ONLY module where the two spaces meet.

Everything upstream (tracker, gestures, bow_input) thinks in mirrored
normalized [0,1] coordinates; everything downstream (game, render) thinks in
pixels. Capture and window are both 16:9, so the mapping is a plain scale.

This module also places the **sight** (M4c, PLAN.md §4.7.4). The 4b sight pin
was a fixed screen offset above the bow grip, so turning or aiming the bow
never moved it. The crosshair now follows the real 3D line from the draw hand
through the bow hand: moving the bow arm, turning the torso, or shifting the
draw hand all move it. `aim_angles()` (below) turns the two hands' metric
camera positions (bow_input.py's bow_position_m / draw_position_m) into
(yaw, pitch, baseline); `Mapper.map()` smooths the angles, zeroes them by the
calibrated offset (`apply_calibration`), applies AIM_GAIN, clamps to the
screen edges, and projects through the game's own pinhole (FOCAL_PX) to a
pixel `sight` and a `bow_forward` target direction. Only DRAWN computes a
fresh sight, ramped in by `aim_weight` over the first few centimetres of draw
since the line is meaningless with the hands together at the nock; RELEASED
holds the last one so the arrow's flight still has something to aim at;
HELD/DOCKED show none — there is no arrow strung yet to aim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fletchflow import config
from fletchflow.input.bow_input import BowSnapshot, BowState
from fletchflow.vision.smoothing import OneEuroFilter

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class FireEvent:
    origin: Vec2      # px — bow anchor at release
    direction: Vec2   # unit vector, screen space
    power: float      # 0..1


@dataclass(frozen=True)
class BowPose:
    """Everything the game/render layers ever learn about the player."""

    anchor: Vec2 | None       # px — bow hand
    draw_point: Vec2 | None   # px — string hand grip point
    aim: Vec2                 # unit vector; last known when indeterminate
    power: float              # 0..1
    state: BowState
    fire: FireEvent | None    # set only on the release frame
    scale: float = 1.0        # bow-hand depth scale, passed through from BowSnapshot
    sight: Vec2 | None = None # px — the aimed crosshair (§4.7.4); None while aim_weight == 0
    # M4c (PLAN.md §4.7.10)
    bow_forward: tuple[float, float, float] = (0.0, 0.0, 1.0)  # game axes, unit
    bow_up: tuple[float, float, float] = (0.0, -1.0, 0.0)      # game axes, unit, orthogonal to forward
    aim_weight: float = 0.0                                     # crosshair alpha, 0..1
    render_scale: float = 1.0


def _to_screen(p: Vec2) -> Vec2:
    return (p[0] * config.WINDOW_SIZE[0], p[1] * config.WINDOW_SIZE[1])


UP: Vec2 = (0.0, -1.0)


def aim_angles(bow_m: Vec3, draw_m: Vec3) -> tuple[float, float, float]:
    """(yaw, pitch, baseline) in radians/metres, from the 3D line draw->bow.

    Camera metres to game direction: a = normalize(d.x, d.y, -d.z) with
    d = bow_m - draw_m; yaw = atan2(a.x, a.z); pitch = atan2(a.y, hypot(a.x, a.z)).
    The bow hand sits nearer the camera than the draw hand, so d.z < 0 and
    a.z > 0 (PLAN.md §4.7.3/4.7.4). atan2 is homogeneous of degree 0, so yaw
    and pitch come out identical whether `a` is unit-normalized first or not
    — this skips the redundant sqrt and uses (d.x, d.y, -d.z) directly.
    """
    dx = bow_m[0] - draw_m[0]
    dy = bow_m[1] - draw_m[1]
    dz = bow_m[2] - draw_m[2]
    baseline = math.sqrt(dx * dx + dy * dy + dz * dz)
    ax, ay, az = dx, dy, -dz
    yaw = math.atan2(ax, az)
    pitch = math.atan2(ay, math.hypot(ax, az))
    return (yaw, pitch, baseline)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize3(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _slerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Spherical interpolation between two unit vectors. Falls back to a
    normalized linear interpolation (nlerp) when they are within ~1e-4 rad of
    each other, where sin(angle) is too small for the slerp weights to be
    numerically safe (PLAN.md §4.7.3)."""
    dot = _clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1.0, 1.0)
    angle = math.acos(dot)
    if angle < 1e-4:
        return _normalize3((
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ))
    sin_angle = math.sin(angle)
    wa = math.sin((1.0 - t) * angle) / sin_angle
    wb = math.sin(t * angle) / sin_angle
    return _normalize3((
        a[0] * wa + b[0] * wb,
        a[1] * wa + b[1] * wb,
        a[2] * wa + b[2] * wb,
    ))


class Mapper:
    def __init__(self) -> None:
        self._last_aim: Vec2 = UP
        self._aim_filter = OneEuroFilter(
            min_cutoff=config.AIM_MIN_CUTOFF,
            beta=config.AIM_BETA,
            d_cutoff=config.AIM_D_CUTOFF,
        )
        self._aim_yaw0 = 0.0
        self._aim_pitch0 = 0.0
        self._prev_state: BowState | None = None

        # 3D sight (§4.7.4): cached so RELEASED can hold DRAWN's last values.
        self._sight: Vec2 | None = None
        self._aim_weight: float = 0.0
        self._forward_target: Vec3 = (0.0, 0.0, 1.0)
        self._bow_forward: Vec3 = (0.0, 0.0, 1.0)
        self._bow_up: Vec3 = (0.0, -1.0, 0.0)
        self._last_forward_ts_s: float | None = None

    def apply_calibration(self, result) -> None:
        """Adopt the sight zero offsets measured by calibration step 5 (§4.7.9)."""
        self._aim_yaw0 = math.radians(result.aim_yaw0_deg)
        self._aim_pitch0 = math.radians(result.aim_pitch0_deg)

    def map(self, snap: BowSnapshot) -> BowPose:
        anchor = _to_screen(snap.anchor)
        draw_point = _to_screen(snap.draw_point) if snap.draw_point else None

        # -- 2D aim vector (unchanged since M4b): drives the fallback 2D bow
        # renderer's string/arrow direction and FireEvent.direction. This is
        # deliberately independent of the new 3D sight below.
        if snap.state == BowState.DRAWN and draw_point is not None:
            dx = anchor[0] - draw_point[0]
            dy = anchor[1] - draw_point[1]
            length = math.hypot(dx, dy)
            # A correct 3D draw collapses the on-screen separation, and aim
            # drives bow orientation — without this floor the bow spins.
            if length > config.AIM_MIN_SEPARATION_PX:
                self._last_aim = (dx / length, dy / length)
            aim = self._last_aim
        elif snap.state == BowState.RELEASED:
            aim = self._last_aim  # hold through release so the fire uses it
        else:
            aim = self._last_aim = UP  # docked/held bows point up

        # -- 3D sight along the arrow (§4.7.4) --------------------------------
        # Reset the aim-angle filter whenever state leaves the {DRAWN,
        # RELEASED} pair, so every new draw starts clean instead of
        # inheriting stale yaw/pitch velocity from the previous one.
        if self._prev_state in (BowState.DRAWN, BowState.RELEASED) and snap.state not in (
            BowState.DRAWN,
            BowState.RELEASED,
        ):
            self._aim_filter.reset()
        self._prev_state = snap.state

        if snap.state == BowState.DRAWN:
            self._sight, self._aim_weight, self._forward_target = self._aim_from_snapshot(snap)
        elif snap.state == BowState.RELEASED:
            pass  # hold the last sight / weight / forward target from DRAWN
        else:  # HELD / DOCKED: no arrow strung, nothing to aim
            self._sight, self._aim_weight, self._forward_target = None, 0.0, (0.0, 0.0, 1.0)

        bow_forward = self._ease_bow_forward(self._forward_target, snap.timestamp_ms)
        bow_up = self._compute_bow_up(snap.knuckle_dir, bow_forward)

        fire = None
        if snap.fired_power is not None:
            fire = FireEvent(origin=anchor, direction=aim, power=snap.fired_power)

        return BowPose(
            anchor=anchor,
            draw_point=draw_point,
            aim=aim,
            power=snap.power,
            state=snap.state,
            fire=fire,
            scale=snap.scale,
            sight=self._sight,
            bow_forward=bow_forward,
            bow_up=bow_up,
            aim_weight=self._aim_weight,
            render_scale=snap.render_scale,
        )

    def _aim_from_snapshot(self, snap: BowSnapshot) -> tuple[Vec2 | None, float, Vec3]:
        """DRAWN only: the gained, zeroed sight and forward target from the 3D
        draw->bow line, or (None, 0.0, (0,0,1)) while a position is missing or
        the baseline is too short to trust — hands still together at the nock
        (PLAN.md §4.7.4)."""
        if snap.bow_position_m is None or snap.draw_position_m is None:
            return None, 0.0, (0.0, 0.0, 1.0)

        yaw, pitch, baseline = aim_angles(snap.bow_position_m, snap.draw_position_m)
        weight = _clamp01(
            (baseline - config.AIM_BASELINE_MIN_M) / config.AIM_BASELINE_RAMP_M
        )
        if weight <= 0.0:
            return None, weight, (0.0, 0.0, 1.0)

        smoothed = self._aim_filter(
            np.array([yaw, pitch], dtype=np.float64), snap.timestamp_ms / 1000.0
        )
        yaw_s, pitch_s = float(smoothed[0]), float(smoothed[1])
        yaw_g = config.AIM_GAIN * (yaw_s - self._aim_yaw0)
        pitch_g = config.AIM_GAIN * (pitch_s - self._aim_pitch0)

        # Clamp to AIM_SCREEN_MARGIN of the half-FOV so an extreme aim pins at
        # the edge instead of sliding off. The vertical bound depends on yaw:
        # the projection below is f*tan(pitch)/cos(yaw), so clamping pitch on
        # its own lets the sight leave the window by up to 53 px once yaw is
        # also extreme. Solving |f*tan(p)/cos(yaw)| <= margin*h/2 for p gives
        # the cos(yaw) factor, applied after yaw is clamped.
        w, h = config.WINDOW_SIZE
        yaw_clamp = math.atan(config.AIM_SCREEN_MARGIN * (w / 2.0) / config.FOCAL_PX)
        yaw_g = _clamp(yaw_g, -yaw_clamp, yaw_clamp)
        pitch_clamp = math.atan(
            config.AIM_SCREEN_MARGIN * (h / 2.0) * math.cos(yaw_g) / config.FOCAL_PX
        )
        pitch_g = _clamp(pitch_g, -pitch_clamp, pitch_clamp)

        cx, cy = w / 2.0, h / 2.0
        sight = (
            cx + config.FOCAL_PX * math.tan(yaw_g),
            cy + config.FOCAL_PX * math.tan(pitch_g) / math.cos(yaw_g),
        )
        forward_target = (
            math.cos(pitch_g) * math.sin(yaw_g),
            math.sin(pitch_g),
            math.cos(pitch_g) * math.cos(yaw_g),
        )
        return sight, weight, forward_target

    def _ease_bow_forward(self, target: Vec3, timestamp_ms: int) -> Vec3:
        """Ease bow_forward toward `target` with time constant
        BOW_ORIENT_TAU_S (§4.7.3), snapping straight to it on the very first
        frame so the bow doesn't sweep in from a default at startup."""
        ts = timestamp_ms / 1000.0
        if self._last_forward_ts_s is None:
            self._bow_forward = _normalize3(target)
        else:
            dt = _clamp(ts - self._last_forward_ts_s, 0.0, 0.1)
            t = 1.0 - math.exp(-dt / config.BOW_ORIENT_TAU_S)
            self._bow_forward = _slerp(self._bow_forward, _normalize3(target), t)
        self._last_forward_ts_s = ts
        return self._bow_forward

    def _compute_bow_up(self, knuckle_dir: Vec2, forward: Vec3) -> Vec3:
        """(kx, ky, 0) orthogonalized against forward; keep the previous up if
        that degenerates near zero (§4.7.3)."""
        u = (knuckle_dir[0], knuckle_dir[1], 0.0)
        d = u[0] * forward[0] + u[1] * forward[1] + u[2] * forward[2]
        ortho = (u[0] - d * forward[0], u[1] - d * forward[1], u[2] - d * forward[2])
        norm = math.sqrt(ortho[0] ** 2 + ortho[1] ** 2 + ortho[2] ** 2)
        if norm < 0.2:
            return self._bow_up
        self._bow_up = (ortho[0] / norm, ortho[1] / norm, ortho[2] / norm)
        return self._bow_up
