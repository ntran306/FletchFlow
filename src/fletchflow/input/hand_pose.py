"""Metric 3D hand pose from MediaPipe image + world landmarks. Camera space.

Scaffolding for milestone 4c (PLAN.md §4.7.2): the data shape is fixed here so
every phase can code against it; `estimate_hand_pose` is implemented in phase 1.

Camera metric axes: +x right in the mirrored image, +y down, +z away from the
camera. Pure numpy — no MediaPipe, cv2 or pygame imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fletchflow import config


@dataclass(frozen=True)
class HandPose3D:
    position_m: tuple[float, float, float]  # grip point, camera metres; +z away from camera
    depth_m: float                          # == position_m[2], unsmoothed
    px_per_m: float                         # Procrustes scale k
    residual_px: float                      # RMS fit residual over POSE_PALM_POINTS
    inplane_deg: float                      # world->image rotation; ~0 if world axes are camera-aligned
    fit: str = "weak"                       # "persp" / "persp_flipped" / "weak": which model gave the depth


_PERSP_ITERS = 8
_MIN_POINT_DEPTH_M = 0.05


def _fit_perspective(
    img: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    phi0: float,
    z0: float,
    f_px: float,
    W: float,
    H: float,
) -> tuple[float, float, float] | None:
    """Full-perspective fit of the hand's translation and in-plane rotation.

    Model, per palm point i, with world x,y rotated by phi about the view axis:
        (wx', wy') = R(phi) (wx, wy)
        img_i = c + f * (T_xy + w'_xy_i) / (T_z + wz_i)
    Gauss-Newton over (Tx, Ty, Tz, phi), seeded from the weak-perspective depth
    and angle. Solving phi jointly matters: the weak fit's angle is biased by
    perspective, and borrowing it left a 7.5% worst-case depth error on
    noiseless data at 0.35 m that vanishes once phi is free.

    Returns (depth of the palm-point centroid, RMS residual px, phi), or None
    if it diverges or puts a point behind or nearly on the camera.
    """
    cx, cy = W / 2.0, H / 2.0
    n = len(wx)
    phi = float(phi0)

    def rotated(angle: float) -> tuple[np.ndarray, np.ndarray]:
        c, s_ = math.cos(angle), math.sin(angle)
        return c * wx - s_ * wy, s_ * wx + c * wy

    rx, ry = rotated(phi)
    u0, v0 = img.mean(axis=0)
    p = np.array([
        (u0 - cx) * z0 / f_px - rx.mean(),
        (v0 - cy) * z0 / f_px - ry.mean(),
        z0 - wz.mean(),
        phi,
    ])
    J = np.zeros((2 * n, 4))
    for _ in range(_PERSP_ITERS):
        rx, ry = rotated(p[3])
        Z = p[2] + wz
        if np.any(Z < _MIN_POINT_DEPTH_M):
            return None
        X, Y = p[0] + rx, p[1] + ry
        r = np.concatenate([img[:, 0] - (cx + f_px * X / Z), img[:, 1] - (cy + f_px * Y / Z)])
        J[:n, 0] = f_px / Z
        J[:n, 1] = 0.0
        J[:n, 2] = -f_px * X / Z**2
        J[:n, 3] = f_px * (-ry) / Z          # d(rx)/dphi = -ry
        J[n:, 0] = 0.0
        J[n:, 1] = f_px / Z
        J[n:, 2] = -f_px * Y / Z**2
        J[n:, 3] = f_px * rx / Z             # d(ry)/dphi = rx
        step = np.linalg.lstsq(J, r, rcond=None)[0]
        p = p + step
        if not np.all(np.isfinite(p)):
            return None
        if float(np.linalg.norm(step[:3])) < 1e-7 and abs(float(step[3])) < 1e-9:
            break
    rx, ry = rotated(p[3])
    Z = p[2] + wz
    if np.any(Z < _MIN_POINT_DEPTH_M):
        return None
    du = img[:, 0] - (cx + f_px * (p[0] + rx) / Z)
    dv = img[:, 1] - (cy + f_px * (p[1] + ry) / Z)
    residual = float(np.sqrt(np.mean(du**2 + dv**2)))
    depth = float(p[2] + wz.mean())
    phi = math.atan2(math.sin(p[3]), math.cos(p[3]))   # wrap to (-pi, pi]
    if not (math.isfinite(residual) and math.isfinite(depth)):
        return None
    return depth, residual, phi


def estimate_hand_pose(
    image_pts: np.ndarray, world_pts: np.ndarray | None, grip_norm: tuple[float, float]
) -> HandPose3D | None:
    """Weak-perspective Procrustes fit of the 5 palm points (PLAN.md §4.7.2).

    MediaPipe gives two parallel views of the same hand: `image_pts`, its
    normalized image landmarks (px once scaled by `config.CAPTURE_SIZE`), and
    `world_pts`, its `hand_world_landmarks` in metres, hand-centred and
    camera-aligned. Both are already mirrored upstream (tracker.py). Over
    `config.POSE_PALM_POINTS` — wrist + the four MCPs, which stay visible in a
    fist — treat the two point sets as centred complex numbers `zi` (image px)
    and `zw` (world metres) and solve for the single complex similarity
    `s = sum(zi * conj(zw)) / sum(|zw|^2)` that best maps world onto image.
    `k = |s|` is the fitted scale in image px per world metre; `theta = arg(s)`
    is the in-plane (image-plane) rotation implied between the two frames;
    `residual` is the RMS leftover in px after applying `s` to the world
    points. Depth follows from the pinhole relation `z = f_px / k` with
    `f_px = config.CAM_FOCAL_NORM * W`, and the grip point's (u, v) pixel
    position back-projects to camera-metric (X, Y) at that depth.

    Rotation-robust: rotating the hand about any axis foreshortens the image
    extent and the world extent of the palm points together — a tilt that
    shrinks the image span by some factor shrinks the projected world span by
    (approximately) the same factor, so their ratio `k`, and the depth it
    implies, stays close to constant. This is what replaces the old apparent
    palm-size proxy, whose *image-only* extent had nothing to normalize
    against and swung x1.61-x1.84 with hand rotation at a fixed distance; the
    Procrustes fit only swings x1.14-x1.25 (measured, §4.7.2).

    Complex modulus, not a full 2D affine fit: `k = |s|` depends only on the
    *length* `s` scales world vectors by, never the *direction* it rotates
    them through — that direction becomes `theta` instead. So the fit needs
    no assumption that MediaPipe's world-landmark axes are camera-aligned;
    even if they are rotated in-plane relative to the image, that rotation is
    absorbed entirely into `theta` and leaves `k` (and therefore `z`)
    unaffected. `theta` is returned as `inplane_deg` so a playtest can confirm
    it stays near 0, i.e. that the axes really are camera-aligned in practice.

    Measured on synthetic full-perspective hands (PLAN.md §4.7.2; reproduced
    by tests/test_hand_pose.py): noiseless worst depth error 13.6/10.4/7.7/
    5.8/4.6% at z = 0.35/0.45/0.6/0.8/1.0 m (weak-perspective error, a
    systematic bias largest close to the camera); rotation swing at a fixed
    0.45 m x1.19 (was x1.71 for the old apparent-size proxy); with 1.5 px
    image noise plus 4 mm world noise over 400 random poses, per-frame depth
    error p50 3.8-6.1%, p95 12-23% (before the One Euro filter smooths it
    further); fit residual on moderate poses with noise, p50 7.5 px / p95
    15.8 px / max 22 px — hence `POSE_MAX_RESIDUAL_PX = 25`.

    Returns None — the caller holds the last accepted pose — if `world_pts`
    is None, any value this fit actually uses (the 5 palm points and the grip
    point) is non-finite, the world points are degenerate
    (`sum(|zw|^2) < 1e-10`), the fitted scale `k <= 0`, `residual` exceeds
    `config.POSE_MAX_RESIDUAL_PX`, or the implied depth falls outside
    `config.POSE_DEPTH_RANGE_M`.
    """
    if world_pts is None:
        return None

    W, H = config.CAPTURE_SIZE
    idx = list(config.POSE_PALM_POINTS)

    img = image_pts[idx, :2].astype(np.float64) * np.array([W, H], dtype=np.float64)
    world = world_pts[idx, :2].astype(np.float64)

    if not (np.all(np.isfinite(img)) and np.all(np.isfinite(world))):
        return None

    img_c = img - img.mean(axis=0)
    world_c = world - world.mean(axis=0)

    zi = img_c[:, 0] + 1j * img_c[:, 1]
    zw = world_c[:, 0] + 1j * world_c[:, 1]

    denom = float(np.sum(np.abs(zw) ** 2))
    if not (denom >= 1e-10):
        return None

    s = np.sum(zi * np.conj(zw)) / denom
    k = float(abs(s))
    if not (k > 0.0):
        return None
    theta = float(np.angle(s))
    residual = float(math.sqrt(np.mean(np.abs(zi - s * zw) ** 2)))
    if not (math.isfinite(k) and math.isfinite(theta) and math.isfinite(residual)):
        return None

    f_px = config.CAM_FOCAL_NORM * W
    z = f_px / k
    fit = "weak"

    # Refine with full perspective. The weak fit assumes every palm point sits
    # at one depth, which fails exactly in the bow hand's natural pose: a fist
    # pointed at the camera puts the wrist ~9 cm behind the knuckles, a 25%
    # scale difference at 35 cm. Measured at 0.35 m, pitch 50-85 deg: weak
    # alone is rejected by the residual gate on 71-85% of frames and averages
    # ~15% depth error; with this refinement, 5.4-5.7% at a ~15 px residual.
    # World z is only trustworthy up to its sign convention, which cannot be
    # verified without a live hand, so fit both and keep the better residual.
    # At small pitch the two agree (z barely matters); at large pitch the
    # wrong sign is clearly worse. `fit` is logged so a playtest can confirm
    # MediaPipe's convention.
    world3 = world_pts[idx, :3].astype(np.float64)
    if np.all(np.isfinite(world3)):
        best = None
        for sign, name in ((1.0, "persp"), (-1.0, "persp_flipped")):
            refined = _fit_perspective(
                img, world3[:, 0], world3[:, 1], sign * world3[:, 2], theta, z, f_px, W, H
            )
            if refined is not None and (best is None or refined[1] < best[1]):
                best = (refined[0], refined[1], refined[2], name)
        if best is not None and best[1] < residual:
            z, residual, theta, fit = best

    if residual > config.POSE_MAX_RESIDUAL_PX:
        return None

    lo, hi = config.POSE_DEPTH_RANGE_M
    if not (math.isfinite(z) and lo <= z <= hi):
        return None

    u = grip_norm[0] * W
    v = grip_norm[1] * H
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    x = (u - W / 2.0) * z / f_px
    y = (v - H / 2.0) * z / f_px

    return HandPose3D(
        position_m=(float(x), float(y), float(z)),
        depth_m=float(z),
        px_per_m=k,
        residual_px=residual,
        inplane_deg=math.degrees(theta),
        fit=fit,
    )
