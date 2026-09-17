"""Metric 3D hand pose from MediaPipe image + world landmarks. Camera space.

Scaffolding for milestone 4c (PLAN.md §4.7.2): the data shape is fixed here so
every phase can code against it; `estimate_hand_pose` is implemented in phase 1.

Camera metric axes: +x right in the mirrored image, +y down, +z away from the
camera. Pure numpy — no MediaPipe, cv2 or pygame imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandPose3D:
    position_m: tuple[float, float, float]  # grip point, camera metres; +z away from camera
    depth_m: float                          # == position_m[2], unsmoothed
    px_per_m: float                         # Procrustes scale k
    residual_px: float                      # RMS fit residual over POSE_PALM_POINTS
    inplane_deg: float                      # world->image rotation; ~0 if world axes are camera-aligned


def estimate_hand_pose(
    image_pts: np.ndarray, world_pts: np.ndarray, grip_norm: tuple[float, float]
) -> HandPose3D | None:
    """Weak-perspective Procrustes fit (PLAN.md §4.7.2). Implemented in phase 1."""
    raise NotImplementedError("M4c phase 1")
