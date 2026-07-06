"""MediaPipe HandLandmarker wrapper producing HandFrame data.

Mirroring happens here, exactly once: with config.MIRROR the landmark x
coordinates are flipped (x' = 1 - x) so everything downstream lives in the
same mirrored space the player sees on screen.

Handedness labels: MediaPipe assigns Left/Right assuming the input image is
already selfie-mirrored. We feed the raw (unmirrored) camera frame, so the
labels arrive flipped relative to the player's real hands — they are swapped
unconditionally so HandFrame.left is always the player's actual left hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from fletchflow import config
from fletchflow.vision.camera import Frame


@dataclass(frozen=True)
class HandFrame:
    """Tracked hands for one camera frame, in mirrored normalized coords."""

    timestamp_ms: int
    left: np.ndarray | None   # (21, 3) normalized xyz — player's actual left hand
    right: np.ndarray | None


class HandTracker:
    def __init__(self) -> None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def close(self) -> None:
        self._landmarker.close()

    def track(self, frame: Frame) -> HandFrame | None:
        """Detect hands in a camera frame.

        Returns None without running detection if this frame was already
        processed — VIDEO mode requires strictly increasing timestamps, and
        the 60 Hz game loop polls faster than the ~30 Hz camera delivers.
        """
        timestamp_ms = int(frame.timestamp * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            return None
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        left: np.ndarray | None = None
        right: np.ndarray | None = None
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
            )
            if config.MIRROR:
                points[:, 0] = 1.0 - points[:, 0]
            # Swapped on purpose — see module docstring on handedness
            is_right = handedness[0].category_name == "Left"
            # If the model labels both hands the same, keep both by spilling
            # into the free slot rather than overwriting
            if is_right:
                if right is None:
                    right = points
                else:
                    left = points
            else:
                if left is None:
                    left = points
                else:
                    right = points
        return HandFrame(timestamp_ms=timestamp_ms, left=left, right=right)
