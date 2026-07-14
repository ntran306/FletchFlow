"""Central tunables. Everything you'll want to tweak while playtesting lives here."""

from pathlib import Path

# --- Paths ---
# Repo root — valid because we run from an editable install of the src layout
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "assets" / "models" / "hand_landmarker.task"

# --- Camera (input space) ---
CAMERA_INDEX = 0
CAPTURE_SIZE = (1280, 720)  # requested from the webcam; driver may pick the nearest mode
CAPTURE_FPS = 30            # must be requested explicitly — otherwise the driver
                            # bistably picks a 15 fps low-light mode (measured 2026-07-06)
MANUAL_EXPOSURE = None      # None = auto. In a dim room auto-exposure can still drop
                            # the camera to ~16 fps; set to -5 (1/32 s) to pin 30 fps.
                            # Don't leave -5 set in a bright room — it may overexpose.
MIRROR = True               # selfie view so moving your hand right moves the bow right

# --- Window (screen space / game resolution) ---
WINDOW_SIZE = (1280, 720)
TARGET_FPS = 60

# --- Hand tracking (vision/tracker.py) ---
NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- Landmark indices (MediaPipe 21-point hand model) ---
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9  # base knuckle of the middle finger — most stable "palm center"
KEY_LANDMARKS = (WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP)  # the only points we smooth and use

# --- Smoothing (vision/smoothing.py, One Euro) ---
# Lower min_cutoff = smoother at rest but laggier; higher beta = less lag during fast draws
ONE_EURO_MIN_CUTOFF = 1.5  # Hz
ONE_EURO_BETA = 0.3
ONE_EURO_D_CUTOFF = 1.0    # Hz

# --- Gestures & bow state machine (input/) ---
# pinch_ratio = dist(thumb tip, index tip) / dist(wrist, middle MCP)
PINCH_ON = 0.35        # DRAWN when ratio drops below this...
PINCH_OFF = 0.55       # ...RELEASED when it rises above this (hysteresis gap)
PINCH_ON_FRAMES = 3    # consecutive frames required (debounce)
PINCH_OFF_FRAMES = 2
ARM_FRAMES = 5         # both hands visible this long -> ARMED
HAND_LOST_GRACE_MS = 200   # hand missing longer than this cancels a draw
IDLE_TIMEOUT_MS = 500      # both hands missing this long -> IDLE
COOLDOWN_MS = 300          # RELEASED -> ARMED
DRAW_MIN = 0.12        # wrist-to-wrist distance (normalized) at zero power
DRAW_MAX = 0.55        # ...at full power (calibration will overwrite)
FIRE_POWER_WINDOW = 5  # fire power = max power over the last N tracked frames
DOMINANT_HAND = "right"  # provisional bow hand before any pinch = the other one

# --- Bow rendering (render/bow.py) ---
BOW_SPAN_PX = 340        # tip-to-tip along the bow
BOW_FLEX_MIN_PX = 18     # limb flex at zero power...
BOW_FLEX_MAX_PX = 80     # ...and at full power
ARROW_LENGTH_PX = 260

# --- Debug ---
SHOW_FPS = True
DEBUG_OVERLAY = True   # start with landmarks + state readout on; F1 toggles
