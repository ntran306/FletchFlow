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
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9  # base knuckle of the middle finger — most stable "palm center"
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20
# (tip, mcp) per finger — fist_ratio averages dist(tip,wrist)/dist(mcp,wrist)
# over these. Both terms foreshorten together, so it survives hand rotation.
FINGER_PAIRS = (
    (INDEX_TIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_MCP),
    (RING_TIP, RING_MCP),
    (PINKY_TIP, PINKY_MCP),
)
# The only points we smooth and use. Grew from 4 to 10 for fist detection and
# the rotation-robust palm_size; the One Euro filter is elementwise, so the
# extra cost is a (10,2) array instead of (4,2).
KEY_LANDMARKS = (
    WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP,
    MIDDLE_TIP, RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP,
)

# --- Smoothing (vision/smoothing.py, One Euro) ---
# Lower min_cutoff = smoother at rest but laggier; higher beta = less lag during fast draws
ONE_EURO_MIN_CUTOFF = 1.5  # Hz
ONE_EURO_BETA = 0.3
ONE_EURO_D_CUTOFF = 1.0    # Hz

# --- Gestures & bow state machine (input/) ---
# pinch_ratio = dist(thumb tip, index tip) / dist(wrist, middle MCP)
PINCH_ON = 0.32        # pinch engages below this...
PINCH_OFF = 0.55       # ...and releases above this (hysteresis gap)
PINCH_ON_FRAMES = 3    # consecutive frames required (debounce)
PINCH_OFF_FRAMES = 2
BOW_DROP_FRAMES = 6    # bow-hand pinch must stay open this long to drop the bow
HAND_LOST_GRACE_MS = 200   # draw hand missing longer than this cancels the draw
BOW_LOST_MS = 400          # bow hand missing this long -> bow returns to dock
COOLDOWN_MS = 300          # RELEASED -> HELD

DOCK_POS = (0.5, 0.20)     # bow rest position, mirrored normalized coords
GRAB_RADIUS = 0.11         # pinch within this of the dock grabs the bow
STRING_GRAB_RADIUS = 0.11  # pinch within this of the bow anchor grabs the string
FIRE_POWER_WINDOW = 5  # fire power = max power over the last N tracked frames

# --- Fist gesture: the bow hand holds a closed fist (M4b) ---
# fist_ratio = mean over the four fingers of dist(tip, wrist) / dist(mcp, wrist)
# Open hand ~1.9-2.3, closed fist ~0.7-1.1.
FIST_ON = 1.25          # fist closes below this...   (UNVALIDATED — calibration replaces)
FIST_OFF = 1.60         # ...and opens above this (hysteresis)
FIST_ON_FRAMES = 3
FIST_OFF_FRAMES = 2
PALM_WIDTH_RATIO = 0.85    # dist(5,17) / dist(0,9) on a typical hand
GRIP_PALM_FRACTION = 0.60  # grip_point = wrist + f*(middle_mcp - wrist)

# --- Draw power in 3D (M4b) ---
# The draw hand moves back toward the face, i.e. mostly in DEPTH, so a 2D screen
# distance measured almost nothing. Both terms are expressed in hand-widths,
# which makes them commensurable and camera-distance invariant. See PLAN.md 4.6.2.
CAM_FOCAL_NORM = 0.87   # webcam focal length in normalized-x units (~60 deg FOV)
                        # (UNVALIDATED — a wrong value is a gain error on the
                        # depth term only, and does not break invariance)
DRAW_FULL_HW = 2.0      # hand-widths of pull for full power. Not a new guess:
                        # this is the old DRAW_RANGE / REFERENCE_HAND_SIZE
                        # (0.22 / 0.11), so the tuned feel is preserved.
DRAW_SIZE_SMOOTHING = 0.25  # EMA alpha on palm_size, per tracked frame
DEPTH_HW_MAX = 3.0          # clamp on the depth term

REFERENCE_HAND_SIZE = 0.11   # normalized wrist->MCP at typical desk distance (tunable)
DEPTH_SCALE_MIN = 0.55       # clamp for the bow scale factor
DEPTH_SCALE_MAX = 1.60
DEPTH_SCALE_SMOOTHING = 0.15 # EMA alpha per tracked frame (~30 Hz)

# --- Bow rendering (render/bow.py) ---
BOW_SPAN_PX = 340        # tip-to-tip along the bow
BOW_FLEX_MIN_PX = 18     # limb flex at zero power...
BOW_FLEX_MAX_PX = 80     # ...and at full power
ARROW_LENGTH_PX = 260

# --- Perspective world (game/) ---
# Right-handed, metres, eye at the origin: +x right, +y DOWN, +z INTO the screen.
FOCAL_PX = 900.0            # pinhole focal length; sets the field of view
NEAR_PLANE_M = 0.2
GRAVITY_MS2 = 1.6           # arcade-light: full draw flies nearly flat,
                            # a weak draw visibly arcs short
PHYSICS_DT = 1.0 / 120.0

TARGET_RADIUS_M = 0.9
TARGET_DEPTHS_M = (9.0, 14.0, 20.0)   # longer course = visible flight time
TARGET_Y_RANGE_M = (-1.2, 1.2)
TARGET_X_MARGIN = 0.75      # fraction of the half-FOV width targets may occupy
TARGET_RESPAWN_S = 0.8

ARROW_LAUNCH_Z_M = 1.5
ARROW_LAUNCH_DROP_PX = 380  # spawn this far BELOW the aim point, as if from your
                            # hands: without the offset the arrow flies straight
                            # down the eye ray and every part of it projects to
                            # the same pixel, so it renders as a zero-length line
SIGHT_DEPTH_M = 10.0        # where the launch ray meets the crosshair
ARROW_SPEED_MIN_MS = 9.0
ARROW_SPEED_MAX_MS = 19.0
ARROW_LENGTH_M = 0.55
ARROW_MAX_DEPTH_M = 45.0

AIM_LEAD_GAIN = 0.0         # 0 = crosshair sits on the bow hand (point at what you hit)
SCORE_RINGS = ((0.28, 10), (0.60, 5), (1.00, 2))  # (radius fraction, points)
ROUND_ARROWS = 10
ROUND_SECONDS = 90.0

# --- Crosshair (render/hud.py) ---
CROSSHAIR_GAP_MAX_PX = 48   # arm gap at zero power...
CROSSHAIR_GAP_MIN_PX = 13   # ...and at full power: the sight tightens as you pull
CROSSHAIR_ARM_PX = 22
HIT_FEEDBACK_S = 0.8        # how long a hit burst stays on screen

# The reticle is a sight pin above the grip, not the grip itself (M4b). It used
# to sit exactly on the bow hand, so aiming meant putting your hand on the
# target — high, near the frame edge where tracking drops it.
CROSSHAIR_RISE_PX = 110.0   # px above the anchor, scaled by depth. The bow body
                            # paints exactly 75*scale px up, so this clears it.
RETICLE_MIN_CUTOFF = 0.8    # Hz — much heavier than the hands' 1.5, so the
                            # sight has weight and does not inherit hand tremor
RETICLE_BETA = 0.0006       # NOTE: One Euro's beta is unit-dependent and this
                            # filter runs on PIXELS, not the normalized coords
                            # the hand smoother uses. The hands' beta of 0.3 in
                            # [0,1] space is ~0.0002 in px; anything near 0.3
                            # here would push the cutoff past 15 Hz on an
                            # ordinary aim sweep, i.e. no smoothing at all.
RETICLE_D_CUTOFF = 1.0
AIM_MIN_SEPARATION_PX = 25.0  # below this the aim vector is degenerate and the
                              # last good one is held. A correct 3D draw
                              # collapses the on-screen hand separation, and aim
                              # drives bow orientation — without this it spins.

# --- Calibration (game/calibration.py, M4b measurement half) ---
CALIB_STEP_S = (2.0, 2.0, 2.0, 3.0)
CALIB_MIN_SEPARATION = 0.45   # reject if open_med - closed_med is below this
CALIB_ON_FRACTION = 0.65
CALIB_OFF_FRACTION = 0.30
CALIB_DRAW_PERCENTILE = 90
CALIB_DRAW_CLAMP = (1.2, 4.0)
CALIB_MIN_SAMPLES = 10

# --- Debug ---
SHOW_FPS = True
DEBUG_OVERLAY = True   # start with landmarks + state readout on; F1 toggles
