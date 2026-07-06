"""Central tunables. Everything you'll want to tweak while playtesting lives here."""

# --- Camera (input space) ---
CAMERA_INDEX = 0
CAPTURE_SIZE = (1280, 720)  # requested from the webcam; driver may pick the nearest mode
MIRROR = True               # selfie view so moving your hand right moves the bow right

# --- Window (screen space / game resolution) ---
WINDOW_SIZE = (1280, 720)
TARGET_FPS = 60

# --- Debug ---
SHOW_FPS = True
