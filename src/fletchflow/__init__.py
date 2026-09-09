"""FletchFlow — computer-vision archery game."""

import os

# Must be set before cv2 is imported anywhere, so it lands here in the package
# __init__ rather than in vision/camera.py. Without it OpenCV's MSMF backend
# spends ~20 s negotiating hardware transforms before the first frame; with it,
# MSMF opens in ~1.5 s AND sustains 30 fps, where DSHOW manages only 10.
# Measured 2026-09-08 — see PLAN.md §7.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

__version__ = "0.1.0"
