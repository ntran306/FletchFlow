# FletchFlow

A computer-vision archery game. Your webcam tracks both hands: the front hand holds a virtual bow, the back hand pinches to nock and draw. Pull back, aim, release — hit targets on screen.

See [PLAN.md](PLAN.md) for the full design, architecture, and roadmap.

## Setup (Windows)

Requires Python 3.9–3.12 (MediaPipe does not support 3.13+ yet) and a webcam.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Download the MediaPipe hand landmarker model (needed from milestone 1 onward):

```powershell
mkdir assets\models -Force
Invoke-WebRequest https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -OutFile assets\models\hand_landmarker.task
```

## Run

```powershell
fletchflow
```

(or `python -m fletchflow`). ESC quits. Tunables — camera index, resolution, mirroring — live in `src/fletchflow/config.py`.

Good lighting helps tracking a lot; face a window or lamp, not away from it.
