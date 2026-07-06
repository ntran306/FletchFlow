# FletchFlow — Project Plan

A computer-vision archery game: your webcam tracks both hands, one hand "holds" a virtual bow, the other nocks and draws an arrow. Pull back, aim, release — hit targets on screen.

---

## 1. Core Concept & Interaction Design

The whole game hinges on mapping two tracked hands to one intuitive bow:

| Role | Hand | Signal |
|---|---|---|
| **Bow hand** | Front hand (usually non-dominant) | Anchors the bow's position on screen |
| **Draw hand** | Back hand (usually dominant) | Pinch (thumb + index) = grab string; release pinch = fire |

- **Aim direction** = vector from draw hand → bow hand (like a real bow: the arrow flies away from your face, along the line between your hands).
- **Draw power** = distance between the two hands, normalized against the player's arm span (calibrated at start), clamped to \[0, 1].
- **Bow state machine**: `IDLE → ARMED (two hands visible) → DRAWN (pinch held near bow) → RELEASED (pinch opened) → cooldown → ARMED`.

This state machine is the heart of the game — get it feeling good before adding anything else.

### Two coordinate spaces — the defining design decision

The camera is an **input device**, not the game world:

| Space | What lives there | Measured in |
|---|---|---|
| **Camera space** (input) | Hand landmarks, pinch detection, draw distance, aim vector, bow state machine | Normalized \[0,1] MediaPipe coords, mirrored |
| **Screen space** (game) | Bow's on-screen position, arrow trajectory & physics, targets, hit detection, scoring | Pixels, fixed game resolution |

A single **mapping layer** sits between them: it converts the bow hand's camera position to an on-screen bow anchor, and carries the aim direction + power across. Everything downstream — arrow flight, gravity, collision, score — is pure screen-space math with zero knowledge of the camera. Benefits:

- Targets and scoring are ordinary game entities; they never care about camera resolution, aspect ratio, or tracking quality.
- Physics is deterministic and unit-testable (no camera needed).
- You can tune "game feel" (arrow speed, gravity, target size) independently from "input feel" (smoothing, thresholds).
- The camera feed becomes optional set dressing: render it faint in the background or as a small picture-in-picture for feedback — the game itself is drawn in crisp screen space on top, not AR-overlaid onto the video.

## 2. Tech Stack (local-first)

**Language: Python 3.11 or 3.12** (MediaPipe supports 3.9–3.12; avoid 3.13 for now).

| Layer | Choice | Why |
|---|---|---|
| Hand tracking | **MediaPipe Hand Landmarker** (`mediapipe`) | 21 landmarks per hand, 2-hand support, handedness labels, runs 30+ FPS on CPU — no GPU needed |
| Camera capture | **OpenCV** (`opencv-python`) | Simple, reliable webcam access on Windows |
| Game engine | **Pygame-CE** (`pygame-ce`) | Proper game loop, sprites, sound, text — much better than drawing UI in OpenCV |
| Math | **NumPy** | Vector math for aiming/physics |

### Installation

```powershell
cd D:\Projects\FletchFlow
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install mediapipe opencv-python pygame-ce numpy
```

Also download the MediaPipe hand landmarker model file (`hand_landmarker.task`) into `assets/models/` — the new MediaPipe Tasks API loads it explicitly, which is good: it makes the dependency visible and versionable.

Verify the environment with a 10-line script that opens the webcam and prints FPS before building anything else.

## 3. Project Structure

```
FletchFlow/
├── PLAN.md
├── README.md
├── requirements.txt
├── pyproject.toml              # optional, but nice for tooling config
├── assets/
│   ├── models/hand_landmarker.task
│   ├── sprites/                # bow, arrow, targets
│   └── sounds/                 # draw creak, release twang, hit thunk
├── src/fletchflow/
│   ├── __main__.py             # entry point: python -m fletchflow
│   ├── config.py               # tunables: thresholds, smoothing, resolution
│   ├── vision/
│   │   ├── camera.py           # webcam capture (own thread)
│   │   ├── tracker.py          # MediaPipe wrapper → HandFrame data
│   │   └── smoothing.py        # One Euro filter / EMA for landmarks
│   ├── input/
│   │   ├── gestures.py         # pinch detection, hand-role assignment
│   │   ├── bow_input.py        # bow state machine → BowState events
│   │   └── mapping.py          # camera space → screen space (bow anchor, aim, power)
│   ├── game/
│   │   ├── entities.py         # Arrow, Target, Bow
│   │   ├── physics.py          # projectile motion, collision
│   │   ├── scoring.py
│   │   └── scenes.py           # Menu / Calibration / Play / GameOver
│   └── render/
│       ├── renderer.py         # pygame drawing: camera feed + game layer
│       └── hud.py              # score, power meter, debug overlay
└── tests/
    ├── test_gestures.py        # pure logic — easily testable
    ├── test_bow_state.py
    └── test_physics.py
```

**The key boundary:** everything in `vision/` and `input/` produces plain data (dataclasses like `HandFrame`, `BowState`), and `mapping.py` is the *only* place camera coordinates become screen coordinates. The `game/` layer never touches MediaPipe, the camera, or normalized coords — it's a normal screen-space 2D game that happens to be fed by hands instead of a mouse. This is also what makes the future mobile port feasible — swap the vision layer, keep the game.

## 4. Architecture & Design Patterns

### 4.1 Pipeline architecture (the big one)

```
Camera thread                    Main thread (60 FPS game loop)
┌──────────┐  latest frame   ┌─────────┐ ┌────────┐ ┌───────┐ ┌────────┐ ┌────────┐
│ camera.py│ ──────────────► │tracker  │►│gestures│►│mapping│►│ game   │►│ render │
└──────────┘  (1-slot queue) └─────────┘ └────────┘ └───────┘ └────────┘ └────────┘
                              landmarks   BowState   screen-   entities   pixels
                              (camera     (camera    space
                               space)      space)    BowPose
```

The `mapping` stage is the camera→screen boundary from §1: everything left of it thinks in normalized camera coordinates; everything right of it thinks in pixels. `game/` receives only a screen-space `BowPose` (anchor point, aim direction, power, fire event) and simulates arrows and targets purely on screen.

- **Camera on its own thread**, publishing only the *latest* frame (drop stale ones). Never let the game loop block on the webcam — this is the #1 cause of laggy CV games.
- **Each stage consumes and produces plain data.** Stages are individually testable without a camera.
- The game loop runs at a fixed timestep regardless of camera FPS (webcam ~30 FPS, game ~60 FPS — interpolate/smooth between tracking updates).

### 4.2 State machines (two of them)

1. **Scene state machine**: `MENU → CALIBRATION → PLAYING → PAUSED / GAME_OVER`. Calibration is a real scene, not an afterthought — measure the player's max hand separation and pinch distances there.
2. **Bow state machine** (in `bow_input.py`): as in §1. Use *hysteresis* on every threshold — e.g. pinch engages below 4 cm but only disengages above 6 cm — otherwise the bow will stutter-fire at the boundary.

### 4.3 Smoothing / filtering

Raw landmarks jitter by a few pixels every frame. Non-negotiable techniques:

- **One Euro filter** on hand positions (the standard for pointing interfaces — low lag when moving fast, smooth when still). Simple EMA is an acceptable v1.
- **Debounce gestures over time**: a pinch must hold for ~3 frames before it counts; a release must persist ~2 frames. Single-frame glitches are common.
- **Grace period for lost tracking**: if a hand disappears for < 200 ms, hold last known state instead of dropping the arrow.

### 4.4 Everything tunable lives in `config.py`

Pinch thresholds, smoothing coefficients, power curve, arrow speed. You will tune these constantly; add a debug overlay (toggle with a key) showing landmarks, distances, and current bow state live. That overlay will save you hours.

## 5. Milestones

| # | Milestone | Definition of done |
|---|---|---|
| 0 | **Environment** | Webcam feed in a pygame window at 30+ FPS, mirrored (selfie view) |
| 1 | **Tracking** | Both hands' 21 landmarks drawn on the feed, handedness labeled, smoothed |
| 2 | **Gestures** | Pinch on/off detected reliably; debug overlay shows bow state machine transitions |
| 3 | **The Bow** | Bow sprite at the mapped screen anchor follows the bow hand, string pulls with the draw hand, power meter fills, arrow renders nocked |
| 4 | **Firing** | Release fires arrow along aim vector with power-scaled speed — pure screen-space physics with simple gravity; arrows stick where they land |
| 5 | **Game** | Static screen-space targets, hit detection, score, rounds, sounds |
| 6 | **Feel & polish** | Calibration scene, moving targets, screen shake / hit effects, difficulty ramp |

Milestones 0–2 are pure CV plumbing; the "game" only starts at 3. Resist skipping ahead — a janky bow makes everything after it unfun.

## 6. Skills to Learn (roughly in order of need)

1. **Coordinate spaces** — MediaPipe gives normalized \[0,1] coordinates with y-down; pygame uses pixels. You must also *mirror* the x-axis for selfie view or aiming feels reversed. Most CV-game bugs live here.
2. **The 21-landmark hand model** — which indices matter (4 = thumb tip, 8 = index tip, 0 = wrist, 9 = middle MCP for a stable "palm center").
3. **Signal filtering** — EMA and the One Euro filter; why cutoff/beta parameters trade lag vs. smoothness.
4. **Finite state machines** — with hysteresis and time-based debouncing for noisy inputs.
5. **Game loop fundamentals** — fixed timestep vs. variable, delta time, decoupling simulation from rendering.
6. **2D vector math** — normalize, scale, dot product; projectile motion for arrows.
7. **Basic threading** — producer/consumer with a 1-slot "latest value" queue for the camera.
8. **Profiling** — `cProfile` / simple frame timers, so when FPS drops you find out whether it's tracking, rendering, or copying frames.

Not needed yet: neural network internals, CUDA, 3D math, any web/mobile tech.

## 7. Known Gotchas

- **Hands crossing/overlapping** confuses tracking — the bow pose keeps hands apart, which helps, but handle the "one hand lost" case gracefully.
- **Handedness flip-flops** when confidence is low. Assign bow/draw roles by *relative position* (hand closer to screen center or farther from body = bow hand) with stickiness, rather than trusting left/right labels each frame.
- **Lighting matters a lot.** Test near a window and at night; mention it in the README.
- **Webcam auto-exposure** can tank FPS in low light (drops to 10–15 FPS). Lock camera settings via OpenCV if needed.
- **Fatigue is a design constraint**: holding arms up is tiring. Short rounds (60–90 s) are a feature, not a limitation.

## 8. Future Mobile Note (parking lot — not now)

Just to keep the door open: the architecture above already does the one thing that matters — the game logic only sees plain data (`HandFrame`, `BowState`), never MediaPipe. When the time comes, the realistic paths are (a) a **web version** using MediaPipe Tasks for JS, which runs in mobile browsers and would mean porting game logic to TypeScript, or (b) native MediaPipe on Android/iOS. Either way, nothing in this plan needs to change today.
