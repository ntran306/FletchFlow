# FletchFlow — Project Plan

A computer-vision archery game: your webcam tracks both hands, one hand "holds" a virtual bow, the other nocks and draws an arrow. Pull back, aim, release — hit targets on screen.

All numeric constants below are **starting values** — they live in `src/fletchflow/config.py` and get tuned during playtesting. But every one of them is a real number you can code against today.

---

## 1. Core Concept & Interaction Design

| Role | Which hand | Signal |
|---|---|---|
| **Draw hand** | The hand currently pinching (thumb tip + index tip) | Pinch = grab string, hand position = draw point, open pinch = fire |
| **Bow hand** | The other hand | Wrist position = bow anchor on screen |

Role assignment is dynamic: whichever hand starts the pinch becomes the draw hand, and roles are **sticky while DRAWN** (no re-evaluation mid-draw, even if handedness labels flicker).

### Key measurements (camera space, mirrored normalized coords)

Landmark indices used: `0` = wrist, `4` = thumb tip, `8` = index tip, `9` = middle-finger MCP (base knuckle — the most stable "palm center" reference).

- **Pinch ratio** (scale-invariant, works at any distance from camera):
  `pinch_ratio = dist(lm4, lm8) / dist(lm0, lm9)`
  Pinched ≈ 0.2–0.3, open hand ≈ 0.8–1.2.
- **Draw distance**: `d = dist(bow_wrist, draw_wrist)` in normalized coords.
- **Power**: `power = clamp((d − DRAW_MIN) / (DRAW_MAX − DRAW_MIN), 0, 1)` with `DRAW_MIN = 0.12`, `DRAW_MAX = 0.55`. Calibration (milestone 6) overwrites `DRAW_MAX` with 90% of the max separation observed during a 3-second "draw as far as you can" hold.
- **Aim direction**: unit vector from draw wrist → bow wrist, computed in screen space after mapping.
- **Fire power**: the **max** power over the last 5 frames before release — hands drift together during the release motion, so sampling at the release frame undershoots.

### Bow state machine (full transition table)

| From | To | Condition |
|---|---|---|
| `IDLE` | `ARMED` | both hands tracked for ≥ 5 consecutive frames |
| `ARMED` | `DRAWN` | either hand's `pinch_ratio < 0.35` for ≥ 3 consecutive frames (~100 ms at 30 fps) |
| `DRAWN` | `RELEASED` | draw hand's `pinch_ratio > 0.55` for ≥ 2 consecutive frames → emit `FireEvent` |
| `DRAWN` | `ARMED` | either hand lost for > 200 ms (draw cancelled, no fire) |
| `RELEASED` | `ARMED` | 300 ms cooldown elapsed |
| any | `IDLE` | both hands lost for > 500 ms |

The 0.35 / 0.55 gap is deliberate **hysteresis** — a single threshold stutter-fires at the boundary. The frame-count requirements debounce single-frame tracking glitches. Constants: `PINCH_ON = 0.35`, `PINCH_OFF = 0.55`, `PINCH_ON_FRAMES = 3`, `PINCH_OFF_FRAMES = 2`, `HAND_LOST_GRACE_MS = 200`, `IDLE_TIMEOUT_MS = 500`, `COOLDOWN_MS = 300`.

### Two coordinate spaces — the defining design decision

The camera is an **input device**, not the game world:

| Space | What lives there | Units |
|---|---|---|
| **Camera space** (input) | Landmarks, pinch detection, draw distance, bow state machine | Normalized \[0,1], **mirrored** |
| **Screen space** (game) | Bow anchor, arrow trajectory & physics, targets, hit detection, scoring | Pixels, 1280×720 |

**Mirroring is handled once, in the tracker**: when `config.MIRROR` is true, `tracker.py` outputs `x' = 1 − x` for every landmark and swaps MediaPipe's `Left`/`Right` handedness labels (they describe the unmirrored image). Everything downstream lives in mirrored space that matches what the player sees.

**Mapping is trivial by construction**: capture and window are both 16:9 (1280×720), so `screen = (x' · 1280, y · 720)`. If the window size ever diverges from the capture aspect, `mapping.py` letterboxes — that's the only module allowed to know about it.

Downstream benefits: physics and scoring are deterministic, unit-testable, ordinary 2D game code; input feel (smoothing, thresholds) tunes independently from game feel (arrow speed, gravity, target size); the camera feed is set dressing — rendered at 40% brightness as the background, with the game drawn crisp on top (not AR-overlaid).

## 2. Tech Stack — verified working 2026-07-05

| Layer | Package | Verified version |
|---|---|---|
| Runtime | Python (venv at `.venv`) | 3.12.13 (system 3.14 unsupported by MediaPipe; venv built with `py -V:Astral/CPython3.12.13`) |
| Hand tracking | `mediapipe` | 0.10.35 |
| Camera capture | `opencv-python` | 5.0.0 |
| Game engine | `pygame-ce` | 2.5.7 |
| Math | `numpy` | 2.5.1 |
| Model | `hand_landmarker.task` (float16) | 7.8 MB, in `assets/models/`, gitignored |

Webcam verified: **1280×720 @ 30.5 fps** through the threaded `Camera` class (opened with `cv2.CAP_DSHOW` — faster startup than MSMF on Windows).

Setup is done (see README for reproduction). MediaPipe tracker configuration to use in `tracker.py`:

```python
HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="assets/models/hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,        # detect_for_video(frame, timestamp_ms)
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

`RunningMode.VIDEO` requires strictly increasing integer timestamps in ms — derive them from `Frame.timestamp` (already `perf_counter`-based). Measured detect cost on this machine at 720p: **~19 ms with no hands, ~29 ms with two hands** (per-hand landmark inference dominates — downscaling the input to 640×360 saves under 3 ms, and 480×270 is *worse*). Tracking therefore runs on its **own thread** (`vision/pipeline.py`): `detect_for_video` releases the GIL during inference (verified 2026-07-06 — capture held ~27 fps alongside a tight detect loop), so capture, tracking, and rendering all run concurrently, and a 29 ms detect never blocks the 16.7 ms render frame.

## 3. Project Structure & Data Contracts

```
FletchFlow/
├── PLAN.md / README.md / pyproject.toml / requirements.txt
├── assets/
│   ├── models/hand_landmarker.task     # downloaded, gitignored
│   ├── sprites/                        # bow, arrow, targets (M3+)
│   └── sounds/                         # draw creak, twang, thunk (M5)
├── src/fletchflow/
│   ├── __main__.py         # entry point (done: M0 camera feed)
│   ├── config.py           # every constant named in this plan
│   ├── vision/
│   │   ├── camera.py       # done: threaded capture, latest-frame-only, fps/exposure pinning
│   │   ├── tracker.py      # done: MediaPipe wrapper → HandFrame (mirrors + swaps handedness)
│   │   ├── smoothing.py    # done: One Euro filter + HandSmoother
│   │   └── pipeline.py     # done: tracking thread (tracker + smoother, latest-wins)
│   ├── input/
│   │   ├── gestures.py     # pinch_ratio, hand-role assignment
│   │   ├── bow_input.py    # state machine of §1 → BowState
│   │   └── mapping.py      # camera → screen: BowPose (the ONLY camera↔screen boundary)
│   ├── game/
│   │   ├── entities.py     # Arrow, Target, Bow
│   │   ├── physics.py      # fixed-timestep projectile sim
│   │   ├── scoring.py      # ring scoring, round state
│   │   └── scenes.py       # MENU / CALIBRATION / PLAYING / GAME_OVER
│   └── render/
│       ├── renderer.py     # camera background + game layer
│       └── hud.py          # score, power bar, F1 debug overlay
└── tests/
    ├── test_gestures.py    # pinch_ratio math on synthetic landmarks
    ├── test_bow_state.py   # scripted pinch_ratio sequences through every table row above
    ├── test_mapping.py     # mirror + scale math, letterbox edge case
    └── test_physics.py     # trajectory apex/range vs closed-form projectile equations
```

### The data that crosses module boundaries

```python
@dataclass(frozen=True)
class HandFrame:            # tracker.py → gestures/bow_input; CAMERA space (mirrored)
    timestamp_ms: int
    left: np.ndarray | None   # (21, 3) normalized xyz, or None if not tracked
    right: np.ndarray | None

@dataclass(frozen=True)
class FireEvent:            # emitted once, on the DRAWN→RELEASED frame
    origin: tuple[float, float]     # px — bow anchor at release
    direction: tuple[float, float]  # unit vector, screen space
    power: float                    # 0..1 (max of last 5 frames)

@dataclass(frozen=True)
class BowPose:              # mapping.py → game/render; SCREEN space
    anchor: tuple[float, float]     # px, One-Euro-smoothed
    draw_point: tuple[float, float] # px (draw hand wrist)
    aim: tuple[float, float]        # unit vector
    power: float                    # 0..1
    state: BowState                 # IDLE/ARMED/DRAWN/RELEASED
    fire: FireEvent | None
```

`game/` receives only `BowPose`. It never imports mediapipe, cv2, or anything from `vision/` — enforce with a lint grep in CI later if desired.

## 4. Architecture

### 4.1 Pipeline

```
Camera thread     Tracking thread (pipeline.py)   Main thread (60 fps game loop)
┌──────────┐ latest ┌────────────────────┐ latest ┌────────┐ ┌───────┐ ┌──────┐ ┌──────┐
│ camera.py│ ─────► │ tracker → smoother │ ─────► │gestures│►│mapping│►│ game │►│render│
└──────────┘ Frame  └────────────────────┘ Hand-  └────────┘ └───────┘ └──────┘ └──────┘
  ~30 fps             ~30 fps               Frame   BowState   BowPose  entities pixels
                     (camera space)                (camera    (screen
                                                    space)     space)
```

Three threads, each handing the newest value to the next via a 1-slot latest-wins slot — nothing ever queues or blocks. This shape is measured, not aesthetic: detection (~29 ms with two hands) is as long as a whole camera frame interval, so on the capture thread it drops camera frames, and on the game thread it kills 60 fps rendering. On its own thread everything runs at full rate (`detect_for_video` releases the GIL). Between tracking updates the game reuses the last `BowPose` — fine at 30 Hz input / 60 Hz render.

### 4.2 Smoothing (`smoothing.py`)

**One Euro filter** on exactly 4 points per hand (wrist 0, thumb tip 4, index tip 8, middle MCP 9) — not all 21. Starting parameters: `min_cutoff = 1.5` Hz, `beta = 0.3`, `d_cutoff = 1.0` Hz. Lower `min_cutoff` = smoother but laggier at rest; higher `beta` = less lag during fast draws. Filter in camera space (before mapping), keyed per hand-role so a role swap resets the filter state.

### 4.3 Game loop & physics

- Render/game tick: `clock.tick(60)`, variable dt for animation.
- **Physics: fixed timestep** `dt = 1/120 s` with an accumulator; semi-implicit Euler (`v += g·dt` then `p += v·dt`). Deterministic → unit-testable.
- Arrow launch: `v0 = 600 + 1400 · power` px/s (600–2000), gravity `g = 500` px/s² downward (arcade-light so mid-power shots arc visibly).
- Arrow = tip point + 40 px trailing segment (for rendering and stick-in effect). Collision = tip vs target circle, checked per physics step (at 2000 px/s and dt=1/120, max step is ~17 px < bullseye diameter, so no tunneling).
- Despawn arrows 200 px past any screen edge.

### 4.4 Targets & scoring (screen space, `scoring.py`)

- Target = circle, radius 60 px. Ring scoring by distance from center at impact: ≤ 15 px → **10**, ≤ 35 px → **5**, ≤ 60 px → **2**.
- v1 layout: three static targets at `(0.75·W, 0.30·H)`, `(0.85·W, 0.55·H)`, `(0.70·W, 0.80·H)` — right side of screen, so a right-handed player draws naturally across the body (revisit after playtesting with `MIRROR` and lefties).
- Round = 10 arrows or 60 s, whichever first. Rounds are short **on purpose** — arms-up fatigue is a design constraint.

### 4.5 Scenes & debug overlay

Scene state machine: `MENU → CALIBRATION → PLAYING → GAME_OVER → MENU`, `P` pauses. Calibration (M6) measures `DRAW_MAX` and per-player pinch thresholds.

**F1 debug overlay** (build in M2, it pays for itself immediately): landmark dots, per-hand `pinch_ratio` as a number, current bow state name, power bar, draw-distance readout, and ms timings for camera/tracker/render.

## 5. Milestones with acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| 0 | ~~Environment~~ | ✅ Done 2026-07-05: mirrored feed in pygame window, camera 1280×720 @ 30.5 fps, headless smoke test passing |
| 1 | **Tracking** | Code done 2026-07-06, throughput verified headless: 30.7 fps tracked with no hands, 26.2 fps with two hands (≥ 25 bar met). Pending first playtest: handedness labels correct in the mirror; held-still fingertip jitter < 3 px |
| 2 | **Gestures + state machine** | F1 overlay shows live `pinch_ratio` and state transitions; 20 consecutive deliberate pinch–release cycles produce exactly 20 `RELEASED` transitions (zero false fires, zero missed) |
| 3 | **The Bow** | Bow sprite tracks the mapped anchor with no perceptible lag on smooth motion; string vertex follows draw point; power bar sweeps 0→1 over a full draw |
| 4 | **Firing** | Arrows launch along the aim line at power-scaled speed and arc under gravity; `test_physics.py` validates range/apex against closed-form projectile math; arrows stick where they land |
| 5 | **Game** | Full round: 3 targets, ring scoring (10/5/2), 10-arrow / 60 s round, score + best-score screen, release/hit sounds |
| 6 | **Feel & polish** | Calibration scene sets `DRAW_MAX` + pinch thresholds; moving targets (sine drift, amplitude 80 px, period 3 s); hit particles; difficulty ramp |

Milestones 0–2 are CV plumbing; the game starts at 3. Don't skip ahead — a janky bow makes everything after it unfun.

## 6. Skills to Learn (in order of need)

1. **Coordinate spaces** — normalized-mirrored camera coords vs pixels; where the single mirror flip happens and why handedness labels swap with it. Most CV-game bugs live here.
2. **The 21-landmark hand model** — specifically indices 0, 4, 8, 9 used above, and why MCP-9 beats the fingertips as a stable palm reference.
3. **One Euro filter** — what `min_cutoff` and `beta` each trade (rest smoothness vs motion lag). Read the original interactive demo page; implement it yourself in ~40 lines.
4. **Finite state machines with hysteresis and debouncing** — the §1 table is the worked example.
5. **Fixed-timestep game loops** — accumulator pattern, why physics dt is decoupled from render dt ("Fix Your Timestep" article).
6. **2D vector math** — normalize, scale, dot product; closed-form projectile range/apex (used directly in `test_physics.py`).
7. **Producer/consumer threading** — the 1-slot latest-value pattern already in `camera.py`; read that file and understand the lock.
8. **Profiling** — per-stage ms timers in the F1 overlay first, `cProfile` when something's mysterious.

Not needed: neural-net internals, CUDA, 3D math, web/mobile tech.

## 7. Known Gotchas

- **Hands crossing/overlapping** confuses tracking. The bow pose keeps hands apart naturally; the 200 ms lost-hand grace in the state machine covers brief dropouts.
- **Handedness labels flicker** at low confidence — which is why roles are assigned by *who pinches*, sticky during DRAWN, never by Left/Right labels.
- **The webcam silently halves its frame rate** (measured 2026-07-06, two mechanisms): (1) without an explicit `CAP_PROP_FPS` request the driver bistably negotiates a 15 fps low-light mode — varying between opens with identical code; `camera.py` always requests 30. (2) In a dim room, auto-exposure can still drop to ~16 fps mid-session; set `config.MANUAL_EXPOSURE = -5` (1/32 s) to pin 30 fps while developing at night, but don't leave it set in a bright room. The HUD shows a yellow warning whenever camera fps < 20.
- **Lighting**: face a window/lamp — helps both tracking quality and the frame-rate issue above.
- **MediaPipe VIDEO mode timestamp errors**: non-monotonic timestamps raise — always use the camera frame's own timestamp, never `time.time()` at call site.
- **numpy 2.x + mediapipe 0.10.35** verified compatible in our venv — don't "upgrade" pins blindly; re-run the smoke test after any dependency change.
- **Fatigue**: 60–90 s rounds max. This is a feature.

## 8. Future Mobile Note (parking lot — not now)

The `BowPose` boundary is the whole story: game logic never sees MediaPipe. Realistic future paths: (a) web version via MediaPipe Tasks for JS (runs in mobile browsers; game logic ported to TypeScript), or (b) native MediaPipe on Android/iOS. Nothing in this plan changes today either way.
