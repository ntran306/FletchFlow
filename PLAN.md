# FletchFlow — Project Plan

A computer-vision archery game: your webcam tracks both hands, one hand "holds" a virtual bow, the other nocks and draws an arrow. Pull back, aim, release — hit targets on screen.

All numeric constants below are **starting values** — they live in `src/fletchflow/config.py` and get tuned during playtesting. But every one of them is a real number you can code against today.

---

## 1. Core Concept & Interaction Design

**Grab-based interaction (v2, redesigned after playtest 2026-07-16):** the bow
waits DOCKED at `DOCK_POS = (0.5, 0.20)` under a pulsing "Grab the bow!"
prompt. Whichever hand pinches within `GRAB_RADIUS = 0.11` of it first
becomes the **bow hand** — the bow anchors to that hand's thumb+index pinch
point (not the wrist) and is held only while that pinch stays closed. The
other hand pinches within `STRING_GRAB_RADIUS = 0.11` of the bow to grab the
string (**draw hand**); opening that pinch fires. Roles come from grab order,
never from handedness labels, and are sticky until the bow is dropped.

### Key measurements (camera space, mirrored normalized coords)

Landmark indices used: `0` = wrist, `4` = thumb tip, `8` = index tip, `9` = middle-finger MCP (base knuckle — the most stable "palm center" reference).

- **Pinch ratio** (scale-invariant, works at any distance from camera):
  `pinch_ratio = dist(lm4, lm8) / dist(lm0, lm9)`
  Pinched ≈ 0.2–0.3, open hand ≈ 0.8–1.2. Engage threshold `PINCH_ON = 0.32`.
- **Power is a relative pull** (v2 — replaced wrist-span so full power needs
  only a finger-scale motion): at string grab, record baseline
  `d0 = dist(anchor, draw_pinch)`; then
  `power = clamp((dist(anchor, draw_pinch) − d0) / DRAW_RANGE, 0, 1)` with
  `DRAW_RANGE = 0.22` (tunable; calibration in milestone 6).
- **Aim direction**: unit vector from draw pinch → bow anchor, computed in screen space after mapping; docked/held bows point straight up.
- **Fire power**: the **max** power over the last 5 frames before release — hands drift together during the release motion, so sampling at the release frame undershoots.

### Bow state machine (full transition table)

| From | To | Condition |
|---|---|---|
| `DOCKED` | `HELD` | a pinch (`ratio < 0.32` for ≥ 3 frames) within `GRAB_RADIUS` of the dock → that hand becomes the bow hand |
| `HELD` | `DRAWN` | the other hand pinches (same debounce) within `STRING_GRAB_RADIUS` of the anchor → baseline `d0` recorded |
| `DRAWN` | `RELEASED` | draw hand's `pinch_ratio > 0.55` for ≥ 2 consecutive frames → emit `FireEvent` |
| `DRAWN` | `HELD` | draw hand lost for > 200 ms (draw cancelled, no fire) |
| `RELEASED` | `HELD` | 300 ms cooldown elapsed (→ `DOCKED` if the bow was dropped meanwhile) |
| `HELD`/`DRAWN` | `DOCKED` | bow hand's `pinch_ratio > 0.55` for ≥ 6 frames, or bow hand lost > 400 ms (bow drops; never fires) |

The 0.32 / 0.55 gap is deliberate **hysteresis** — a single threshold stutter-fires at the boundary — and the proximity gates are what stop casual pinches from hair-triggering a draw (the v1 hair-trigger was the top playtest complaint). Constants: `PINCH_ON = 0.32`, `PINCH_OFF = 0.55`, `PINCH_ON_FRAMES = 3`, `PINCH_OFF_FRAMES = 2`, `BOW_DROP_FRAMES = 6`, `HAND_LOST_GRACE_MS = 200`, `BOW_LOST_MS = 400`, `COOLDOWN_MS = 300`.

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
| 3D bow body | `moderngl` (GL 3.3, AMD Radeon verified) | 5.x — falls back to the 2D body if context creation fails |
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
│   │   ├── world.py        # done: perspective project/unproject (world <-> screen)
│   │   ├── entities.py     # done: Arrow, Target, spawn rules
│   │   ├── physics.py      # done: fixed-timestep flight + plane-crossing collision
│   │   └── session.py      # done: round state, aim point, scoring
│   └── render/
│       ├── bow.py          # done: shared geometry, 2D fallback body, string/arrow
│       ├── bow3d.py        # done: true-3D body — procedural mesh via moderngl,
│       │                   #   Lambert-lit, FBO readback, cached by (angle, flex)
│       └── hud.py          # done: power bar, grab prompt, F1 debug overlay
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

### 4.3 The perspective world (v3, redesigned 2026-09-04)

The flat side-on view was a **perspective mismatch**: the webcam sees the
player head-on, so the screen reads as a window, but the game was drawn as a
side view of archery. No amount of detail on the bow fixes that. The world is
now first-person, and the bow stays a prop held at the hand.

World space is **metres, eye at the origin**: +x right, +y DOWN (matching
screen convention), +z INTO the screen. Pinhole projection, `FOCAL_PX = 900`:

- `project(x,y,z) -> (CX + x*s, CY + y*s, s)` where `s = FOCAL_PX/z` — `s` is
  px-per-metre at that depth, so an object of world radius R draws at `R*s`.
  Returns `None` behind `NEAR_PLANE_M = 0.2`.
- `unproject(sx,sy,z)` inverts it, giving the world point at depth z on that
  view ray.

`game/world.py` owns both, and is the only place the two spaces meet.

### 4.4 Arrows, targets, scoring (`game/`)

- **Aim**: `aim_point(pose)` = the bow-hand anchor (`AIM_LEAD_GAIN = 0.0`, so
  you point at what you hit; raise it to make the draw hand lead the shot).
- **Launch** (retuned 2026-09-04 after playtest "the arrow just teleports"):
  the arrow spawns `ARROW_LAUNCH_DROP_PX = 380` px **below** the aim point at
  `ARROW_LAUNCH_Z_M = 1.5` and is aimed at the crosshair's world point at
  `SIGHT_DEPTH_M = 10`. The drop is not cosmetic — a shot fired straight down
  the eye ray projects to *the same pixel at every depth*, so tip and tail
  land on top of each other and the arrow renders as a zero-length line that
  shrinks in place. Launching low makes it climb into the crosshair, and that
  climb very nearly cancels gravity at full draw, the way a real sight does.
  Speed `9 + power*10` m/s, gravity `1.6` m/s² (arcade-light).
  Measured result — full draw bullseyes at every depth, weak draw arcs short:

  | depth | full draw | weak draw |
  |---|---|---|
  | 9 m  | 0.39 s (24 frames), 0.23r — 10 pts | 0.83 s, 0.71r — 2 pts |
  | 14 m | 0.66 s (39 frames), 0.06r — 10 pts | 1.39 s, 1.40r — miss |
  | 20 m | 0.97 s (58 frames), 0.03r — 10 pts | 2.06 s, 2.96r — miss |

- **Physics**: fixed `PHYSICS_DT = 1/120` accumulator, semi-implicit Euler,
  accumulator capped at 0.25 s so a stall cannot spiral.
- **Collision by plane crossing** — for a target at depth `tz`, when
  `prev_z < tz <= z` interpolate x/y at that plane and compare to the radius.
  Exact at any speed: a 200 m/s arrow that jumps 1.7 m in one step still
  registers (`test_physics.py` asserts precisely this).
- **Targets**: radius 0.9 m at depths `(9.0, 14.0, 20.0)` m — screen radii
  90/58/40 px, so depth is unmistakable. Respawn 0.8 s after a hit in the
  same depth band.
- **Readability**: a receding arrow foreshortens to nothing, so the renderer
  holds a `MIN_ARROW_PX = 28` on-screen streak. Hits burst for
  `HIT_FEEDBACK_S = 0.8`: white flash, two shockwave rings, rising `+N`.
- **Scoring** by radius fraction at the crossing: `<=0.28 -> 10`,
  `<=0.60 -> 5`, `<=1.00 -> 2`. Round = 10 arrows or 90 s, and it does not end
  until the last arrow lands.

Render order (`__main__`): camera feed -> world (far to near) -> bow ->
crosshair -> power bar -> HUD.

### 4.4b Crosshair

Four diagonal arms, **no centre connection**. The gap shrinks from
`CROSSHAIR_GAP_MAX_PX = 48` to `CROSSHAIR_GAP_MIN_PX = 13` as power rises, so
the sight visibly tightens as you pull; colour warms white -> gold. Dark
underlay keeps it readable over a bright camera feed.

### 4.5 Scenes & debug overlay

Scene state machine: `MENU → CALIBRATION → PLAYING → GAME_OVER → MENU`, `P` pauses. Calibration is split (see §4.6.4): the **measurement** half lands in 4b — per-player gesture thresholds and `DRAW_FULL_HW` — while the 5-point **affine aim mapping** stays in M6.

**F1 debug overlay** (build in M2, it pays for itself immediately): landmark dots, per-hand `pinch_ratio` as a number, current bow state name, power bar, draw-distance readout, and ms timings for camera/tracker/render.

### 4.6 Milestone 4b design (playtest fixes, planned 2026-09-05)

Three findings from the 2026-09-04 group playtest, plus two problems found while
measuring them. Constants marked **(unvalidated)** come from geometry, not from a
recording; §4.6.4 is what replaces them with measured numbers.

#### 4.6.1 Hand shapes: fist holds the bow, the string takes either grip

Testers disliked holding a *sustained* pinch for a whole round. The bow hand
becomes a **closed fist**. The string hand accepts **either** a pinch or a fist
and fires when the hand goes flat — accepting both costs one extra clause and
removes the need to guess which grip a player reaches for.

New per-hand measurements in `gestures.py` (camera space, all scale-invariant):

```python
@dataclass(frozen=True)
class HandGesture:
    wrist: tuple[float, float]
    pinch_point: tuple[float, float]   # thumb/index midpoint — string grip
    grip_point: tuple[float, float]    # NEW: palm centre — bow grip
    pinch_ratio: float
    fist_ratio: float                  # NEW
    size: float                        # wrist->MCP, kept: existing callers use it
    palm_size: float                   # NEW: rotation-robust size proxy
```

- `fist_ratio` = mean over the four fingers of `dist(tip, wrist) / dist(mcp, wrist)`,
  using pairs (8,5), (12,9), (16,13), (20,17). Open hand ~1.9-2.3, fist ~0.7-1.1.
  Numerator and denominator foreshorten together, so it survives hand rotation.
- `grip_point = wrist + 0.60 * (middle_mcp - wrist)` — where a riser sits in the palm.
- `palm_size = max(dist(0,9), dist(5,17) / PALM_WIDTH_RATIO)`, `PALM_WIDTH_RATIO = 0.85`.
  Rotating the hand about its wrist axis foreshortens 0->9 but not 5->17, and
  tilting does the reverse; the max of the two is far steadier than either alone.
  This matters because §4.6.2 divides by it — a false size reads as false depth.

`KEY_LANDMARKS` grows from 4 to 10: `(0, 4, 5, 8, 9, 12, 13, 16, 17, 20)`. The One
Euro filter is elementwise, so this is a (10,2) array instead of (4,2) — negligible.

New constants: `FIST_ON = 1.25`, `FIST_OFF = 1.60` **(unvalidated)**,
`FIST_ON_FRAMES = 3`, `FIST_OFF_FRAMES = 2`.

Revised transition table — **bold** marks what changed from §1:

| From | To | Condition |
|---|---|---|
| `DOCKED` | `HELD` | **`fist_ratio < FIST_ON`** for `FIST_ON_FRAMES`, **`grip_point`** within `GRAB_RADIUS * scale` of `DOCK_POS` -> that hand becomes the bow hand |
| `HELD` | `DRAWN` | other hand **`pinch_ratio < PINCH_ON` OR `fist_ratio < FIST_ON`** for its debounce, within `STRING_GRAB_RADIUS * scale` of the anchor -> baselines `d0`, `s0` recorded |
| `DRAWN` | `RELEASED` | **`pinch_ratio > PINCH_OFF` AND `fist_ratio > FIST_OFF`** for `PINCH_OFF_FRAMES` -> emit `FireEvent` |
| `DRAWN` | `HELD` | draw hand lost > `HAND_LOST_GRACE_MS` (cancel, no fire) |
| `RELEASED` | `HELD` | `COOLDOWN_MS` elapsed (-> `DOCKED` if the bow was dropped) |
| `HELD`/`DRAWN` | `DOCKED` | **bow hand `fist_ratio > FIST_OFF`** for `BOW_DROP_FRAMES`, or bow hand lost > `BOW_LOST_MS` |

The **AND** on the release row is load-bearing, not belt-and-braces. In a tight
fist the thumb lies across the fingers, so `pinch_ratio` sits around 0.3-0.5 —
straddling `PINCH_ON = 0.32` and *below* `PINCH_OFF = 0.55`. A fist-grip release
gated on `pinch_ratio` alone could park at 0.45 and never fire at all. Requiring
both ratios open means "the hand is flat", which is true of every release
regardless of which grip started it.

The bow anchor becomes `grip_point` rather than `pinch_point`.

#### 4.6.2 Draw power measured in 3D

Power was a 2D screen distance, but the draw hand moves *back toward the face* —
i.e. mostly in depth — so a correct draw built almost no power. MediaPipe's
per-landmark `z` is wrist-relative and not comparable between hands, so depth has
to come from apparent size. Expressing both terms in **hand-widths** makes them
commensurable and camera-distance-invariant:

```
lateral_hw = (dist2d(anchor, draw_grip) - d0) / s_bow_smoothed
depth_hw   = CAM_FOCAL_NORM * (1.0 / s_draw_smoothed - 1.0 / s0)
pull_hw    = hypot(max(lateral_hw, 0.0), max(depth_hw, 0.0))
power      = clamp(pull_hw / DRAW_FULL_HW, 0.0, 1.0)
```

where `s_*` are `palm_size` values, `s0` is the draw hand's `palm_size` at string
grab, and `d0` is the 2D anchor->draw distance at string grab.

Derivation: with `s = f * S / z` for physical hand size `S` at depth `z`, a depth
change is `dz / S = f * (1/s - 1/s0)` — hand-widths, with `f` the focal length in
normalized-x units. `CAM_FOCAL_NORM = 0.87` for a 60 deg webcam **(unvalidated)**;
getting it wrong is a gain error on the depth term only, and does not break
invariance.

`DRAW_FULL_HW = 2.0` is not a new guess — it is the existing tuning restated:
`DRAW_RANGE / REFERENCE_HAND_SIZE = 0.22 / 0.11 = 2.0`. Feel is preserved.

Verified numerically before adopting: a 0.20 m pure-depth draw reads 2.22 hw at
0.6 m from the camera and 2.23 hw at 1.2 m, so power is seating-distance
invariant. `DRAW_SIZE_SMOOTHING = 0.25` (EMA per tracked frame) damps `1/s`,
which amplifies size noise at distance. Clamp `depth_hw` to `[0, 3.0]`.

**Consequence that must be handled: `pose.aim` degrades.** `mapping.py` derives
aim from the 2D anchor->draw separation, guarded only by `length > 1.0` px. A
correct 3D draw collapses that separation toward zero, so aim will jitter or
freeze — and aim drives *bow orientation* and the drawn arrow's direction, so the
bow would spin. Fix: raise the guard to `AIM_MIN_SEPARATION_PX = 25` and hold the
last good aim below it. Measured impact of ignoring this: at `scale = 1.6` the
drawn arrow's tip travels from 266 px above the anchor at 150 px separation to
406 px at 10 px separation.

#### 4.6.3 The crosshair comes off the bow: a sight pin

`aim_point()` returned `pose.anchor`, so reticle and bow were the same point and
aiming meant putting your hand literally on the target — high, near the frame
edge where tracking drops it.

Occlusion was **not** the problem: `__main__.py` already draws the crosshair after
the bow, so it was never covered. The fix is a real offset, not a z-order change.

The reticle becomes a sight pin at `CROSSHAIR_RISE_PX * pose.scale` px above the
anchor in **screen** space (screen-up, not `-aim`, which is exactly the unstable
vector §4.6.2 describes), then low-passed by its own One Euro filter so it is
visibly steadier than the shaking bow.

`CROSSHAIR_RISE_PX = 110` (tunable). Measured basis: the bow body paints exactly
`75 * scale` px above the anchor (linear, confirmed at scale 0.55/0.8/1.0/1.3/1.6),
so 110 clears it with ~45% margin at every depth. Ergonomically it puts a
comfortably-held hand's reticle in the y ~= 240-480 px band where targets actually
sit, letting the hand rest lower.

Reticle smoothing: `RETICLE_MIN_CUTOFF = 0.6` Hz, `RETICLE_BETA = 0.08` — much
heavier than the 1.5 Hz / 0.3 used for the hands, so the sight has weight.

`BowPose` gains `sight: Vec2 | None` (px, pin-offset and smoothed). `aim_point()`
returns `pose.sight`. `mapping.py` computes it — the pin is a camera->screen
concern and that module owns the boundary. `game/` still consumes only `BowPose`.

**This is a partial fix and should be stated as one.** The pin supplies an
*offset*, so the hand can rest lower. It does not supply a *gain*, so it does not
fix reach — a player whose comfortable sweep spans 40% of the frame still cannot
cover the screen. Gain is what §4.6.4's deferred half provides.

#### 4.6.4 Calibration: measure now, map later

Split deliberately. The measurement half is cheap and unblocks the constants
above; the aim-mapping half is speculative and stays in M6 where PLAN.md already
had it.

**In 4b — a ~12 s routine, no targets and no fitting:**

| Step | Prompt | Duration | Yields |
|---|---|---|---|
| 1 | "Open both hands flat" | 2.0 s | `fist_ratio` / `pinch_ratio` open distributions |
| 2 | "Close both into fists" | 2.0 s | closed `fist_ratio` distribution |
| 3 | "Pinch thumb and finger" | 2.0 s | closed `pinch_ratio` distribution |
| 4 | "Grab the bow and draw as far as is comfortable, hold" | 3.0 s | per-player `DRAW_FULL_HW` |

Thresholds come from the medians, not the extremes:
`FIST_ON = open_med - 0.65 * (open_med - closed_med)` and
`FIST_OFF = open_med - 0.30 * (open_med - closed_med)`; same shape for pinch. That
keeps the hysteresis gap proportional to the player's own separation. If
`open_med - closed_med < 0.45` the hands are not separating enough to be reliable —
reject and keep the defaults. `DRAW_FULL_HW` = 90th percentile of `pull_hw` during
step 4, clamped to `[1.2, 4.0]`.

**Deferred to M6:** the 5-point affine aim mapping (centre + four corners inset to
20%/80%, `[sx,sy] = A @ [ux,uy,1]` by least squares, rejected if
`det(A[:2,:2]) <= 0`, either singular value outside `[0.5, 3.0]`, or RMS residual
> 60 px). Two reasons to wait: nobody has actually reported a reach problem, and
the pin offset must be settled first or the fit absorbs it and it gets applied
twice.

Calibration must **draw and hold, never shoot**. Impact point is contaminated by
power, gravity and release flinch: §4.4's own table shows a weak draw at 14 m
landing 1.40r off — 1.26 m of error injected purely by how hard the player pulled.
Sample `pose.anchor` only while `state == DRAWN`, take the median over the window.

#### 4.6.5 Two bugs found while measuring

- **`render/bow.py:134` crashes the 2D fallback at any scale != 1.0.** The grip
  wrap passes `w * scale` (float) as a pygame line width, which requires an int:
  `TypeError: 'float' object cannot be interpreted as an integer`. `_draw_arrow`
  at line 164 already does this correctly (`max(2, round(4 * scale))`), so it is an
  oversight, not a convention. It is masked only because the moderngl body renders
  instead; on any machine without GL 3.3 the game dies as soon as the depth-scale
  EMA leaves 1.0, which is immediately. Fix: `int(round(w * scale))`.
- **`tests/test_mapping.py` does not exist** despite §3 listing it. The mirror and
  scale math is untested, and 4b adds the sight pin to that same module.

#### 4.6.6 Acceptance criteria

1. 20 consecutive grab->draw->release cycles yield exactly 20 fires and 0 false
   fires, for a fist string grip and a pinch string grip alike.
2. A 0.20 m pure-depth draw reaches `power >= 0.95` at both 0.6 m and 1.2 m from
   the camera, within 10% of each other.
3. The reticle is never within 40 px of the anchor at any scale in [0.55, 1.60].
4. The 2D fallback body renders without exception at scale 0.55/0.8/1.0/1.3/1.6.
5. Bow orientation stays stable through a full 3D draw — no visible spin as the
   on-screen hand separation passes below 25 px.
6. Calibration completes in under 15 s and either tightens every threshold or
   reports which step failed and keeps the defaults.
7. Test suite green, including a new `test_mapping.py` and fist/hybrid-grip rows
   in `test_bow_state.py`.

## 5. Milestones with acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| 0 | ~~Environment~~ | ✅ Done 2026-07-05: mirrored feed in pygame window, camera 1280×720 @ 30.5 fps, headless smoke test passing |
| 1 | ~~Tracking~~ | ✅ Done 2026-07-13: 26–31 fps tracked, handedness verified live (after fixing a label swap — the Tasks API needs NO swap for raw input, contrary to legacy docs) |
| 2 | **Gestures + state machine** | Code done + unit tests green 2026-07-13 (every transition-table row, incl. glitch debounce and the fire-power window). Pending playtest: 20 consecutive pinch–release cycles → exactly 20 fires, zero false |
| 3 | **The Bow** | v2 done 2026-07-16 after playtest feedback: grab-based flow (docked bow + "Grab the bow!" prompt, anchor = bow-hand pinch point, string grab needs proximity, power = relative finger-scale pull) and a true-3D moderngl body with 2D fallback. Pending playtest: grab flow feel |
| 4 | ~~Firing + gallery~~ | DONE 2026-09-04: perspective world, arrows fly into the screen and shrink, plane-crossing collision (anti-tunnelling test), 3 depth targets, ring scoring, round state, X crosshair, `--telemetry`. 41 tests green |
| 4b | **Playtest fixes (NEXT)** | Designed 2026-09-05 — full spec in §4.6. (a) crosshair becomes a sight pin above the grip, `CROSSHAIR_RISE_PX = 110`, separately smoothed; (b) bow held by a closed fist, string hand accepts a pinch **or** a fist and fires when the hand goes flat; (c) draw power computed in 3D in hand-widths, `DRAW_FULL_HW = 2.0`; plus a ~12 s measure-only calibration that replaces the guessed gesture thresholds, and two bug fixes (§4.6.5). Acceptance criteria in §4.6.6 |
| 5 | **Aim pose + polish** | Analyse `--telemetry` CSV: does "hands converged + size ratio high" reliably precede losing the rear hand? If so add an `AIMING` state that treats occlusion as intent, with release detected on the draw hand reappearing open. Plus sounds and a best-score screen |
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
