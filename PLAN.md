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

Webcam verified: **1280×720 @ 30.2 fps** through the threaded `Camera` class, opened with **`cv2.CAP_MSMF`** and `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` (set in `fletchflow/__init__.py`, before anything imports cv2). This reverses the original `CAP_DSHOW` choice — see §7.

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
│   ├── diagnostics.py      # --selfcheck: off-thread PNG writer, mean-rate verdict
│   ├── telemetry_report.py # python -m fletchflow.telemetry_report <csv>: playtest -> thresholds
│   ├── vision/
│   │   ├── camera.py       # done: threaded capture, latest-frame-only, fps/exposure pinning
│   │   ├── tracker.py      # done: MediaPipe wrapper → HandFrame (mirrors + swaps handedness)
│   │   ├── smoothing.py    # done: One Euro filter + HandSmoother
│   │   └── pipeline.py     # done: tracking thread (tracker + smoother, latest-wins)
│   ├── input/
│   │   ├── gestures.py     # pinch_ratio, fist_ratio, grip_point, palm_size
│   │   ├── bow_input.py    # state machine of §1/§4.6.1 → BowState
│   │   ├── calibration.py  # done: per-player thresholds + draw range (§4.6.4)
│   │   └── mapping.py      # camera → screen: BowPose + the sight pin
│   │                       #   (the ONLY camera↔screen boundary)
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
└── tests/                  # 112 green
    ├── test_gestures.py    # pinch/fist ratio + palm_size on synthetic landmarks
    ├── test_bow_state.py   # scripted sequences through every transition-table row
    ├── test_mapping.py     # mirror + scale, sight pin, aim stability floor
    ├── test_calibration.py # scripted calibration runs, incl. every rejection path
    ├── test_hud.py         # every HUD draw call, headless
    ├── test_telemetry.py   # CSV schema, incl. draw_grip / release_rule columns
    ├── test_telemetry_report.py # report maths, incl. the real state machine through the real logger
    ├── test_bow_render.py  # 2D fallback body at every depth scale
    ├── test_diagnostics.py # PNG writer colors/failure handling, mean rate, feed path
    ├── test_session.py / test_world.py / test_smoothing.py
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

**Revised 2026-09-16 — the AND has a counter-case, so the rule is now selectable.**
The AND is only load-bearing against a *pinch-only* rule. `fist_ratio` averages
all four fingers, and a pinch grip curls just the index. A relaxed pinch —
thumb on index, middle/ring/pinky loosely curled rather than extended — therefore
sits near `fist_ratio ≈ 1.3`, *below* `FIST_OFF = 1.60`. Open that pinch and
`both_open` never fires. That estimate is geometry, not a recording, like the
fist numbers above.

The machine already knows which grip took the string (`_draw_uses_grip`), so the
alternative needs no guessing:

| `RELEASE_RULE` | Pinch grip fires when | Fist grip fires when |
|---|---|---|
| `both_open` (default, as agreed) | `pinch > PINCH_OFF` **and** `fist > FIST_OFF` | `pinch > PINCH_OFF` **and** `fist > FIST_OFF` |
| `grip_aware` | `pinch > PINCH_OFF` | `fist > FIST_OFF` |

Both keep the `PINCH_OFF_FRAMES` debounce. `grip_aware` requires one of
`both_open`'s two conditions, so **it can never fire later** — it only releases
shots that `both_open` is holding back. Its one risk runs the other way: a
premature release. That needs a noise excursion across the full hysteresis gap
for `PINCH_OFF_FRAMES` frames, while the hand holds the opposite extreme of that
same ratio. **G** switches rules live, the debug overlay shows the active one,
and telemetry logs it on every row alongside the draw grip. That way one playtest
compares both, and `python -m fletchflow.telemetry_report` reports release
latency, blocked-open frames and stuck releases per grip and per rule.
Recommendation: try `grip_aware` first. It would have been the default had the
counter-case been spotted at design time.

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

Watch at the next playtest: the pin is screen-up while the *drawn arrow* renders
along `aim`, so when the draw hand sits well off to one side the arrow visibly
points somewhere other than the reticle. The shot still goes to the reticle —
`spawn_arrow` takes the aim point — but the two can disagree on screen. Confirmed
in `--selfcheck --fake-bow`, whose synthetic aim sweeps much wider than a real
draw ever does, so this may never show up in play.

#### 4.6.4 Calibration: measure now, map later

Implemented as `input/calibration.py`, **not** `game/calibration.py` as first
specced: it reads camera-space gesture ratios, and `game/` is supposed to see
nothing but `BowPose`.

Split deliberately. The measurement half is cheap and unblocks the constants
above; the aim-mapping half is speculative and stays in M6 where PLAN.md already
had it.

**In 4b — a 9 s routine (`CALIB_STEP_S = (2, 2, 2, 3)`), no targets and no fitting:**

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

#### 4.6.5 Bugs found while measuring and implementing

- **`render/bow.py:134` crashes the 2D fallback at any scale != 1.0.** The grip
  wrap passes `w * scale` (float) as a pygame line width, which requires an int:
  `TypeError: 'float' object cannot be interpreted as an integer`. `_draw_arrow`
  at line 164 already does this correctly (`max(2, round(4 * scale))`), so it is an
  oversight, not a convention. It is masked only because the moderngl body renders
  instead; on any machine without GL 3.3 the game dies as soon as the depth-scale
  EMA leaves 1.0, which is immediately. Fix: `int(round(w * scale))`.
- **`tests/test_mapping.py` does not exist** despite §3 listing it. The mirror and
  scale math is untested, and 4b adds the sight pin to that same module.
  FIXED in 4b — the file now covers mirror/scale, the pin, and aim stability.

Two more surfaced while building 4b:

- **`render/hud.py` had no importer anywhere in the test suite**, so a syntax
  error in it survived a fully green run and only failed at launch. FIXED:
  `tests/test_hud.py` exercises every drawing function headlessly.
- **`game/` already imports mediapipe and cv2 transitively**, via
  `game/session.py -> input/mapping -> input/bow_input -> input/gestures ->
  vision/tracker`. The isolation rule in §3 is therefore a convention, not
  something the import graph enforces. Nothing depends on breaking it today, but
  the mobile path in §8 assumes the boundary is real — worth a lint check, or
  moving `HandFrame` out of `vision/tracker` into a dependency-free module.

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

### 4.7 Milestone 4c — the bow as a 3D object: pose, aim and model (planned 2026-09-17)

Playtest feedback 2026-09-17, in the player's words: make the bow bigger; when
held, it should rotate with the knuckles so the string is easier to grab; the
crosshair "is still centered in the same spot above the bow rather than allowing
for actual aiming", and "our orientation tracking for aiming is super off and is
not 3D"; and "we need to change the model to be 3D".

All four have the same root. The bow has no 3D pose. The 4b sight pin is a fixed
screen offset from the grip, so rotating the bow never moves it. The moderngl
body is a real mesh, but it is projected orthographically and rotated only
about the screen normal, so it can only ever look like a shaded 2D arc. 4c gives
the bow a true 3D pose driven by the hands. The crosshair, the rendered arrow,
the string grab zone and the model are all derived from that one pose.

#### 4.7.1 What the playtest telemetry showed

`python -m fletchflow.telemetry_report` on the 428 s session:

| Signal | Measured | Consequence for 4c |
|---|---|---|
| Pinch-grip `fist_ratio` during draws | median **1.16**, below `FIST_OFF = 1.60` | Relaxed-pinch counter-case confirmed. **`RELEASE_RULE = "grip_aware"` becomes the default** |
| Releases under `both_open` | 2 stuck; latency p50 34 ms, **p90 537 ms**; 12.8% blocked-open rows | Same |
| Grabs → drops while held | **8 of 8**. 5 were "fist read open" for 6 frames, some from glitch spikes (`fist_ratio` 9.44, 5.24, 4.12); 3 were bow hand unseen > 400 ms | Longer drop debounce, longer bow-lost grace, glitch rejection (§4.7.7) |
| Draws → cancelled | **6 of 11**, all the draw hand lost > 200 ms. Two had hands overlapping in the image (separation 0.024–0.043); one had lasted 5.9 s | 3D aim requires both hands through the draw, so the draw must survive brief loss (§4.7.7) |
| Tracked fps with 2 hands | 29.6 | Two-hand detection cost is not a problem; retired |
| Depth `scale` | p50 **1.55**, p95 1.60 — pinned at `DEPTH_SCALE_MAX` | Partly the depth-proxy bug in §4.7.2 |
| Per-shot max pull | p50 1.64 hw over 5 shots | `DRAW_FULL` starts below the old 2.0 hw (§4.7.6) |

#### 4.7.2 Hand depth: replace apparent size with a metric fit

**The current depth proxy is rotation-sensitive, and part of that is a bug.**
MediaPipe normalizes x by image *width* and y by image *height*. `gestures.py`
takes Euclidean distances in those mixed units, so on a 16:9 frame the same
segment measures up to 1.78× longer vertical than horizontal. `palm_size`
depends on how the hand is turned, and so do depth, `scale` and the depth
term of power. Measured on synthetic hands rendered with full perspective:
across roll ±90° and pitch/yaw ±20° at a *fixed* distance, the depth implied
by `palm_size` swings **×1.84 / ×1.71 / ×1.61** at 0.35 / 0.45 / 0.60 m.

**Replacement: a weak-perspective Procrustes fit.** The Tasks API already
returns `hand_world_landmarks`: metres, hand-centred, camera-aligned. Fit the
5 palm points `POSE_PALM_POINTS = (0, 5, 9, 13, 17)`, which stay visible in a
fist:

```
img_i   = (x_i * W, y_i * H)                 px, W,H = CAPTURE_SIZE, x mirrored
world_i = (wx_i, wy_i)                       metres, wx mirrored (negated)
centre both sets; as complex numbers zi = img_c, zw = world_c
s        = sum(zi * conj(zw)) / sum(|zw|^2)  # similarity: scale and in-plane rotation
k        = |s|          px per metre
theta    = arg(s)       world -> image in-plane rotation; ~0 if axes are camera-aligned
residual = sqrt(mean(|zi - s*zw|^2))         px
f_px     = CAM_FOCAL_NORM * W
depth z  = f_px / k                          metres from the camera
position = ((u - W/2) * z / f_px, (v - H/2) * z / f_px, z) for the grip point (u, v) px
```

Using the complex modulus makes the fit indifferent to whether world axes turn
out to be camera-aligned. `theta` is logged so a playtest can confirm it.
Reject a frame (hold the last depth) when `residual > POSE_MAX_RESIDUAL_PX = 25`
or `z` falls outside `POSE_DEPTH_RANGE_M = (0.15, 2.5)`.

Measured on the same synthetic hands:

| | Result |
|---|---|
| Rotation swing at a fixed distance (same grid as above) | **×1.25 / ×1.19 / ×1.14** at 0.35 / 0.45 / 0.60 m |
| Worst depth error, noiseless, \|pitch\|,\|yaw\| ≤ 20° | 13.6% (weak-perspective error, worst at close range) |
| Depth error per frame, 1.5 px image + 4 mm world noise, random poses | p50 3.8–6.1%, p95 12–23% (before temporal smoothing) |
| Fit residual, moderate poses + noise | p50 7.5 px, p95 15.8 px, max 22 px — hence the 25 px gate |

Depth is smoothed with a One Euro filter: `POSE_DEPTH_MIN_CUTOFF = 0.8` Hz,
`POSE_DEPTH_BETA = 1.0` (Hz per m/s), `d_cutoff = 1.0` Hz.

**Refined 2026-09-18: full perspective, after reviewing phase 1.** Weak
perspective assumes all five palm points share one depth. The bow hand's natural
pose breaks that: a fist pointed at the camera puts the wrist ~9 cm behind the
knuckles, a 25% scale difference at 35 cm. The weak fit now only *seeds* a
Gauss-Newton fit of the hand's translation and in-plane angle `(Tx, Ty, Tz, phi)`
under full perspective, using the world depths too (`_fit_perspective`, 8
iterations, 5 points).

MediaPipe's world-depth sign convention can't be verified without a live hand,
so both signs are fitted and the lower residual wins. At small pitch they agree;
at large pitch the wrong one is clearly worse. `HandPose3D.fit` records which
won (`persp` / `persp_flipped` / `weak`) and telemetry logs it, so the first
playtest settles the convention. Solving `phi` jointly instead of borrowing the
weak fit's perspective-biased angle is what took noiseless error from 7.5% to 0.

| Measured, 1.5 px image + 4 mm world noise | Weak only | Refined |
|---|---|---|
| Noiseless worst error, every z (acceptance grid) | 13.6% at 0.35 m | **0.0%** |
| Rotation swing at a fixed 0.45 m | ×1.19 | **×1.000** |
| **0.35 m, fist toward camera (pitch 50–85°)** | **71–85% rejected**, p50 15% | **0.7% rejected**, p50 4.7%, p95 14.3% |
| Same, world depth sign inverted | — | same accuracy; picks `persp_flipped` |
| Same, world axes rotated 25° in-plane | — | same accuracy; reports −25.5° |

**Roll comes from the image, not the world landmarks.** World landmarks give
3D orientation, but with ~4 mm of noise the knuckle axis is only good to p50
5.8° / p95 ~12°. The image knuckle line, index MCP (5) minus pinky MCP (17), in
isotropic pixels, measured **p50 1.1° / p95 3.2°** per frame at 1.5 px noise.
`knuckle_dir = normalize((x5 - x17) * W, (y5 - y17) * H)`. A thumb-up fist
gives about (0, -1): up.

**Deliberately unchanged:** `pinch_ratio` and `fist_ratio` keep their current
(anisotropic) definition. Their thresholds were tuned in that space and are
working — pinch latency p50 34 ms. Only size and depth, where the error actively
harms play, move to the metric fit. The dock grab radius is unchanged too.

#### 4.7.3 The bow's 3D frame

Axes. Camera metric: +x right in the mirrored image, +y down, +z away from the
camera. Game: +x right, +y down, +z into the screen. The player faces the screen,
so a direction converts camera → game as `(x, y, -z)`. Game axes are right-handed
(x̂ × ŷ = ẑ).

| Axis (game space, unit) | While HELD | While DRAWN / RELEASED |
|---|---|---|
| `bow_forward` | (0, 0, 1) — into the screen | the gained, zeroed aim direction (§4.7.4) |
| `bow_up` | `(kx, ky, 0)` from `knuckle_dir`, orthogonalized against forward | same |
| `bow_right` | `cross(bow_up, bow_forward)` | same |

The up axis is smoothed by a One Euro filter per component, then renormalized
(`BOW_UP_MIN_CUTOFF = 1.5` Hz, `BOW_UP_BETA = 0.5`). If the orthogonalized up
degenerates (`|up| < 0.2`), keep the previous up. Forward eases between its
HELD and DRAWN sources: `forward = slerp(forward, target, 1 - exp(-dt / BOW_ORIENT_TAU_S))`,
`BOW_ORIENT_TAU_S = 0.08`, so nocking does not snap the bow.

`render_scale = clamp(REFERENCE_BOW_DEPTH_M / z_bow, *BOW_SCALE_RANGE)`, with
`REFERENCE_BOW_DEPTH_M = 0.55` and `BOW_SCALE_RANGE = (0.80, 1.25)`. The upper
clamp is tighter than `DEPTH_SCALE_MAX`: the playtest player sat close enough to
pin the old scale at 1.60, which would make a 480 px bow 768 px tall on a
720 px screen.

#### 4.7.4 Aiming along the arrow

A real arrow points from the nock to the arrow rest, so the aim is the 3D line
from the draw hand through the bow hand. Moving the bow arm, turning the torso or
shifting the draw hand all move the crosshair — the "actual aiming" the playtest
asked for.

```
d        = P_bow - P_draw                      camera metres, both from §4.7.2
a        = normalize(d.x, d.y, -d.z)           game direction; the bow hand is nearer the camera, so a.z > 0
baseline = |d|
weight   = clamp((baseline - AIM_BASELINE_MIN_M) / AIM_BASELINE_RAMP_M, 0, 1)
yaw      = atan2(a.x, a.z)
pitch    = atan2(a.y, hypot(a.x, a.z))
(yaw, pitch) -> One Euro: AIM_MIN_CUTOFF = 1.2 Hz, AIM_BETA = 1.8 (Hz per rad/s), d_cutoff 1.0 Hz
yaw'     = AIM_GAIN * (yaw - aim_yaw0)          AIM_GAIN = 1.5; zero offsets from calibration (§4.7.9)
pitch'   = AIM_GAIN * (pitch - aim_pitch0)
clamp |yaw'|   <= atan(AIM_SCREEN_MARGIN * (W_win/2) / FOCAL_PX)    AIM_SCREEN_MARGIN = 0.95
clamp |pitch'| <= atan(AIM_SCREEN_MARGIN * (H_win/2) / FOCAL_PX)
sight    = (CX + FOCAL_PX * tan(yaw'), CY + FOCAL_PX * tan(pitch') / cos(yaw'))
bow_forward target = (cos(pitch') sin(yaw'), sin(pitch'), cos(pitch') cos(yaw'))
```

`AIM_BASELINE_MIN_M = 0.05`, `AIM_BASELINE_RAMP_M = 0.05`: the direction is
meaningless while the hands are together at the nock, and fully trusted at a
10 cm pull.

- **The crosshair shows only while DRAWN**, at alpha `weight`; RELEASED holds
  the last sight. HELD and DOCKED show none: there is no arrow to aim yet.
  (Replaces the 4b sight pin entirely.)
- **You still hit where the crosshair is.** `aim_point()` returns `sight`, and
  `spawn_arrow` already converges the launch onto that point at
  `SIGHT_DEPTH_M`. If a shot fires with `weight == 0`, it goes straight ahead
  to the screen centre.
- **The rendered arrow points at the crosshair by construction.** It is drawn
  along `bow_forward`, and a line's vanishing point is the projection of its
  direction, which is `sight`. This closes 4b's reticle-versus-arrow mismatch.
- **Measured conditioning:** yaw recovered from two synthetic hands 0.40 m apart
  has p50 0.51° / p95 2.49° error per frame at 1.5 px noise, before smoothing.
  Lateral position comes from precise image x; depth error mostly changes the
  vector's length, not its angle.
- **Why the gain.** At `AIM_GAIN = 1.0` (true archery), reaching the screen
  edge takes a 35° turn, enough to carry hands out of frame. 1.5 needs 23°.
  A wrong `CAM_FOCAL_NORM` scales the depth axis only, so it shows up as an
  aim-gain error, and `AIM_GAIN` absorbs it.
- **Known geometric limit:** aiming at the camera lens itself lines the two
  hands up along the lens axis, and the rear hand is occluded. With the camera
  above the screen, that is the top edge; elsewhere perspective keeps the hands
  apart in the image.

#### 4.7.5 A bigger bow, and grabbing the string rather than the grip

`BOW_SPAN_PX = 340 → 480` at `render_scale = 1.0` (384–600 px across
`BOW_SCALE_RANGE`).

The string-grab test today is proximity to the *anchor*, so a bigger bow alone
would not make the string easier to grab. It becomes proximity to the *string
segment* as seen while HELD, in isotropic image-width units (x, y · H/W):

```
c        = anchor (bow grip point)
k        = knuckle_dir (bow hand, smoothed)
h        = (BOW_SPAN_PX / 2) / W_win * render_scale
segment  = [c - k*h, c + k*h]
grab if distance(draw grip point, segment) < STRING_GRAB_RADIUS * render_scale
STRING_GRAB_RADIUS = 0.07          # ~90 px either side of the whole string, not just the grip
```

The draw grip point is `pinch_point` for a pinch and `grip_point` for a fist,
as now.

#### 4.7.6 Draw power in metres

```
at string grab:  d0     = |P_bow - P_draw|
each drawn frame: pull_m = max(0, |P_bow - P_draw| - d0)
                  power  = clamp(pull_m / DRAW_FULL_M, 0, 1)
DRAW_FULL_M = 0.15   # playtest p50 max pull 1.64 hw ~ 0.14-0.15 m; calibration replaces
```

Calibration step 4 measures `DRAW_FULL_M` as the 90th-percentile `pull_m`,
clamped to `CALIB_DRAW_CLAMP_M = (0.08, 0.40)`. Telemetry gains `pull_m`; the
report reads it when present and falls back to `pull_hw`. The hand-width
constants (`DRAW_FULL_HW`, `DEPTH_HW_MAX`, `DRAW_SIZE_SMOOTHING`) retire with
the old formula.

#### 4.7.7 Robustness, from the playtest data

Bow state machine changes, against the §4.6.1 table:

| Row | Before | 4c |
|---|---|---|
| DRAWN → HELD (draw hand lost) | lost > 200 ms | lost > **`HAND_LOST_GRACE_MS = 600`**. While lost, freeze draw point, `pull_m` and aim at their last values. On reappearance, resume, and evaluate release normally (debounce counts reappeared frames only) |
| HELD/DRAWN → DOCKED (fist opens) | `fist_ratio > FIST_OFF` for 6 frames | for **`BOW_DROP_FRAMES = 12`** (400 ms). A reading above **`FIST_RATIO_GLITCH = 3.0`** is a tracking glitch: it neither advances nor resets the counter |
| HELD/DRAWN → DOCKED (bow hand lost) | > 400 ms | > **`BOW_LOST_MS = 900`** |
| DRAWN → RELEASED | `RELEASE_RULE = "both_open"` | **`"grip_aware"`** by default; G still switches |

#### 4.7.8 The 3D model and its render

**One geometry module, two renderers.** A new `render/bow_model.py` (numpy only,
no GL, fully unit-testable) builds the bow in model space and projects it. Both
the moderngl body and the 2D fallback consume it, so they cannot disagree about
where the tips are.

Model space, in metres. +Y = `bow_up`, +Z = `bow_forward`, +X = `bow_right`.
Length `L = BOW_SPAN_PX * BOW_RENDER_DEPTH_M / FOCAL_PX = 0.48 m`, with
`BOW_RENDER_DEPTH_M = 0.9`:

| Part | Geometry |
|---|---|
| Riser | Y ∈ ±0.17 L; flat section 0.045 L wide (X) × 0.07 L deep (Z); grip wrap Y ∈ ±0.06 L, darker; arrow shelf notch at Y = +0.02 L |
| Limbs | from Y = ±0.17 L to tips at ±0.50 L; elliptical cross-section tapering 0.06 L → 0.022 L wide and 0.018 L → 0.008 L thick; centreline Z(t) = −flex · t² + recurve · max(0, t − 0.8)², with t = 0 at the riser and 1 at the tip |
| Flex | `flex = BRACE_FLEX * L + power * DRAW_FLEX * L`, with `BRACE_FLEX = 0.10`, `DRAW_FLEX = 0.12`, `recurve = 0.35 L` |
| String | Cylinders r = 0.004 L from top tip → nock → bottom tip. Nock at (0, +0.02 L, −(0.18 L + power · 0.55 L)) |
| Arrow (DRAWN only) | Shaft r = 0.007 L from the nock along +Z, length 0.95 L; head cone 0.06 L; 3 fletches over the rear 0.12 L |

Placement: bow centre in world = `unproject(anchor_px, BOW_RENDER_DEPTH_M / render_scale)`,
so a bigger scale means nearer and larger, the perspective way.
World = centre + R · model, with R's columns = (`bow_right`, `bow_up`,
`bow_forward`) — a proper rotation, det +1. Projection is the game camera's own
pinhole, `FOCAL_PX = 900`, so the bow shares the world's perspective.

Render per frame, with no orientation cache: build the mesh (vectorized numpy),
project it, and take the screen bounding box of all vertices plus 8 px, clamped
to the window and to 900 × 900. Render into an FBO of that size with an off-axis
projection mapping exactly that pixel rectangle. Lighting is Lambert + Blinn
specular (shininess 24) + rim, with the key light from upper-left-front in view
space. Read back, `convert_alpha()`, blit at the box origin. Reuse the last
surface when the `BowPose` object is identical, since the 60 Hz loop reuses one
pose between ~30 Hz tracking updates.

Budget: bow render (build + GL + readback + convert + blit) **p95 ≤ 7 ms**;
render loop holds ≥ 58 fps. The 2D fallback draws the same projected
centrelines, string and arrow with pygame lines.

#### 4.7.9 Calibration: zeroing the sight

Step 5, "Draw and aim at the centre dot — hold" (2.0 s; `CALIB_STEP_S` gains a
fifth entry, total 11 s). The HUD draws a dot at (CX, CY). While DRAWN with
`weight == 1`, record the raw pre-gain, pre-zero `(yaw, pitch)`.
`aim_yaw0` / `aim_pitch0` are the medians. The step is rejected — defaults
kept, zero offsets — if fewer than `CALIB_MIN_SAMPLES`, or if either angle's
median absolute deviation exceeds `CALIB_AIM_MAX_MAD_DEG = 6.0` (unsteady).
This is the whole of the deferred M6 "aim mapping" that matters: offset plus
gain, with gain as a constant.

#### 4.7.10 Data shapes

```python
# vision/tracker.py
@dataclass(frozen=True)
class HandFrame:
    timestamp_ms: int
    left: np.ndarray | None               # (21,3) normalized image coords, x mirrored
    right: np.ndarray | None
    left_world: np.ndarray | None = None  # NEW (21,3) metres, hand-centred, camera axes, x mirrored
    right_world: np.ndarray | None = None

# input/hand_pose.py  (NEW, pure numpy)
@dataclass(frozen=True)
class HandPose3D:
    position_m: tuple[float, float, float]  # camera metres, grip point; +z away from camera
    depth_m: float                          # == position_m[2], unsmoothed
    px_per_m: float                         # Procrustes scale k
    residual_px: float                      # RMS fit residual, 5 palm points
    inplane_deg: float                      # Procrustes rotation; ~0 if world axes are camera-aligned
    fit: str = "weak"                       # persp / persp_flipped / weak: which model gave the depth
def estimate_hand_pose(image_pts, world_pts, grip_norm) -> HandPose3D | None

# input/gestures.py — HandGesture gains
    knuckle_dir: tuple[float, float] = (0.0, -1.0)  # isotropic-pixel unit vector, pinky MCP -> index MCP
    pose: HandPose3D | None = None

# input/bow_input.py — BowSnapshot gains (all defaulted)
    bow_position_m: tuple[float, float, float] | None = None   # smoothed
    draw_position_m: tuple[float, float, float] | None = None  # DRAWN only; frozen while the draw hand is lost
    knuckle_dir: tuple[float, float] = (0.0, -1.0)             # bow hand, smoothed
    pull_m: float = 0.0
    render_scale: float = 1.0

# input/mapping.py — BowPose gains (all defaulted)
    bow_forward: tuple[float, float, float] = (0.0, 0.0, 1.0)  # game axes, unit
    bow_up: tuple[float, float, float] = (0.0, -1.0, 0.0)      # game axes, unit, orthogonal to forward
    aim_weight: float = 0.0                                     # crosshair alpha, 0..1
    render_scale: float = 1.0
    # `sight` stays: now the aimed crosshair, or None when aim_weight == 0
class Mapper:
    def apply_calibration(self, result) -> None  # NEW: sets aim_yaw0 / aim_pitch0

# input/calibration.py — CalibrationResult gains
    aim_yaw0_deg: float = 0.0
    aim_pitch0_deg: float = 0.0
```

Telemetry gains `pull_m` and per-hand `depth_m`, `residual_px`, `inplane_deg`, `pose_fit`.

#### 4.7.11 Work split and acceptance criteria

Shared definitions — constants, the dataclass fields above, and
`Mapper.apply_calibration` as a no-op stub — land first, in one scaffolding
commit, so no two agents edit the same file. Phase 1 runs alone; phases 2 and 3
then run in parallel.

| Phase | Owns | Acceptance |
|---|---|---|
| **1 · Pose** | `vision/tracker.py`, new `input/hand_pose.py`, `input/gestures.py`, `vision/telemetry.py` (pose columns), tests | 1. Synthetic, full perspective, \|pitch\|,\|yaw\| ≤ 20°, roll ±90°, z ∈ {0.35, 0.45, 0.6, 0.8, 1.0} m: worst depth error ≤ 15% at every z and ≤ 9% at z ≥ 0.6 m; median ≤ 5% at every z (prototype: worst 13.6/10.4/7.7/5.8/4.6%, median 4.1/3.2/2.4/1.8/1.4% — weak-perspective error, systematic, largest up close). 2. Depth swing at a fixed 0.45 m ≤ ×1.25. 3. With 1.5 px + 4 mm noise over 400 random poses: depth error p50 ≤ 7%. 4. `knuckle_dir` roll error p95 ≤ 4° at 1.5 px noise. 5. Mirrored input gives the same depth. 6. No gameplay change: every existing test passes untouched |
| **2 · Input & aim** | `input/bow_input.py`, `input/mapping.py`, `game/session.py`, `input/calibration.py`, `vision/telemetry.py` (`pull_m`), `telemetry_report.py` (`pull_m`), tests | 7. Synthetic hands with the arrow line turned 10° right put `sight.x` at CX + 900·tan(15°) ± 2 px. 8. No sight while HELD; weight 1 at a 0.10 m baseline. 9. A 500 ms draw-hand loss does not cancel; 700 ms does. 10. 11 frames at `fist_ratio` 1.7 do not drop the bow, 12 do; a 9.44 frame does not count. 11. The string grabs at a point 0.20 from the anchor but 0.05 from the segment. 12. `grip_aware` is the default. 13. Calibration step 5 zeroes: after applying, the same aim puts the sight at (CX, CY) ± 2 px. 14. Real `BowStateMachine` → `TelemetryLogger` → report integration still passes |
| **3 · Model & render** | new `render/bow_model.py`, `render/bow3d.py`, `render/bow.py`, `render/hud.py`, `__main__.py`, tests | 15. For 20 random poses, projected tips from `bow_model` match the GL render's painted tip pixels within 3 px. 16. `--selfcheck 12 --fake-bow` → SELFCHECK OK, render ≥ 58 fps, and a new reported bow-render p95 ≤ 7 ms. 17. Frames show the bow from behind, with visible foreshortening when yawed ±20°. 18. The drawn arrow's vanishing point lies within 10 px of `sight`. 19. The 2D fallback renders every pose without exception |
| **Playtest** | — | 20. Pose residual p95 < 16 px; `inplane_deg` p50 within ±10° (camera-aligned axes); and `pose_fit` on the bow hand while HELD is ≥ 90% one of `persp` / `persp_flipped` — which one settles MediaPipe's world-depth sign convention. 21. ≥ 80% of draws end in a fire (5 of 11 before). 22. Zero stuck releases under `grip_aware`. 23. ≤ 1 drop per 5 grabs (8 of 8 before) |

## 5. Milestones with acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| 0 | ~~Environment~~ | ✅ Done 2026-07-05: mirrored feed in pygame window, camera 1280×720 @ 30.5 fps, headless smoke test passing |
| 1 | ~~Tracking~~ | ✅ Done 2026-07-13: 26–31 fps tracked, handedness verified live (after fixing a label swap — the Tasks API needs NO swap for raw input, contrary to legacy docs) |
| 2 | **Gestures + state machine** | Code done + unit tests green 2026-07-13 (every transition-table row, incl. glitch debounce and the fire-power window). Pending playtest: 20 consecutive pinch–release cycles → exactly 20 fires, zero false |
| 3 | **The Bow** | v2 done 2026-07-16 after playtest feedback: grab-based flow (docked bow + "Grab the bow!" prompt, anchor = bow-hand pinch point, string grab needs proximity, power = relative finger-scale pull) and a true-3D moderngl body with 2D fallback. Pending playtest: grab flow feel |
| 4 | ~~Firing + gallery~~ | DONE 2026-09-04: perspective world, arrows fly into the screen and shrink, plane-crossing collision (anti-tunnelling test), 3 depth targets, ring scoring, round state, X crosshair, `--telemetry`. 41 tests green |
| 4b | ~~Playtest fixes~~ | ✅ Designed + implemented 2026-09-08; playtested 2026-09-17 (findings in §4.7.1); spec in §4.6. (a) crosshair is a sight pin `CROSSHAIR_RISE_PX * scale` above the grip with its own heavy One Euro filter; (b) bow held by a closed fist, string takes a pinch **or** a fist and fires when the hand goes flat; (c) draw power in 3D, in hand-widths; plus a ~9 s measure-only calibration (`--calibrate`, or `C`) and four bug fixes (§4.6.5). Release rule switchable live with G (§4.6.1). 94 tests green. **Pending playtest**: does the fist grab feel better than the pinch, and does a real 3D draw reach full power? |
| 4c | **3D bow: pose, aim, model (NEXT)** | Planned 2026-09-17 from playtest feedback + telemetry; spec in §4.7. Metric hand depth from a Procrustes fit of MediaPipe world landmarks (rotation swing ×1.19 vs ×1.71 today); crosshair aimed along the 3D draw-hand→bow-hand line (replaces the 4b sight pin); bow rolls with the knuckles, 480 px, string grabbed anywhere along it; perspective-correct procedural 3D model with a 3D string and arrow; grip_aware default and loss-tolerant draws from the playtest data. Acceptance criteria 1–23 in §4.7.11 |
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
- **The capture backend was costing 3x the frame rate** (measured 2026-09-08).
  `CAP_DSHOW` sustained **10.0 fps** at 720p on this machine; `CAP_MSMF` sustains
  **31.7**, with identical detect cost. It is not a bandwidth limit — 640×480
  YUY2 also capped at 10.0 — nor a codec one, since forcing MJPG on DSHOW changed
  nothing. DSHOW was originally chosen because MSMF took ~20 s to deliver a first
  frame; setting `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` before cv2 is
  imported cuts that to **1.5 s**, which is *faster* than DSHOW's 3.4 s. So MSMF
  now wins on both axes. `config.PREFER_MSMF = False` restores the old path, and
  `Camera` falls back to DSHOW automatically if MSMF cannot open the device.
- **Never EMA a frame rate as `1/interval`** (fixed 2026-09-08). Both `camera.py`
  and `pipeline.py` smoothed the *reciprocal* of the inter-frame gap. The
  reciprocal is convex, so a few sub-millisecond gaps among many long ones drag
  the mean far above the true rate: a 10 fps camera reported **100–190 fps**, and
  the readout decayed slowly enough across runs to look plausible. That is what
  hid the backend problem for two months, and it made the HUD's "low camera fps"
  warning fire never. Smooth the interval and invert it instead.
- **In-app tracking runs at the camera's rate: ~30 fps** (measured 2026-09-16,
  15 s steady-state windows after a 5 s warmup, probes on both threads). This
  corrects an earlier entry here that claimed ~12 fps and blamed CPU contention
  with the render loop. Both halves were wrong: this machine has 16 cores, and
  with the render loop running at 60 fps the tracker still measured 29.4-29.9 fps
  — no different from the pipeline with no render loop at all (29.7). The ~12 fps
  came from `--selfcheck` itself, below. So §4.1 holds, and the input tuning,
  which is per *tracked frame* and assumes 30 Hz, stands.
- **`--selfcheck` was measuring its own stall.** It saved a PNG once per second
  with `pygame.image.save`, which holds the GIL for the entire encode — up to
  ~350 ms at 720p. That froze the camera and tracking threads with it: 14 gaps
  over 100 ms in 15 s, camera 22.1 / tracker 22.4 fps, and a SELFCHECK FAIL,
  against 0 such gaps, 29.0 / 29.9 fps and a pass with the saves disabled. Its
  verdict also read end-of-run EMAs, so whatever happened at the instant the run
  stopped decided pass or fail. The general rule this teaches: **nothing on the
  render thread may hold the GIL for long**, because detection's release of the
  GIL only helps while the other threads can actually get it back — a slow load,
  encode or screenshot on the main thread stalls hand input as well as frames.
- **Where render time actually goes** (60 fps, `--fake-bow`, no hands): loop body
  6.5 ms of the 16.7 ms budget, main thread busy 40%. `draw_bow` dominates at
  221 ms/s — inflated here because `--fake-bow` sweeps aim, power and scale
  continuously and so misses the 3D body's quantized cache far more than real
  play would — then the camera feed at 75 ms/s. Everything else together is
  under 30 ms/s. There is headroom; nothing here needs `TARGET_FPS` lowered.
- **Still unmeasured: two hands in frame.** Every number above had no hands in
  view, so detect cost ~14-19 ms. §2 measured ~29 ms with two hands, which is
  close to the camera's 33 ms frame interval — the one place tracking could
  genuinely fall behind the camera in play. `--telemetry` during a real session
  will show it.
- **Lighting**: face a window/lamp — helps both tracking quality and the frame-rate issue above.
- **MediaPipe VIDEO mode timestamp errors**: non-monotonic timestamps raise — always use the camera frame's own timestamp, never `time.time()` at call site.
- **numpy 2.x + mediapipe 0.10.35** verified compatible in our venv — don't "upgrade" pins blindly; re-run the smoke test after any dependency change.
- **Fatigue**: 60–90 s rounds max. This is a feature.

## 8. Future Mobile Note (parking lot — not now)

The `BowPose` boundary is the whole story: game logic never sees MediaPipe. Realistic future paths: (a) web version via MediaPipe Tasks for JS (runs in mobile browsers; game logic ported to TypeScript), or (b) native MediaPipe on Android/iOS. Nothing in this plan changes today either way.
