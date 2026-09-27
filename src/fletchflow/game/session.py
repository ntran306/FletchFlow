"""Round state for the shooting gallery.

Consumes only BowPose (screen space) — never MediaPipe, never camera coords.
"""

from __future__ import annotations

import random

from fletchflow import config
from fletchflow.game.entities import Arrow, Target, random_target_pos, spawn_arrow, spawn_targets
from fletchflow.game.physics import Hit, step_arrows
from fletchflow.input.mapping import BowPose

MAX_CATCHUP_S = 0.25  # cap the accumulator so a long stall cannot spiral


def aim_point(pose: BowPose | None) -> tuple[float, float] | None:
    """Where on screen the shot will go: the sight, not the bow hand.

    M4c (PLAN.md §4.7.4): the sight now follows the real 3D line from the draw
    hand through the bow hand, and is None whenever aim_weight == 0 — HELD or
    DOCKED (no arrow strung), or a DRAWN frame still too close to the nock to
    trust. Falling back to the window centre, not the bow hand, means a shot
    fired before the aim is established goes straight ahead rather than
    snapping to wherever the bow happens to be held.
    """
    if pose is None:
        return None
    if pose.sight is not None:
        return pose.sight
    return (config.WINDOW_SIZE[0] / 2.0, config.WINDOW_SIZE[1] / 2.0)


class GallerySession:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self.targets: list[Target] = spawn_targets(self._rng)
        self.arrows: list[Arrow] = []
        self.score = 0
        self.arrows_left = config.ROUND_ARROWS
        self.elapsed = 0.0
        self.recent_hits: list[tuple[Hit, float]] = []  # (hit, time it landed)
        self._accum = 0.0
        self._last_fire = None  # dedupe: one pose is reused across render ticks

    @property
    def finished(self) -> bool:
        if self.elapsed >= config.ROUND_SECONDS:
            return True
        # let the last arrow land before calling the round
        return self.arrows_left <= 0 and not any(a.alive for a in self.arrows)

    def update(self, pose: BowPose | None, dt: float) -> None:
        if self.finished:
            return
        self.elapsed += dt

        if pose is not None and pose.fire is not None and pose.fire is not self._last_fire:
            self._last_fire = pose.fire
            target_point = aim_point(pose)
            if target_point is not None and self.arrows_left > 0:
                self.arrows.append(spawn_arrow(target_point, pose.fire.power))
                self.arrows_left -= 1

        self._accum = min(self._accum + dt, MAX_CATCHUP_S)
        while self._accum >= config.PHYSICS_DT:
            self._accum -= config.PHYSICS_DT
            for hit in step_arrows(self.arrows, self.targets, config.PHYSICS_DT):
                self.score += hit.points
                self.recent_hits.append((hit, self.elapsed))
                hit.target.respawn_at = self.elapsed + config.TARGET_RESPAWN_S

        self._respawn_targets()
        self.arrows = [a for a in self.arrows if a.alive]
        self.recent_hits = [h for h in self.recent_hits if self.elapsed - h[1] < 1.0]

    def _respawn_targets(self) -> None:
        for target in self.targets:
            if not target.alive and target.respawn_at is not None:
                if self.elapsed >= target.respawn_at:
                    target.pos = random_target_pos(target.pos[2], self._rng)
                    target.alive = True
                    target.respawn_at = None
