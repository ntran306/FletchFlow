"""Offline analysis of playtest telemetry CSVs (PLAN.md 4.6.4 / M4b).

`vision/telemetry.py` logs one row per tracked camera frame while the bow
state machine (`input/bow_input.py`) runs. This module turns a recorded CSV
into a report: session health (fps, tracking gaps, skipped/malformed rows),
time spent in each bow state, transition counts (grabs/nocks/fires/cancels/
cooldowns/drops), shot quality, bow-hand grip quality, per-grip release
responsiveness for the draw hand (pinch vs. fist, and by release_rule when
recorded), and suggested replacements for the still-unvalidated
FIST_ON/FIST_OFF/PINCH_ON/PINCH_OFF/DRAW_FULL_HW constants -- using the exact
formula `input/calibration.py` uses live, just applied after the fact to a
whole session instead of a 9 s scripted routine.

Run as:

    python -m fletchflow.telemetry_report path/to/playtest.csv

Standard library only, plus fletchflow.config -- no pandas/numpy/cv2/pygame/
mediapipe, so it runs instantly and never touches the camera. The required
20-column schema (t_ms..fired_power) is pinned as REQUIRED_COLUMNS rather
than read from the live fletchflow.vision.telemetry.COLUMNS, which grows as
optional columns (draw_grip, release_rule) are added; a CSV recorded before
those existed still loads. A truncated last line from a killed game (or any
row with a non-integer t_ms or an unrecognized state) is skipped and counted
rather than crashing.

Nothing here feeds gameplay.
"""

from __future__ import annotations

import csv
import statistics
import sys
from dataclasses import dataclass

from fletchflow import config

# The 20 columns every telemetry CSV has always had, t_ms..fired_power.
# Deliberately NOT derived from fletchflow.vision.telemetry.COLUMNS, which
# grows over time (draw_grip, release_rule appended after fired_power) --
# those two are optional, so a pre-existing CSV without them must still load.
REQUIRED_COLUMNS = (
    "t_ms", "state", "left_seen", "right_seen", "left_size", "right_size",
    "pinch_dist_2d", "size_ratio", "bow_side", "draw_side",
    "left_pinch", "right_pinch", "left_fist", "right_fist",
    "left_palm", "right_palm", "power", "pull_hw", "scale", "fired_power",
)

_VALID_STATES = frozenset({"docked", "held", "drawn", "released"})

# States that persist a grabbed bow (bow_side stays set through all three).
_BOW_HELD_STATES = ("held", "drawn", "released")

_TRANSITION_NAMES = {
    ("docked", "held"): "grabs",
    ("held", "drawn"): "nocks",
    ("drawn", "released"): "fires",
    ("drawn", "held"): "cancels",
    ("held", "docked"): "drops_while_held",
    ("drawn", "docked"): "drops_while_drawn",
    ("released", "docked"): "docked_after_release",
    ("released", "held"): "cooldowns",
}


# -- loading ---------------------------------------------------------------


def _is_valid_row(row: dict) -> bool:
    """A row is usable if t_ms parses as an integer and state is recognized.

    A killed game can leave a truncated last CSV line; csv.DictReader fills
    any fields that line is short of with None rather than raising, so this
    (not a try/except around int(row["t_ms"])) is what keeps that from
    surfacing as a crash several calls later.
    """
    t_ms = row.get("t_ms")
    if t_ms is None:
        return False
    try:
        int(t_ms)
    except (TypeError, ValueError):
        return False
    return row.get("state") in _VALID_STATES


class _LoadedRows(list):
    """list[dict] of well-formed rows; skipped_rows counts what load() dropped."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.skipped_rows = 0


def load(path) -> list[dict]:
    """Read a telemetry CSV into a list of `csv.DictReader` rows.

    Raises ValueError (never a raw OSError/KeyError) naming what's wrong if
    the file can't be opened, or if it's missing any of REQUIRED_COLUMNS --
    e.g. a pre-2026-09-08 CSV that stops at `draw_side`. draw_grip and
    release_rule are optional and never trigger this.

    Rows that fail _is_valid_row are dropped rather than raising; the count
    is carried on the returned list as `.skipped_rows` (analyze() also
    re-derives it defensively, so it's correct even if rows reach analyze()
    some other way).
    """
    try:
        handle = open(path, "r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"cannot read telemetry CSV {path!r}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                "telemetry CSV is missing required columns (old format?): "
                + ", ".join(missing)
            )
        result = _LoadedRows()
        for raw_row in reader:
            if _is_valid_row(raw_row):
                result.append(raw_row)
            else:
                result.skipped_rows += 1
        return result


# -- small helpers -----------------------------------------------------------


def _pct(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (same approach as calibration._percentile)."""
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _percentiles(values: list[float], ps: tuple[float, ...]) -> dict[str, float] | None:
    """None (insufficient data) for an empty sample; otherwise {"p5": ..., ...}."""
    if not values:
        return None
    return {f"p{int(p)}": _pct(values, p) for p in ps}


def _fraction(values: list, predicate) -> float | None:
    if not values:
        return None
    return sum(1 for v in values if predicate(v)) / len(values)


def _parse_float(row: dict, key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _s(row: dict, key: str) -> str:
    """A string field, defaulting to "" whether the value is missing or None.

    dict.get(key, "") only substitutes when the key itself is absent; a row
    truncated partway through still has every column as a key, just mapped
    to None from that point on, so callers that want draw_side/bow_side/
    draw_grip/release_rule to never be None go through this instead.
    """
    return row.get(key) or ""


def _seen(row: dict, side: str) -> bool:
    return row.get(f"{side}_seen") == "1"


def _has_column(rows: list[dict], name: str) -> bool:
    return bool(rows) and name in rows[0]


# -- section 1: session -----------------------------------------------------


def _session_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n < 2:
        return {
            "row_count": n,
            "duration_s": None,
            "tracked_fps": None,
            "gap_count": 0,
            "max_gap_ms": None,
            "fps_by_hands": {0: None, 1: None, 2: None},
        }

    t = [int(r["t_ms"]) for r in rows]
    duration_s = (t[-1] - t[0]) / 1000.0
    tracked_fps = (n - 1) / duration_s if duration_s > 0 else None

    gap_count = 0
    max_gap = 0.0
    intervals = {0: 0, 1: 0, 2: 0}
    seconds = {0: 0.0, 1: 0.0, 2: 0.0}
    for i in range(1, n):
        dt_ms = t[i] - t[i - 1]
        max_gap = max(max_gap, dt_ms)
        if dt_ms > 100:
            gap_count += 1
        hands = int(_seen(rows[i], "left")) + int(_seen(rows[i], "right"))
        intervals[hands] += 1
        seconds[hands] += dt_ms / 1000.0

    fps_by_hands = {
        k: (intervals[k] / seconds[k]) if seconds[k] > 0 else None for k in (0, 1, 2)
    }

    return {
        "row_count": n,
        "duration_s": duration_s,
        "tracked_fps": tracked_fps,
        "gap_count": gap_count,
        "max_gap_ms": max_gap,  # always reported now, not just when > 100ms
        "fps_by_hands": fps_by_hands,
    }


# -- section 2: state durations ----------------------------------------------


def _state_seconds(rows: list[dict]) -> dict:
    seconds = {"docked": 0.0, "held": 0.0, "drawn": 0.0, "released": 0.0}
    for i in range(1, len(rows)):
        dt_s = (int(rows[i]["t_ms"]) - int(rows[i - 1]["t_ms"])) / 1000.0
        state = rows[i - 1]["state"]
        if state in seconds:
            seconds[state] += dt_s
    return seconds


# -- section 3: transitions --------------------------------------------------


def _transitions(rows: list[dict]) -> dict:
    counts = {name: 0 for name in _TRANSITION_NAMES.values()}
    counts["other"] = 0
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1]["state"], rows[i]["state"]
        if prev == cur:
            continue
        counts[_TRANSITION_NAMES.get((prev, cur), "other")] += 1
    return counts


# -- section 4: shots ---------------------------------------------------------


def _per_shot_max_pulls(rows: list[dict]) -> list[float]:
    """For each fired row, the max pull_hw over the drawn run right before it."""
    max_pulls = []
    for i, row in enumerate(rows):
        if row.get("fired_power", "") == "":
            continue
        run = []
        j = i - 1
        while j >= 0 and rows[j]["state"] == "drawn":
            pull = _parse_float(rows[j], "pull_hw")
            if pull is not None:
                run.append(pull)
            j -= 1
        if run:
            max_pulls.append(max(run))
    return max_pulls


def _shots(rows: list[dict], max_pulls: list[float]) -> dict:
    fired = [_parse_float(r, "fired_power") for r in rows]
    fired = [v for v in fired if v is not None]

    return {
        "count": len(fired),
        "fired_power": _percentiles(fired, (10, 50, 90)),
        "fired_power_high_fraction": _fraction(fired, lambda v: v >= 0.95),
        "max_pull": _percentiles(max_pulls, (10, 50, 90)),
        "max_pull_full_draw_fraction": _fraction(
            max_pulls, lambda v: v >= config.DRAW_FULL_HW
        ),
    }


# -- section 5: bow hand grip -------------------------------------------------


def _bow_grip(rows: list[dict]) -> dict:
    values = []
    for r in rows:
        side = _s(r, "bow_side")
        if r["state"] in _BOW_HELD_STATES and side:
            v = _parse_float(r, f"{side}_fist")
            if v is not None:
                values.append(v)
    return {
        "sample_count": len(values),
        "fist": _percentiles(values, (5, 50, 95)),
        "fraction_above_fist_on": _fraction(values, lambda v: v > config.FIST_ON),
        "fraction_above_fist_off": _fraction(values, lambda v: v > config.FIST_OFF),
    }


# -- section 6: release responsiveness ---------------------------------------
#
# A "draw" is a maximal run of consecutive `drawn` rows. Its grip is read
# from the (optional) draw_grip column on the run's first row if present and
# non-empty, else inferred from the median of the draw hand's fist_ratio over
# the run's first few rows: a pinch grip only curls the index finger, so
# fist_ratio stays well above a true fist's even mid-pinch.


@dataclass(frozen=True)
class _Draw:
    start: int
    end: int                  # inclusive index of the run's last 'drawn' row
    side: str                 # draw_side from the run's first row
    grip: str                 # "pinch" / "fist" / "unknown"
    fired: bool
    fire_row: int | None
    release_rule: str         # "" if the column is absent or blank


def _draw_runs(rows: list[dict]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive 'drawn' rows, as (start, end) inclusive."""
    runs = []
    start = None
    for i, r in enumerate(rows):
        if r["state"] == "drawn":
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(rows) - 1))
    return runs


def _infer_grip(rows: list[dict], start: int, end: int, side: str) -> str:
    if not side:
        return "unknown"
    n = min(5, end - start + 1)
    samples = []
    for i in range(start, start + n):
        v = _parse_float(rows[i], f"{side}_fist")
        if v is not None:
            samples.append(v)
    if not samples:
        return "unknown"
    return "fist" if statistics.median(samples) < config.FIST_ON else "pinch"


def _classify_draws(rows: list[dict]) -> list[_Draw]:
    has_grip_col = _has_column(rows, "draw_grip")
    has_rule_col = _has_column(rows, "release_rule")

    draws = []
    for start, end in _draw_runs(rows):
        first = rows[start]
        side = _s(first, "draw_side")

        grip = _s(first, "draw_grip") if has_grip_col else ""
        if not grip:
            grip = _infer_grip(rows, start, end, side)

        rule = _s(first, "release_rule") if has_rule_col else ""

        fired = False
        fire_row = None
        nxt = end + 1
        if (
            nxt < len(rows)
            and rows[nxt]["state"] == "released"
            and _s(rows[nxt], "fired_power")
        ):
            fired = True
            fire_row = nxt

        draws.append(_Draw(start, end, side, grip, fired, fire_row, rule))
    return draws


def _own_signal_open(row: dict, side: str, grip: str) -> bool | None:
    """None when the needed ratio isn't present on this row."""
    if grip == "pinch":
        v = _parse_float(row, f"{side}_pinch")
        return None if v is None else v > config.PINCH_OFF
    if grip == "fist":
        v = _parse_float(row, f"{side}_fist")
        return None if v is None else v > config.FIST_OFF
    return None


def _other_signal_closed(row: dict, side: str, grip: str) -> bool | None:
    if grip == "pinch":
        v = _parse_float(row, f"{side}_fist")
        return None if v is None else v <= config.FIST_OFF
    if grip == "fist":
        v = _parse_float(row, f"{side}_pinch")
        return None if v is None else v <= config.PINCH_OFF
    return None


def _grip_row_refs(draws: list[_Draw], grip: str) -> list[tuple[int, str]]:
    """(row index, draw_side) for every row belonging to a draw of this grip."""
    refs = []
    for d in draws:
        if d.grip == grip:
            refs.extend((i, d.side) for i in range(d.start, d.end + 1))
    return refs


def _grip_metrics(
    rows: list[dict], draws: list[_Draw], grip: str, rule: str | None
) -> dict:
    """rule=None aggregates every release_rule; a rule string scopes to it."""
    selected = [d for d in draws if d.grip == grip and (rule is None or d.release_rule == rule)]
    fired = [d for d in selected if d.fired]
    unfired = [d for d in selected if not d.fired]

    # Latency: from the start of the final unbroken run of rows in which the
    # grip's own signal reads open, ending at the fire row, to the fire row.
    # Scanning backward from the fire rather than forward from the start of
    # the draw matters on real data: one noisy frame early in the pull would
    # otherwise be measured as a release the player made seconds ago.
    latencies = []
    for d in fired:
        if not _own_signal_open(rows[d.fire_row], d.side, grip):
            continue  # grip misclassified, or the ratio is missing on the fire row
        open_row = d.fire_row
        for i in range(d.end, d.start - 1, -1):
            if not _own_signal_open(rows[i], d.side, grip):
                break
            open_row = i
        latencies.append(int(rows[d.fire_row]["t_ms"]) - int(rows[open_row]["t_ms"]))

    # Blocked-open fraction and grip-ratio percentiles, over every drawn row
    # belonging to a selected draw.
    blocked = 0
    total_rows = 0
    pinch_vals, fist_vals = [], []
    for d in selected:
        for i in range(d.start, d.end + 1):
            row = rows[i]
            total_rows += 1
            pv = _parse_float(row, f"{d.side}_pinch")
            fv = _parse_float(row, f"{d.side}_fist")
            if pv is not None:
                pinch_vals.append(pv)
            if fv is not None:
                fist_vals.append(fv)
            if _own_signal_open(row, d.side, grip) and _other_signal_closed(row, d.side, grip):
                blocked += 1

    # Stuck releases: unfired draws whose own signal was open long enough
    # that debounce alone should have fired it under grip_aware.
    stuck = 0
    for d in unfired:
        streak = best = 0
        for i in range(d.start, d.end + 1):
            if _own_signal_open(rows[i], d.side, grip):
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        if best >= config.PINCH_OFF_FRAMES:
            stuck += 1

    return {
        "draws": len(selected),
        "fired": len(fired),
        "unfired": len(unfired),
        "latency_ms": _percentiles(latencies, (50, 90)),
        "blocked_open_fraction": (blocked / total_rows) if total_rows else None,
        "stuck_releases": stuck,
        "pinch": _percentiles(pinch_vals, (5, 50, 95)),
        "fist": _percentiles(fist_vals, (5, 50, 95)),
    }


def _release_responsiveness(rows: list[dict], draws: list[_Draw]) -> dict:
    has_grip_col = _has_column(rows, "draw_grip")
    has_rule_col = _has_column(rows, "release_rule")
    unknown = sum(1 for d in draws if d.grip == "unknown")

    by_grip = {g: _grip_metrics(rows, draws, g, rule=None) for g in ("pinch", "fist")}

    by_grip_rule: dict[str, dict[str, dict]] = {"pinch": {}, "fist": {}}
    if has_rule_col:
        for g in ("pinch", "fist"):
            rules = sorted({d.release_rule for d in draws if d.grip == g and d.release_rule})
            for rule in rules:
                by_grip_rule[g][rule] = _grip_metrics(rows, draws, g, rule=rule)

    return {
        "has_draw_grip_column": has_grip_col,
        "has_release_rule_column": has_rule_col,
        "unknown_grip_draws": unknown,
        "by_grip": by_grip,
        "by_grip_rule": by_grip_rule,
    }


# -- section 7: suggested thresholds -----------------------------------------


def _threshold_suggestion(
    closed: list[float], open_: list[float], on_frac: float, off_frac: float,
    min_sep: float, min_samples: int,
) -> dict:
    """Exactly calibration.py's formula, applied to whole-session samples."""
    result = {
        "closed_n": len(closed),
        "open_n": len(open_),
        "closed_median": statistics.median(closed) if closed else None,
        "open_median": statistics.median(open_) if open_ else None,
        "span": None,
        "on": None,
        "off": None,
        "ok": False,
        "reason": None,
    }

    if len(closed) < min_samples or len(open_) < min_samples:
        result["reason"] = (
            f"too few samples (closed n={len(closed)}, open n={len(open_)}, "
            f"need >= {min_samples} each)"
        )
        return result

    span = result["open_median"] - result["closed_median"]
    result["span"] = span
    if span < min_sep:
        result["reason"] = f"span {span:.3f} < CALIB_MIN_SEPARATION {min_sep}"
        return result

    result["on"] = result["open_median"] - on_frac * span
    result["off"] = result["open_median"] - off_frac * span
    result["ok"] = True
    return result


def _suggested_thresholds(
    rows: list[dict], draws: list[_Draw], max_pulls: list[float]
) -> dict:
    closed_fist, open_fist, open_pinch = [], [], []

    for r in rows:
        state = r["state"]
        bow_side = _s(r, "bow_side")
        draw_side = _s(r, "draw_side")

        if state in ("held", "drawn") and bow_side:
            v = _parse_float(r, f"{bow_side}_fist")
            if v is not None:
                closed_fist.append(v)

        if state == "released" and draw_side:
            fv = _parse_float(r, f"{draw_side}_fist")
            if fv is not None:
                open_fist.append(fv)
            pv = _parse_float(r, f"{draw_side}_pinch")
            if pv is not None:
                open_pinch.append(pv)

    # Pinch "closed" samples: draw-hand pinch over every row of draws already
    # classified "pinch" (grip classification, not a per-row fist>=FIST_ON test).
    closed_pinch = []
    for i, side in _grip_row_refs(draws, "pinch"):
        v = _parse_float(rows[i], f"{side}_pinch")
        if v is not None:
            closed_pinch.append(v)

    fist = _threshold_suggestion(
        closed_fist, open_fist,
        config.CALIB_ON_FRACTION, config.CALIB_OFF_FRACTION,
        config.CALIB_MIN_SEPARATION, config.CALIB_MIN_SAMPLES,
    )
    fist["current_on"] = config.FIST_ON
    fist["current_off"] = config.FIST_OFF

    pinch = _threshold_suggestion(
        closed_pinch, open_pinch,
        config.CALIB_ON_FRACTION, config.CALIB_OFF_FRACTION,
        config.CALIB_MIN_SEPARATION, config.CALIB_MIN_SAMPLES,
    )
    pinch["current_on"] = config.PINCH_ON
    pinch["current_off"] = config.PINCH_OFF

    low, high = config.CALIB_DRAW_CLAMP
    draw = {
        "shot_n": len(max_pulls),
        "current": config.DRAW_FULL_HW,
        "suggested": None,
        "ok": False,
        "reason": None,
    }
    if len(max_pulls) < 3:
        draw["reason"] = f"only {len(max_pulls)} shots (need >= 3)"
    else:
        draw["suggested"] = min(max(_pct(max_pulls, 50), low), high)
        draw["ok"] = True

    return {"fist": fist, "pinch": pinch, "draw_full_hw": draw}


# -- section 8: depth scale ---------------------------------------------------


def _depth_scale(rows: list[dict]) -> dict:
    values = []
    for r in rows:
        if r["state"] in ("held", "drawn"):
            v = _parse_float(r, "scale")
            if v is not None:
                values.append(v)
    return {"sample_count": len(values), "scale": _percentiles(values, (5, 50, 95))}


# -- section 9: hand loss during the draw -------------------------------------


def _hand_loss_during_draw(rows: list[dict]) -> dict:
    before_loss = []
    event_count = 0
    for i in range(1, len(rows)):
        if rows[i]["state"] != "drawn":
            continue
        for side in ("left", "right"):
            prev_seen = _seen(rows[i - 1], side)
            cur_seen = _seen(rows[i], side)
            if prev_seen and not cur_seen:
                event_count += 1
                sr = _parse_float(rows[i - 1], "size_ratio")
                if sr is not None:
                    before_loss.append(sr)

    all_drawn = []
    for r in rows:
        if r["state"] == "drawn":
            v = _parse_float(r, "size_ratio")
            if v is not None:
                all_drawn.append(v)

    ok = event_count >= 3
    return {
        "event_count": event_count,
        "ok": ok,
        "median_size_ratio_before_loss": (
            statistics.median(before_loss) if ok and before_loss else None
        ),
        "median_size_ratio_all_drawn": (
            statistics.median(all_drawn) if ok and all_drawn else None
        ),
    }


# -- top level ----------------------------------------------------------------


def analyze(rows: list[dict]) -> dict:
    """Pure: every number the report needs, no printing. Degrades gracefully --
    any statistic without enough samples comes back as None, never a crash.

    Re-validates rows itself (not just trusting load()), so a malformed row
    is skipped and counted correctly even if this is called directly on a
    list that didn't go through load().
    """
    skipped = getattr(rows, "skipped_rows", 0)
    clean = []
    for r in rows:
        if _is_valid_row(r):
            clean.append(r)
        else:
            skipped += 1
    rows = clean

    max_pulls = _per_shot_max_pulls(rows)
    draws = _classify_draws(rows)

    session = _session_stats(rows)
    session["skipped_rows"] = skipped

    return {
        "row_count": len(rows),
        "session": session,
        "state_seconds": _state_seconds(rows),
        "transitions": _transitions(rows),
        "shots": _shots(rows, max_pulls),
        "bow_grip": _bow_grip(rows),
        "release_responsiveness": _release_responsiveness(rows, draws),
        "suggested_thresholds": _suggested_thresholds(rows, draws, max_pulls),
        "depth_scale": _depth_scale(rows),
        "hand_loss_during_draw": _hand_loss_during_draw(rows),
    }


# -- report formatting ---------------------------------------------------------

_INSUFFICIENT = "insufficient data"


def _fmt(value, digits=3) -> str:
    return _INSUFFICIENT if value is None else f"{value:.{digits}f}"


def _fmt_pcts(pcts: dict | None, digits=3) -> str:
    if pcts is None:
        return _INSUFFICIENT
    return "  ".join(f"{k}={v:.{digits}f}" for k, v in pcts.items())


def _fmt_pct_val(value, digits=1) -> str:
    return _INSUFFICIENT if value is None else f"{value * 100:.{digits}f}%"


def _section(title: str, entries: list[tuple[str, str]], notes: list[str] = ()) -> str:
    lines = [title, "-" * len(title)]
    width = max((len(label) for label, _ in entries), default=0)
    for label, value in entries:
        lines.append(f"  {label.ljust(width)} : {value}")
    for note in notes:
        lines.append(f"  ({note})")
    lines.append("")
    return "\n".join(lines)


def _grip_entries(label: str, m: dict) -> list[tuple[str, str]]:
    return [
        (f"{label}: draws / fired / unfired", f"{m['draws']} / {m['fired']} / {m['unfired']}"),
        (f"{label}: latency_ms, own signal->fire", _fmt_pcts(m["latency_ms"], 0)),
        (f"{label}: blocked-open fraction", _fmt_pct_val(m["blocked_open_fraction"])),
        (f"{label}: stuck releases (of unfired)", str(m["stuck_releases"])),
        (f"{label}: pinch_ratio", _fmt_pcts(m["pinch"])),
        (f"{label}: fist_ratio", _fmt_pcts(m["fist"])),
    ]


def format_report(result: dict) -> str:
    if result["row_count"] == 0:
        skipped = result["session"].get("skipped_rows", 0)
        extra = f" ({skipped} malformed row(s) skipped)" if skipped else ""
        return f"{_INSUFFICIENT}: CSV has no data rows{extra}"

    out = []

    s = result["session"]
    fps_h = s["fps_by_hands"]
    out.append(_section("1. Session", [
        ("rows", str(s["row_count"])),
        ("skipped (malformed) rows", str(s.get("skipped_rows", 0))),
        ("duration_s", _fmt(s["duration_s"], 1)),
        ("tracked_fps", _fmt(s["tracked_fps"], 1)),
        ("gaps > 100ms", str(s["gap_count"])),
        ("max_gap_ms", _fmt(s["max_gap_ms"], 0)),
        ("fps @ 0 hands", _fmt(fps_h.get(0), 1)),
        ("fps @ 1 hand", _fmt(fps_h.get(1), 1)),
        ("fps @ 2 hands", _fmt(fps_h.get(2), 1)),
    ], notes=[
        "two-hand detection costs ~29ms vs the camera's 33ms frame interval -- "
        "watch whether fps @ 2 hands falls noticeably below the others",
    ]))

    ss = result["state_seconds"]
    out.append(_section("2. States (seconds)", [
        ("docked", _fmt(ss["docked"], 1)),
        ("held", _fmt(ss["held"], 1)),
        ("drawn", _fmt(ss["drawn"], 1)),
        ("released", _fmt(ss["released"], 1)),
    ]))

    trans = result["transitions"]
    out.append(_section("3. Transitions", [
        ("grabs (docked->held)", str(trans["grabs"])),
        ("nocks (held->drawn)", str(trans["nocks"])),
        ("fires (drawn->released)", str(trans["fires"])),
        ("cancels (drawn->held)", str(trans["cancels"])),
        ("cooldowns (released->held)", str(trans["cooldowns"])),
        ("drops while held (held->docked)", str(trans["drops_while_held"])),
        ("drops while drawn (drawn->docked)", str(trans["drops_while_drawn"])),
        ("docked after release (released->docked)", str(trans["docked_after_release"])),
        ("other", str(trans["other"])),
    ], notes=[
        "drops while drawn is the most important feel signal -- the player "
        "lost the bow mid-shot",
    ]))

    sh = result["shots"]
    out.append(_section("4. Shots", [
        ("count", str(sh["count"])),
        ("fired_power", _fmt_pcts(sh["fired_power"])),
        ("fired_power >= 0.95", _fmt_pct_val(sh["fired_power_high_fraction"])),
        ("per-shot max pull_hw", _fmt_pcts(sh["max_pull"])),
        (f"max pull reached DRAW_FULL_HW ({config.DRAW_FULL_HW})",
         _fmt_pct_val(sh["max_pull_full_draw_fraction"])),
    ]))

    bg = result["bow_grip"]
    out.append(_section("5. Bow hand grip (held/drawn/released)", [
        ("samples", str(bg["sample_count"])),
        ("fist_ratio", _fmt_pcts(bg["fist"])),
        (f"fraction > FIST_ON ({config.FIST_ON})", _fmt_pct_val(bg["fraction_above_fist_on"])),
        (f"fraction > FIST_OFF ({config.FIST_OFF})", _fmt_pct_val(bg["fraction_above_fist_off"])),
    ], notes=[
        "sustained time above FIST_OFF drops the bow",
    ]))

    rr = result["release_responsiveness"]
    entries = []
    for grip in ("pinch", "fist"):
        entries += _grip_entries(grip, rr["by_grip"][grip])
    for grip in ("pinch", "fist"):
        for rule, m in rr["by_grip_rule"].get(grip, {}).items():
            entries += _grip_entries(f"{grip} / {rule}", m)
    if rr["unknown_grip_draws"]:
        entries.append(("unknown-grip draws (excluded above)", str(rr["unknown_grip_draws"])))

    out.append(_section("6. Release responsiveness", entries, notes=[
        "both_open fires when pinch_ratio > PINCH_OFF AND fist_ratio > FIST_OFF; "
        "grip_aware fires a pinch grip on pinch_ratio > PINCH_OFF alone, a fist "
        "grip on fist_ratio > FIST_OFF alone",
        f"latency runs from the grip's own signal opening to the fire row; "
        f"debounce alone accounts for about PINCH_OFF_FRAMES "
        f"({config.PINCH_OFF_FRAMES}) frame intervals (~33-66 ms), so latency "
        "well above ~70 ms means the release rule is holding a release the "
        "player already made",
        "blocked-open = the grip's own signal reads open while the other "
        "ratio still reads closed -- e.g. a relaxed pinch whose other fingers "
        "stay curled keeps fist_ratio near 1.3 (below FIST_OFF), so both_open "
        "never fires even though the pinch opened",
    ]))

    st = result["suggested_thresholds"]
    fist_t, pinch_t, draw_t = st["fist"], st["pinch"], st["draw_full_hw"]

    def _threshold_entries(name, t):
        entries = [
            (f"{name} closed/open n", f"{t['closed_n']} / {t['open_n']}"),
            (f"{name} closed/open median", f"{_fmt(t['closed_median'])} / {_fmt(t['open_median'])}"),
            (f"{name} span (min {config.CALIB_MIN_SEPARATION})", _fmt(t["span"])),
        ]
        if t["ok"]:
            entries.append((f"{name} suggested on/off", f"{t['on']:.3f} / {t['off']:.3f}"))
        else:
            entries.append((f"{name} suggested on/off", f"{_INSUFFICIENT} ({t['reason']})"))
        entries.append((f"{name} current on/off", f"{t['current_on']:.3f} / {t['current_off']:.3f}"))
        return entries

    threshold_entries = (
        _threshold_entries("fist", fist_t) + _threshold_entries("pinch", pinch_t)
    )
    threshold_entries.append(("draw shots used", str(draw_t["shot_n"])))
    if draw_t["ok"]:
        threshold_entries.append(("draw suggested DRAW_FULL_HW", f"{draw_t['suggested']:.3f}"))
    else:
        threshold_entries.append(
            ("draw suggested DRAW_FULL_HW", f"{_INSUFFICIENT} ({draw_t['reason']})")
        )
    threshold_entries.append(("draw current DRAW_FULL_HW", f"{draw_t['current']:.3f}"))

    out.append(_section(
        "7. Suggested thresholds (calibration.py formula, whole-session samples)",
        threshold_entries,
    ))

    ds = result["depth_scale"]
    out.append(_section("8. Depth scale (held/drawn)", [
        ("samples", str(ds["sample_count"])),
        ("scale", _fmt_pcts(ds["scale"])),
    ]))

    hl = result["hand_loss_during_draw"]
    if hl["ok"]:
        loss_entries = [
            ("events", str(hl["event_count"])),
            ("median size_ratio just before loss", _fmt(hl["median_size_ratio_before_loss"])),
            ("median size_ratio, all drawn rows", _fmt(hl["median_size_ratio_all_drawn"])),
        ]
    else:
        loss_entries = [
            ("events", str(hl["event_count"])),
            ("comparison", f"{_INSUFFICIENT} (fewer than 3 events)"),
        ]
    out.append(_section(
        "9. Hand loss during the draw (occlusion-as-intent groundwork)",
        loss_entries,
    ))

    return "\n".join(out).rstrip() + "\n"


# -- CLI ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m fletchflow.telemetry_report <playtest.csv>", file=sys.stderr)
        return 2

    try:
        rows = load(argv[0])
    except ValueError as exc:
        print(f"telemetry_report: {exc}", file=sys.stderr)
        return 2

    print(format_report(analyze(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
