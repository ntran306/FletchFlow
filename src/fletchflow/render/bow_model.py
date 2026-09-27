"""The bow's geometry in model space: load an asset, or build one procedurally.

numpy only, no GL, so it is fully unit-testable and both renderers — the
moderngl body and the 2D fallback — consume the same vertices and cannot
disagree about where the tips are (PLAN.md §4.7.8).

**Model space** is metres, +Y = `bow_up`, +Z = `bow_forward`, +X = `bow_right`,
origin at the grip, with the limbs spanning `L = LIMB_SPAN_M` tip to tip. A
downloaded asset is authored in whatever space its artist felt like, so
`fit_asset` rotates, recentres and rescales it into that convention rather than
asking the player to get the export exactly right (§4.7.8a).

**Part identification.** The renderer needs the bow and the arrow separately,
because the arrow is positioned from the nock every frame. Assets rarely name
their objects helpfully — the one supplied for this milestone calls them
`Cube_Cube.001` and `Cube.001_Cube.002` — so parts are identified by object
name, then by material name, then by shape, and the rule that fired is
recorded on the result so a mis-identification is visible rather than puzzling.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from fletchflow import config
from fletchflow.input.bow_input import BowState

# Tip-to-tip bow length in model space. BOW_SPAN_PX is the on-screen span we
# want at render_scale 1.0, and BOW_RENDER_DEPTH_M is where the bow is placed,
# so this is just that pixel span unprojected through the game's own pinhole.
LIMB_SPAN_M = config.BOW_SPAN_PX * config.BOW_RENDER_DEPTH_M / config.FOCAL_PX

_BOW_WORDS = ("bow", "riser", "limb", "handle")
_ARROW_WORDS = ("arrow", "shaft", "bolt", "dart")
_STRING_WORDS = ("string",)


@dataclass(frozen=True)
class Mesh:
    """One part, triangulated, in whatever space it was loaded or fitted to."""

    name: str
    positions: np.ndarray          # (N, 3) float32
    normals: np.ndarray            # (N, 3) float32
    indices: np.ndarray            # (M, 3) uint32
    uvs: np.ndarray | None = None  # (N, 2) float32
    material: str = ""

    @property
    def extent(self) -> np.ndarray:
        """(3,) span along each axis."""
        if len(self.positions) == 0:
            return np.zeros(3, dtype=np.float64)
        return self.positions.max(axis=0) - self.positions.min(axis=0)

    @property
    def centre(self) -> np.ndarray:
        if len(self.positions) == 0:
            return np.zeros(3, dtype=np.float64)
        return (self.positions.max(axis=0) + self.positions.min(axis=0)) / 2.0


@dataclass(frozen=True)
class BowAsset:
    bow: Mesh
    arrow: Mesh | None
    source: str = ""
    how: str = ""       # which identification rule fired, for the startup log
    fitted: bool = False


class AssetError(Exception):
    """The file is unreadable, or does not contain a usable bow."""


# -- Wavefront OBJ ------------------------------------------------------------


def parse_obj(text: str) -> list[Mesh]:
    """Every `o`/`g` group in an OBJ, triangulated, as its own Mesh.

    Written out rather than pulled from a library because OBJ is plain text and
    this is the whole of it: shared 1-based vertex pools, optional `vt`/`vn`
    references, n-gons, and negative (relative) indices. A dependency would buy
    nothing and cost a wheel on every install.

    OBJ indexes position, uv and normal *independently*, while GPUs want one
    index per vertex, so each distinct `v/vt/vn` triple becomes one vertex here.
    """
    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []

    meshes: list[Mesh] = []
    name = ""
    material = ""
    # per-group vertex accumulation, keyed by the OBJ triple so it is shared
    combined: dict[tuple[int, int, int], int] = {}
    out_pos: list[tuple[float, float, float]] = []
    out_uv: list[tuple[float, float]] = []
    out_nrm: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    any_uv = False
    any_nrm = False

    def flush() -> None:
        nonlocal combined, out_pos, out_uv, out_nrm, tris, any_uv, any_nrm
        if tris:
            meshes.append(
                _make_mesh(name or f"part{len(meshes)}", out_pos, out_nrm,
                           out_uv if any_uv else None, tris,
                           has_normals=any_nrm, material=material)
            )
        combined, out_pos, out_uv, out_nrm, tris = {}, [], [], [], []
        any_uv = any_nrm = False

    def resolve(idx: int, pool: list) -> int:
        """OBJ indices are 1-based; negative means relative to the end."""
        return idx - 1 if idx > 0 else len(pool) + idx

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        kind, _, rest = line.partition(" ")
        rest = rest.strip()

        if kind == "v":
            parts = rest.split()
            positions.append((float(parts[0]), float(parts[1]), float(parts[2])))
        elif kind == "vt":
            parts = rest.split()
            uvs.append((float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0))
        elif kind == "vn":
            parts = rest.split()
            normals.append((float(parts[0]), float(parts[1]), float(parts[2])))
        elif kind in ("o", "g"):
            flush()
            name = rest
        elif kind == "usemtl":
            material = rest
        elif kind == "f":
            corners = []
            for token in rest.split():
                bits = (token.split("/") + ["", ""])[:3]
                pi = resolve(int(bits[0]), positions)
                ti = resolve(int(bits[1]), uvs) if bits[1] else -1
                ni = resolve(int(bits[2]), normals) if bits[2] else -1
                key = (pi, ti, ni)
                if key not in combined:
                    combined[key] = len(out_pos)
                    out_pos.append(positions[pi])
                    out_uv.append(uvs[ti] if 0 <= ti < len(uvs) else (0.0, 0.0))
                    out_nrm.append(
                        normals[ni] if 0 <= ni < len(normals) else (0.0, 0.0, 0.0)
                    )
                    if ti >= 0:
                        any_uv = True
                    if ni >= 0:
                        any_nrm = True
                corners.append(combined[key])
            # Fan-triangulate. Every face in the supplied asset is a quad, and
            # a fan is correct for any convex polygon, which exported faces are.
            for k in range(1, len(corners) - 1):
                tris.append((corners[0], corners[k], corners[k + 1]))

    flush()
    return meshes


def _make_mesh(name, pos, nrm, uv, tris, *, has_normals: bool, material: str) -> Mesh:
    positions = np.asarray(pos, dtype=np.float32)
    indices = np.asarray(tris, dtype=np.uint32)
    normals = np.asarray(nrm, dtype=np.float32)
    if not has_normals or not np.isfinite(normals).all() or len(normals) != len(positions):
        normals = compute_normals(positions, indices)
    else:
        zero = np.linalg.norm(normals, axis=1) < 1e-8
        if zero.any():  # a partial `vn` set: fill the gaps from the faces
            normals = np.where(zero[:, None], compute_normals(positions, indices), normals)
    return Mesh(
        name=name,
        positions=positions,
        normals=_normalize_rows(normals),
        indices=indices,
        uvs=np.asarray(uv, dtype=np.float32) if uv is not None else None,
        material=material,
    )


def compute_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals, for an asset that ships without them.

    The cross product of two edges is already proportional to twice the
    triangle's area, so accumulating it unnormalized weights each face by its
    size — which is what keeps a large flat limb from being dragged around by
    a cluster of tiny triangles at the tip.
    """
    normals = np.zeros_like(positions, dtype=np.float32)
    if len(indices) == 0:
        return normals
    a, b, c = (positions[indices[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(normals, indices[:, i], face)
    return _normalize_rows(normals)


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9).astype(np.float32)


# -- part identification ------------------------------------------------------


def _match(text: str, words) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{w}|{w}\b", low) for w in words)


def identify(meshes: list[Mesh]) -> tuple[Mesh, Mesh | None, str]:
    """Pick the bow and the arrow out of an asset's parts.

    Three rules, most trustworthy first, because downloaded assets are named
    after whatever primitive they started life as:

    1. **object name** — an export that says `Bow` / `Arrow` means it;
    2. **material name** — artists name materials after the thing far more
       reliably than they rename meshes. The supplied asset is exactly this
       case: objects `Cube_Cube.001` / `Cube.001_Cube.002`, materials
       `dirty_arrow` / `bow`;
    3. **shape** — of two parts, the bow is the one whose longest axis is
       longest; an arrow is a thin stick beside it.

    A string part is discarded: the string has to bend to the draw hand every
    frame, so it is drawn procedurally and any modelled one would fight it.
    """
    usable = [m for m in meshes if len(m.indices) and not _match(m.name, _STRING_WORDS)
              and not _match(m.material, _STRING_WORDS)]
    if not usable:
        raise AssetError("no drawable geometry in the asset")

    for rule, field in (("object name", "name"), ("material name", "material")):
        bows = [m for m in usable if _match(getattr(m, field), _BOW_WORDS)]
        arrows = [m for m in usable if _match(getattr(m, field), _ARROW_WORDS)]
        # "dirty_arrow" must not also count as a bow; require a clean split
        bows = [m for m in bows if m not in arrows]
        if len(bows) == 1 and len(arrows) <= 1:
            return bows[0], (arrows[0] if arrows else None), f"by {rule}"

    if len(usable) == 1:
        return usable[0], None, "single part"

    ordered = sorted(usable, key=lambda m: float(m.extent.max()), reverse=True)
    return ordered[0], ordered[1], "by shape (longest part is the bow)"


# -- stripping a modelled string ----------------------------------------------


def split_components(mesh: Mesh) -> list[np.ndarray]:
    """The mesh's connected components, as arrays of triangles.

    Welds by *position* first. An OBJ splits one corner into several vertices
    wherever a uv or normal seam runs through it, so index-only connectivity
    reports a single solid part as a handful of shells.
    """
    n = len(mesh.positions)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    seen: dict[tuple, int] = {}
    for i, key in enumerate(map(tuple, np.round(mesh.positions, 5))):
        if key in seen:
            union(i, seen[key])
        else:
            seen[key] = i
    for t in mesh.indices:
        union(int(t[0]), int(t[1]))
        union(int(t[1]), int(t[2]))

    groups: dict[int, list] = {}
    for t in mesh.indices:
        groups.setdefault(find(int(t[0])), []).append(t)
    return [np.asarray(v, dtype=np.uint32) for v in groups.values()]


def strip_string(
    mesh: Mesh, limb_axis: int, *, min_length: float = 0.7, max_cross: float = 0.05
) -> tuple[Mesh, int]:
    """Remove a modelled bowstring, returning (mesh, triangles removed).

    Almost every bow asset ships with its string modelled in, and ours cannot
    use one: the string has to run from the limb tips to wherever the draw
    hand actually is, changing shape every frame, so a rigid tip-to-tip
    cylinder would sit straight through the drawn one.

    A string is the connected component that runs nearly the whole length of
    the bow while being negligibly thin across it. On the supplied asset the
    three components measure, as a fraction of the limb span, length/cross =
    1.00/0.355 (body), 0.98/0.021 (string) and 0.09/0.049 (grip wrap) — so the
    thresholds have an order of magnitude of room on both sides.
    """
    comps = split_components(mesh)
    if len(comps) < 2:
        return mesh, 0
    span = float(mesh.extent[limb_axis])
    if span <= 1e-9:
        return mesh, 0
    cross_axes = [a for a in range(3) if a != limb_axis]

    keep = []
    removed = 0
    for tris in comps:
        pts = mesh.positions[np.unique(tris.ravel())]
        ext = pts.max(axis=0) - pts.min(axis=0)
        is_string = (
            ext[limb_axis] / span >= min_length
            and max(ext[a] for a in cross_axes) / span <= max_cross
        )
        if is_string:
            removed += len(tris)
        else:
            keep.append(tris)

    if not keep or removed == 0:
        return mesh, 0
    return _compact(mesh, np.concatenate(keep)), removed


def _compact(mesh: Mesh, tris: np.ndarray) -> Mesh:
    """Rebuild a mesh around a subset of its triangles, dropping loose vertices."""
    used = np.unique(tris.ravel())
    remap = np.full(len(mesh.positions), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return replace(
        mesh,
        positions=mesh.positions[used],
        normals=mesh.normals[used],
        uvs=mesh.uvs[used] if mesh.uvs is not None else None,
        indices=remap[tris].astype(np.uint32),
    )


# -- fitting into model space -------------------------------------------------


def fit_asset(asset: BowAsset, limb_span_m: float = LIMB_SPAN_M) -> BowAsset:
    """Rotate, recentre and rescale an asset into model space.

    Axes are derived from the geometry rather than trusted from the export,
    since "+Y up" means different things to different tools and the supplied
    asset is authored with its limbs along Z and its arrow along X:

    * **limb axis → +Y** — the bow's longest extent. A bow is longer tip to
      tip than it is deep or wide, by a lot.
    * **forward → +Z** — the arrow's longest extent when there is an arrow,
      since an arrow points where it is shot. Without one, the bow's *second*
      longest extent, which is the riser's depth.
    * **right → +X** — completed as a cross product so the frame is a proper
      rotation (det +1) and the model is not mirrored.

    The bow is then recentred on its limb-axis midpoint — the grip, which is
    what the hand holds and what the bow must rotate about — and scaled so its
    limbs span `limb_span_m`. The arrow is recentred on itself and scaled by
    the *same* factor, which keeps the artist's bow-to-arrow proportion rather
    than imposing one.
    """
    bow, arrow = asset.bow, asset.arrow
    limb = _dominant_axis(bow.extent)
    if arrow is not None:
        forward = _dominant_axis(arrow.extent, exclude=limb)
    else:
        forward = _dominant_axis(bow.extent, exclude=limb)

    up_v = _unit(limb)
    fwd_v = _unit(forward)
    if arrow is not None and _arrow_points_negative(arrow, forward):
        fwd_v = -fwd_v
    right_v = np.cross(up_v, fwd_v)
    # Columns (right, up, forward); transposing gives world->model, which is
    # what we apply to the authored vertices.
    rot = np.stack([right_v, up_v, fwd_v], axis=0).astype(np.float32)
    if np.linalg.det(rot) < 0:
        rot[0] = -rot[0]

    span = float(bow.extent[limb])
    scale = limb_span_m / span if span > 1e-9 else 1.0
    grip = bow.centre

    def place(m: Mesh, origin: np.ndarray) -> Mesh:
        pos = ((m.positions - origin) @ rot.T) * scale
        return replace(
            m,
            positions=pos.astype(np.float32),
            normals=_normalize_rows(m.normals @ rot.T),
        )

    return replace(
        asset,
        bow=place(bow, grip),
        arrow=place(arrow, arrow.centre) if arrow is not None else None,
        fitted=True,
    )


def _dominant_axis(extent: np.ndarray, exclude: int | None = None) -> int:
    order = np.argsort(extent)[::-1]
    for axis in order:
        if exclude is None or int(axis) != exclude:
            return int(axis)
    return 0


def _unit(axis: int) -> np.ndarray:
    v = np.zeros(3, dtype=np.float32)
    v[axis] = 1.0
    return v


def _arrow_points_negative(arrow: Mesh, axis: int) -> bool:
    """True when the arrow's head lies at the low end of `axis`.

    The head is the thin end: three fletches make the nock end measurably
    fatter in cross-section. Compared over the outer fifth at each end, so the
    shaft's own taper does not decide it.
    """
    along = arrow.positions[:, axis]
    lo, hi = float(along.min()), float(along.max())
    cut = (hi - lo) * 0.2
    if cut <= 1e-9:
        return False
    others = [a for a in range(3) if a != axis]
    def girth(mask):
        pts = arrow.positions[mask][:, others]
        return float(np.ptp(pts, axis=0).sum()) if mask.any() else 0.0
    return girth(along <= lo + cut) < girth(along >= hi - cut)


# -- loading ------------------------------------------------------------------


def load_asset(path: str | Path, *, fit: bool = True) -> BowAsset:
    """Load `.obj` (and, once a `.glb` turns up, glTF) into model space.

    Raises AssetError for anything the caller should fall back from — a
    missing file, an unreadable one, or geometry with no usable bow in it.
    """
    p = Path(path)
    if not p.exists():
        raise AssetError(f"no such asset: {p}")
    suffix = p.suffix.lower()
    if suffix != ".obj":
        raise AssetError(f"unsupported asset format {suffix!r} (expected .obj)")
    try:
        meshes = parse_obj(p.read_text(encoding="utf-8", errors="replace"))
    except AssetError:
        raise
    except Exception as exc:  # a truncated or malformed file
        raise AssetError(f"could not parse {p.name}: {exc}") from exc

    bow, arrow, how = identify(meshes)
    # Before fitting, so the grip is centred on the bow body rather than being
    # pulled off by a string sitting at the back of the bounding box.
    bow, cut = strip_string(bow, _dominant_axis(bow.extent))
    if cut:
        how = f"{how}, string stripped ({cut} tri)"
    asset = BowAsset(bow=bow, arrow=arrow, source=str(p), how=how)
    return fit_asset(asset) if fit else asset


def describe(asset: BowAsset) -> str:
    """One line for the startup log — enough to spot a mis-identified part."""
    def part(m: Mesh | None, label: str) -> str:
        if m is None:
            return f"{label}=none"
        e = m.extent
        return (f"{label}={m.name!r}[{m.material or '-'}] "
                f"{len(m.indices)}tri {e[0]:.2f}x{e[1]:.2f}x{e[2]:.2f}m")
    return (f"{Path(asset.source).name}: {part(asset.bow, 'bow')}, "
            f"{part(asset.arrow, 'arrow')} ({asset.how}"
            f"{', fitted' if asset.fitted else ''})")


# -- placement and projection -------------------------------------------------


def to_world(
    mesh: Mesh,
    centre_m: tuple[float, float, float],
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    forward: tuple[float, float, float],
) -> np.ndarray:
    """Model-space vertices to world metres: `centre + R . model`.

    R's columns are (right, up, forward), which is a proper rotation as long
    as the caller passes an orthonormal right-handed triple — `BowPose` builds
    exactly that from the hands (§4.7.3).
    """
    rot = np.stack(
        [np.asarray(right, dtype=np.float32),
         np.asarray(up, dtype=np.float32),
         np.asarray(forward, dtype=np.float32)],
        axis=1,
    )
    return (mesh.positions @ rot.T) + np.asarray(centre_m, dtype=np.float32)


def project(
    points_m: np.ndarray,
    focal_px: float = config.FOCAL_PX,
    centre_px: tuple[float, float] | None = None,
) -> np.ndarray:
    """World metres to pixels through the game's own pinhole.

    The same camera the targets and arrows use, so the bow shares their
    perspective instead of being pasted on top in its own projection. Points
    at or behind the eye have no projection; their depth is clamped to
    NEAR_PLANE_M rather than dropped, so a vertex that swings briefly behind
    the camera stretches instead of making the triangle vanish.
    """
    if centre_px is None:
        centre_px = (config.WINDOW_SIZE[0] / 2.0, config.WINDOW_SIZE[1] / 2.0)
    z = np.maximum(points_m[:, 2], config.NEAR_PLANE_M)
    return np.stack(
        [centre_px[0] + focal_px * points_m[:, 0] / z,
         centre_px[1] + focal_px * points_m[:, 1] / z],
        axis=1,
    )


def _preview(argv: list[str] | None = None) -> int:
    """Render a fitted asset to a PNG, so skins can be compared without
    starting the game: `python -m fletchflow.render.bow_model <file.obj>`.
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Preview a bow asset as a PNG.")
    ap.add_argument("asset")
    ap.add_argument("--out", default="bow_preview.png")
    ap.add_argument("--yaw", type=float, default=25.0, help="degrees, 0 = face on")
    args = ap.parse_args(argv)

    asset = load_asset(args.asset)
    print(describe(asset))

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    w, h = config.WINDOW_SIZE
    surface = pygame.Surface((w, h))
    surface.fill((24, 26, 32))

    a = math.radians(args.yaw)
    right = (math.cos(a), 0.0, -math.sin(a))
    up = (0.0, -1.0, 0.0)          # +y is DOWN in world axes, so bow-up is -y
    forward = (math.sin(a), 0.0, math.cos(a))
    centre = (0.0, 0.0, config.BOW_RENDER_DEPTH_M)

    light = np.array([-0.4, -0.7, -0.6], dtype=np.float32)
    light /= np.linalg.norm(light)

    for mesh, tint in ((asset.bow, (168, 132, 86)), (asset.arrow, (200, 196, 188))):
        if mesh is None:
            continue
        world = to_world(mesh, centre, right, up, forward)
        screen = project(world)
        rot = np.stack([np.asarray(right), np.asarray(up), np.asarray(forward)], axis=1)
        nrm = mesh.normals @ rot.T
        tri = mesh.indices
        depth = world[tri, 2].mean(axis=1)
        shade = np.clip((nrm[tri].mean(axis=1) @ light), 0.0, 1.0) * 0.75 + 0.25
        for k in np.argsort(-depth):          # painter's algorithm, far first
            pts = [(float(screen[i][0]), float(screen[i][1])) for i in tri[k]]
            c = tuple(int(v * shade[k]) for v in tint)
            pygame.draw.polygon(surface, c, pts)

    pygame.image.save(surface, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_preview())


# -- placing a fitted asset from a BowPose ------------------------------------

# String geometry in model space, as fractions of the limb span (PLAN.md
# §4.7.8). The nock sits just above centre and pulls back along -Z (toward the
# archer) as power builds; +Z is the direction the arrow flies.
NOCK_RISE = 0.02
NOCK_REST = 0.18
NOCK_PULL = 0.55


@dataclass(frozen=True)
class PlacedPart:
    """One part, projected: screen triangles plus what the painter needs."""

    tris_px: np.ndarray     # (M, 3, 2) float32, screen pixels
    depth_m: np.ndarray     # (M,) float32, mean world depth — painter sort key
    normals: np.ndarray     # (M, 3) float32, world-space face normals
    centroids_m: np.ndarray # (M, 3) float32, world face centres — for back-face culling


@dataclass(frozen=True)
class PlacedBow:
    bow: PlacedPart
    arrow: PlacedPart | None
    tips_px: tuple[tuple[float, float], tuple[float, float]]
    nock_px: tuple[float, float]
    centre_m: tuple[float, float, float]


def tip_anchors(bow: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Where the string attaches, in model space: the mean of the vertices in
    the outermost 5% of each limb.

    Taken from the geometry rather than assumed at (0, +-L/2, 0), because a
    recurve's tips sit well behind the grip in Z and a string drawn to the
    wrong Z crosses through the limbs at any yaw but dead-on.
    """
    y = bow.positions[:, 1]
    lo, hi = float(y.min()), float(y.max())
    cut = (hi - lo) * 0.05
    top = bow.positions[y >= hi - cut]
    bottom = bow.positions[y <= lo + cut]
    return top.mean(axis=0), bottom.mean(axis=0)


def nock_point(power: float, limb_span_m: float = LIMB_SPAN_M) -> np.ndarray:
    """Model-space nock for a given draw power (§4.7.8)."""
    return np.array(
        [0.0,
         NOCK_RISE * limb_span_m,
         -(NOCK_REST + float(power) * NOCK_PULL) * limb_span_m],
        dtype=np.float32,
    )


def pose_basis(
    up: tuple[float, float, float], forward: tuple[float, float, float]
) -> np.ndarray:
    """(3, 3) rotation whose columns are (right, up, forward), det +1.

    `right` is `up x forward`, not `forward x up`. World +y points DOWN while
    model +Y is the bow's up, so the other order produces a determinant of -1
    — a mirrored bow, which on a symmetric mesh looks fine right up until the
    arrow rest appears on the wrong side.

    Forward is preserved exactly and `up` is squared up against it, never the
    other way round. Forward is the aim — the crosshair is computed from the
    same angles (§4.7.4), and acceptance 18 pins the drawn arrow's vanishing
    point to that crosshair — while up is only the bow's roll about it. Nudging
    forward to suit a slightly off-square up would point the arrow somewhere
    the player is not aiming.
    """
    u = np.asarray(up, dtype=np.float32)
    f = np.asarray(forward, dtype=np.float32)
    f = f / max(float(np.linalg.norm(f)), 1e-9)
    u = u - f * float(np.dot(u, f))
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        # up parallel to forward: no roll is recoverable, so pick any square
        # axis rather than dividing by zero and painting NaNs.
        fallback = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, f))) > 0.9:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        u = fallback - f * float(np.dot(fallback, f))
        n = float(np.linalg.norm(u))
    u = u / n
    r = np.cross(u, f)
    return np.stack([r, u, f], axis=1).astype(np.float32)


def unproject(
    px: tuple[float, float], depth_m: float,
    focal_px: float = config.FOCAL_PX,
    centre_px: tuple[float, float] | None = None,
) -> np.ndarray:
    """Screen pixels at a known depth back to world metres."""
    if centre_px is None:
        centre_px = (config.WINDOW_SIZE[0] / 2.0, config.WINDOW_SIZE[1] / 2.0)
    return np.array(
        [(px[0] - centre_px[0]) * depth_m / focal_px,
         (px[1] - centre_px[1]) * depth_m / focal_px,
         depth_m],
        dtype=np.float32,
    )


def _yaw(radians: float) -> np.ndarray:
    """Rotation about model +Y, the bow's own up axis."""
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _pair(v) -> tuple[float, float]:
    return (float(v[0]), float(v[1]))


def _place_part(mesh: Mesh, rot: np.ndarray, centre: np.ndarray) -> PlacedPart:
    world = (mesh.positions @ rot.T) + centre
    screen = project(world)
    tri = mesh.indices
    a, b, c = (world[tri[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    return PlacedPart(
        tris_px=screen[tri].astype(np.float32),
        depth_m=world[tri, 2].mean(axis=1).astype(np.float32),
        normals=_normalize_rows(face),
        centroids_m=((a + b + c) / 3.0).astype(np.float32),
    )


def front_facing(part: PlacedPart) -> np.ndarray:
    """(M,) bool: triangles whose outward normal turns toward the eye.

    The eye is the world origin, so a face at centroid c with outward normal n
    is visible when dot(n, c) < 0. On a closed mesh this is exactly the half
    the far side hides, and dropping it halves the painter's per-triangle
    Python loop — the one thing in this path that is not vectorized.

    Open sheets, like the procedural fletches, are built with both windings
    for this reason: one of each pair always survives the cull.
    """
    return np.einsum("ij,ij->i", part.normals, part.centroids_m) < 0.0


def place_bow(pose, asset: BowAsset) -> PlacedBow:
    """Put a fitted asset in the world for this frame and project it.

    The bow is hung at the bow hand's own depth: `render_scale` is
    REFERENCE_BOW_DEPTH_M over that depth (§4.7.3), so dividing the render
    depth by it puts the bow nearer when the player leans in and further when
    they lean out — which is what makes it grow and shrink the perspective way
    rather than by scaling a sprite.

    The arrow is only placed while DRAWN, riding the nock: there is no arrow
    on the string before the draw, and the same rule already governs the
    crosshair (§4.7.4).
    """
    rot = pose_basis(pose.bow_up, pose.bow_forward)
    scale = max(float(getattr(pose, "render_scale", 1.0)), 1e-6)
    centre = unproject(pose.anchor, config.BOW_RENDER_DEPTH_M / scale)

    # The body may be canted for legibility; the nock may not. An archer
    # sights along the arrow, so a bow seen from directly behind is a vertical
    # sliver — correct, but it reads as a stick. Canting the BODY alone shows
    # the limb curve, while the nock (and so the string's apex and the arrow)
    # keeps the true aim that acceptance 18 pins to the sight.
    body_rot = rot @ _yaw(math.radians(config.BOW_RENDER_CANT_DEG))

    top_m, bottom_m = tip_anchors(asset.bow)
    nock_m = nock_point(pose.power)
    # The nock rides WITH the canted body, so the string still runs tip to tip
    # through it instead of cutting across the limbs. Only the arrow's
    # direction stays true: it starts at this nock and points along the real
    # forward, which is what keeps its vanishing point on the sight.
    nock_world = (nock_m @ body_rot.T) + centre
    tips_px = project((np.stack([top_m, bottom_m]) @ body_rot.T) + centre)
    nock_px = project(nock_world[None, :])[0]

    arrow = None
    if asset.arrow is not None and pose.state == BowState.DRAWN:
        # The arrow's own origin is its centre, so slide it forward half its
        # length to sit its nock end on the string.
        shift = np.array([0.0, 0.0, float(asset.arrow.extent[2]) / 2.0], dtype=np.float32)
        nocked = replace(asset.arrow, positions=asset.arrow.positions + shift)
        arrow = _place_part(nocked, rot, nock_world)

    return PlacedBow(
        bow=_place_part(asset.bow, body_rot, centre),
        arrow=arrow,
        # Plain Python floats, not numpy scalars: pygame's draw calls reject
        # a numpy float32 pair with "invalid start_pos argument".
        tips_px=(_pair(tips_px[0]), _pair(tips_px[1])),
        nock_px=_pair(nock_px),
        centre_m=(float(centre[0]), float(centre[1]), float(centre[2])),
    )


# Fletching detection, shared by loaded and procedural arrows. The vanes sit
# at the rear of the shaft and stand well clear of it, which is enough to find
# them without knowing how the mesh was authored.
FLETCH_REAR_FRACTION = 0.35   # of the arrow's length, measured from the nock
FLETCH_RADIUS_FACTOR = 1.6    # ...and this much further from the axis than the shaft


def fletch_mask(arrow: Mesh) -> np.ndarray:
    """(M,) bool: which of the arrow's triangles are fletching.

    Colouring the vanes separately is most of what makes a drawn bow read,
    because the arrow points nearly at the camera: the shaft foreshortens to
    almost nothing and the fletching is the only part with any area. Found by
    shape rather than by name or by index, so it works the same on a loaded
    asset as on the procedural one.
    """
    if len(arrow.indices) == 0:
        return np.zeros(0, dtype=bool)
    pos = arrow.positions
    z = pos[:, 2]
    lo, hi = float(z.min()), float(z.max())
    radius = np.hypot(pos[:, 0], pos[:, 1])
    shaft_r = float(np.median(radius))
    if shaft_r <= 1e-9:
        return np.zeros(len(arrow.indices), dtype=bool)

    # Judged on the triangle's centroid, not on all three corners: a vane is
    # a sheet standing off the shaft, so its inner edge sits ON the shaft and
    # an all-corners rule rejects every triangle it is made of.
    cz = z[arrow.indices].mean(axis=1)
    cr = radius[arrow.indices].mean(axis=1)
    return (cz <= lo + (hi - lo) * FLETCH_REAR_FRACTION) & (
        cr >= shaft_r * FLETCH_RADIUS_FACTOR
    )


# -- the procedural fallback bow ----------------------------------------------


def _ring(centre: np.ndarray, rx: float, ry: float, n: int) -> np.ndarray:
    """One elliptical cross-section in the X/Y plane at `centre`."""
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = centre[0] + rx * np.cos(a)
    pts[:, 1] = centre[1]
    pts[:, 2] = centre[2] + ry * np.sin(a)
    return pts


def _tube(centres: np.ndarray, rx: np.ndarray, ry: np.ndarray, n: int = 8):
    """A closed tube through `centres`, with per-station elliptical radii.

    Rings lie in X/Z and are stacked along the centreline's own Y, which is
    all a bow limb needs: it bends in Z and runs in Y, never doubling back.
    """
    rings = [_ring(c, float(rx[i]), float(ry[i]), n) for i, c in enumerate(centres)]
    positions = np.concatenate(rings)
    tris = []
    for s in range(len(rings) - 1):
        a0, b0 = s * n, (s + 1) * n
        for k in range(n):
            k2 = (k + 1) % n
            tris.append((a0 + k, b0 + k, b0 + k2))
            tris.append((a0 + k, b0 + k2, a0 + k2))
    return positions, np.asarray(tris, dtype=np.uint32)


def _box(half: tuple[float, float, float], centre=(0.0, 0.0, 0.0)):
    hx, hy, hz = half
    cx, cy, cz = centre
    v = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.uint32)
    return v, f


def _weld(parts) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (positions, indices) pairs into one mesh."""
    pos, idx, base = [], [], 0
    for p, i in parts:
        pos.append(p)
        idx.append(i + base)
        base += len(p)
    return np.concatenate(pos), np.concatenate(idx)


# Power is quantized before the mesh is built so the cache below actually
# hits: power moves continuously, and a fresh mesh per frame costs more than
# the painter does. One step changes limb flex by DRAW_FLEX * L * STEP, about
# 1.2 mm on a 480 px bow — under a pixel.
PROCEDURAL_POWER_STEP = 0.02


def procedural_asset(power: float = 0.0, limb_span_m: float = LIMB_SPAN_M) -> BowAsset:
    """The bow to draw when no asset file is present (PLAN.md §4.7.8).

    Cached on quantized power. The returned BowAsset is shared, so callers
    must treat it as read-only — `place_bow` only ever copies out of it.

    Built in the same model space as a loaded asset and returned as the same
    BowAsset, so the renderer has one path rather than a 3D one for files and
    a flat screen-space one for everybody else. That matters because the
    fallback is what a fresh clone gets: without this the bow would ignore
    `bow_forward` and `bow_up` entirely and stop responding to the pose.

    The limbs flex with `power`: the centreline pulls back along -Z as
    `BRACE_FLEX + power * DRAW_FLEX`, with a recurve kicking the outer fifth
    of each limb forward again, so a hard draw visibly bends the bow.
    """
    step = PROCEDURAL_POWER_STEP
    quantized = round(min(max(float(power), 0.0), 1.0) / step) * step
    return _procedural_cached(quantized, float(limb_span_m))


@functools.lru_cache(maxsize=96)
def _procedural_cached(power: float, limb_span_m: float) -> BowAsset:
    L = limb_span_m
    flex = (config.BOW_BRACE_FLEX + float(power) * config.BOW_DRAW_FLEX) * L
    recurve = config.BOW_RECURVE * L

    stations = 10
    t = np.linspace(0.0, 1.0, stations)
    z = -flex * t**2 + recurve * np.maximum(0.0, t - 0.8) ** 2
    rx = (0.06 + (0.022 - 0.06) * t) * L      # wide at the riser, thin at the tip
    ry = (0.018 + (0.008 - 0.018) * t) * L

    parts = []
    for sign in (1.0, -1.0):
        y = sign * (0.17 + (0.50 - 0.17) * t) * L
        centres = np.stack([np.zeros(stations), y, z], axis=1).astype(np.float32)
        parts.append(_tube(centres, rx, ry))

    parts.append(_box((0.045 * L / 2, 0.17 * L, 0.07 * L / 2)))          # riser
    parts.append(_box((0.05 * L / 2, 0.06 * L, 0.075 * L / 2)))          # grip wrap
    pos, idx = _weld(parts)
    bow = Mesh(name="Bow", positions=pos, normals=compute_normals(pos, idx),
               indices=idx, material="procedural")

    shaft_len = 0.95 * L
    t2 = np.linspace(0.0, 1.0, 4)
    centres = np.stack([np.zeros(4), np.zeros(4), t2 * shaft_len], axis=1).astype(np.float32)
    # Rings are built in X/Z and stacked along Y, so an arrow lying along Z is
    # built along Y here and rotated into place afterwards.
    shaft_pos, shaft_idx = _tube(
        np.stack([np.zeros(4), t2 * shaft_len, np.zeros(4)], axis=1).astype(np.float32),
        np.full(4, 0.007 * L), np.full(4, 0.007 * L), n=6,
    )
    head_pos, head_idx = _tube(
        np.array([[0, shaft_len, 0], [0, shaft_len + 0.06 * L, 0]], dtype=np.float32),
        np.array([0.016 * L, 0.001 * L]), np.array([0.016 * L, 0.001 * L]), n=6,
    )
    # Three fletches over the rear 0.12 L (§4.7.8). They are most of what the
    # player actually sees: the arrow points nearly at the camera, so the
    # shaft foreshortens to almost nothing and the vanes are what say "nocked".
    fletch_len, fletch_rise = 0.12 * L, 0.030 * L
    vanes = []
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0
        ux, uz = math.cos(a), math.sin(a)
        r0 = 0.007 * L
        v = np.array([
            [ux * r0, 0.02 * L, uz * r0],
            [ux * r0, 0.02 * L + fletch_len, uz * r0],
            [ux * (r0 + fletch_rise), 0.02 * L + fletch_len * 0.75,
             uz * (r0 + fletch_rise)],
            [ux * (r0 + fletch_rise), 0.02 * L + fletch_len * 0.15,
             uz * (r0 + fletch_rise)],
        ], dtype=np.float32)
        f = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.uint32)
        vanes.append((v, f))   # both windings: a vane is a flat sheet, seen from either side

    apos, aidx = _weld([(shaft_pos, shaft_idx), (head_pos, head_idx)] + vanes)
    # Y (built) -> Z (model forward), and recentre so the mesh straddles its
    # own origin the way a loaded, fitted arrow does.
    apos = np.stack([apos[:, 0], apos[:, 2], apos[:, 1]], axis=1)
    apos = apos - (apos.max(axis=0) + apos.min(axis=0)) / 2.0
    arrow = Mesh(name="Arrow", positions=apos.astype(np.float32),
                 normals=compute_normals(apos.astype(np.float32), aidx),
                 indices=aidx, material="procedural")

    return BowAsset(bow=bow, arrow=arrow, source="<procedural>",
                    how="procedural", fitted=True)
