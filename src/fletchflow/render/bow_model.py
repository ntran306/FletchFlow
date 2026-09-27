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

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from fletchflow import config

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
