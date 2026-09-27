"""Loading a downloaded bow asset into model space (PLAN.md §4.7.8 / §4.7.8a).

The asset file itself is third-party and untracked pending its licence, so the
tests that read it skip when it is absent; everything else is driven by small
OBJ strings written inline, which also document the format corners the parser
has to survive.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from fletchflow import config
from fletchflow.render import bow_model as bm

# Anchored to the repo, not the cwd: a relative path would silently skip
# these tests whenever pytest is run from anywhere else.
ASSET = Path(__file__).resolve().parent.parent / "assets" / "models" / "ANIMATEDBOW.obj"
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="asset not present")


# -- OBJ parsing --------------------------------------------------------------


QUAD = """
# a single quad, with uvs and normals
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
o Panel
usemtl paint
f 1/1/1 2/2/1 3/3/1 4/4/1
"""


def test_quads_are_triangulated():
    (mesh,) = bm.parse_obj(QUAD)
    assert mesh.name == "Panel"
    assert mesh.material == "paint"
    assert mesh.indices.shape == (2, 3), "a quad must become two triangles"
    assert mesh.uvs is not None
    assert np.allclose(mesh.normals, [0, 0, 1])


def test_ngons_are_fan_triangulated():
    src = "\n".join(
        ["v 0 0 0", "v 1 0 0", "v 2 1 0", "v 1 2 0", "v 0 2 0", "o P", "f 1 2 3 4 5"]
    )
    (mesh,) = bm.parse_obj(src)
    assert mesh.indices.shape == (3, 3), "a pentagon is three triangles"


def test_objects_share_one_vertex_pool_and_split_into_meshes():
    """OBJ numbers vertices globally across objects — a parser that restarts
    the count per object silently builds the second part out of the first
    part's corners."""
    src = "\n".join([
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "v 5 0 0", "v 6 0 0", "v 5 1 0",
        "o First", "f 1 2 3",
        "o Second", "f 4 5 6",
    ])
    first, second = bm.parse_obj(src)
    assert first.name == "First" and second.name == "Second"
    assert second.positions[:, 0].min() == 5.0


def test_negative_indices_are_relative_to_the_end():
    src = "\n".join(["v 0 0 0", "v 1 0 0", "v 0 1 0", "o P", "f -3 -2 -1"])
    (mesh,) = bm.parse_obj(src)
    assert len(mesh.indices) == 1
    assert np.allclose(sorted(mesh.positions[:, 0]), [0.0, 0.0, 1.0])


def test_normals_are_computed_when_the_file_omits_them():
    src = "\n".join(["v 0 0 0", "v 1 0 0", "v 0 1 0", "o P", "f 1 2 3"])
    (mesh,) = bm.parse_obj(src)
    assert np.allclose(np.abs(mesh.normals), [0, 0, 1]), mesh.normals


def test_faces_without_uvs_still_parse():
    src = "\n".join(["v 0 0 0", "v 1 0 0", "v 0 1 0", "vn 0 0 1",
                     "o P", "f 1//1 2//1 3//1"])
    (mesh,) = bm.parse_obj(src)
    assert mesh.uvs is None
    assert np.allclose(mesh.normals, [0, 0, 1])


# -- part identification ------------------------------------------------------


def _mesh(name, extent=(1.0, 1.0, 1.0), material=""):
    """A box of the given extent — only its name, material and size matter."""
    e = np.asarray(extent, dtype=np.float32)
    pos = np.array([[0, 0, 0], e, [e[0], 0, 0]], dtype=np.float32)
    return bm.Mesh(name=name, positions=pos,
                   normals=np.tile([0, 0, 1], (3, 1)).astype(np.float32),
                   indices=np.array([[0, 1, 2]], dtype=np.uint32), material=material)


def test_identify_prefers_object_names():
    bow, arrow, how = bm.identify([_mesh("Arrow"), _mesh("Bow")])
    assert bow.name == "Bow" and arrow.name == "Arrow"
    assert "object name" in how


def test_identify_falls_back_to_material_names():
    """The real asset's case: objects are Cube_Cube.001 / Cube.001_Cube.002
    while the materials say bow / dirty_arrow."""
    parts = [_mesh("Cube_Cube.001", material="dirty_arrow"),
             _mesh("Cube.001_Cube.002", material="bow")]
    bow, arrow, how = bm.identify(parts)
    assert bow.material == "bow" and arrow.material == "dirty_arrow"
    assert "material name" in how


def test_identify_falls_back_to_shape():
    bow, arrow, how = bm.identify([_mesh("Cube", (0.1, 0.1, 2.0)),
                                   _mesh("Cube.001", (0.1, 0.1, 8.0))])
    assert bow.name == "Cube.001", "the longer part is the bow"
    assert "shape" in how


def test_a_modelled_string_part_is_discarded():
    """A separately named string is dropped, not mistaken for the arrow."""
    parts = [_mesh("Bow"), _mesh("Arrow"), _mesh("String")]
    bow, arrow, _ = bm.identify(parts)
    assert bow.name == "Bow" and arrow.name == "Arrow"


def test_identify_rejects_an_empty_asset():
    with pytest.raises(bm.AssetError):
        bm.identify([])


# -- stripping a modelled string ----------------------------------------------


def _bar(x0, x1, y0, y1, z0, z1, start):
    """Two triangles spanning a box, as OBJ text plus the next vertex index."""
    vs = [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)]
    lines = [f"v {x} {y} {z}" for x, y, z in vs]
    i = start
    lines.append(f"f {i} {i+1} {i+2}")
    lines.append(f"f {i} {i+2} {i+3}")
    return lines, start + 4


def test_strip_string_removes_a_long_thin_component():
    """Body: full length, deep. String: full length, hair thin. Grip: short."""
    lines, n = ["o Bow"], 1
    body, n = _bar(-0.01, 0.01, -1.0, 1.0, 0.0, 0.40, n)
    string, n = _bar(-0.001, 0.001, -0.98, 0.98, -0.40, -0.399, n)
    grip, n = _bar(-0.02, 0.02, -0.05, 0.05, 0.0, 0.03, n)
    (mesh,) = bm.parse_obj("\n".join(lines + body + string + grip))

    stripped, removed = bm.strip_string(mesh, limb_axis=1)
    assert removed == 2, "exactly the string's two triangles"
    assert len(stripped.indices) == 4, "body and grip wrap both survive"
    # nothing is left back where the string was
    assert stripped.positions[:, 2].min() > -0.1


def test_strip_string_leaves_a_bow_that_has_none():
    lines, n = ["o Bow"], 1
    body, n = _bar(-0.01, 0.01, -1.0, 1.0, 0.0, 0.40, n)
    (mesh,) = bm.parse_obj("\n".join(lines + body))
    stripped, removed = bm.strip_string(mesh, limb_axis=1)
    assert removed == 0
    assert len(stripped.indices) == len(mesh.indices)


# -- fitting ------------------------------------------------------------------


@needs_asset
def test_the_real_asset_fits_into_model_space():
    """Acceptance 24/25: parts found, limb span exact, grip at the origin."""
    asset = bm.load_asset(ASSET)
    assert asset.arrow is not None, "the bow and arrow must be separable"
    assert asset.fitted

    for mesh in (asset.bow, asset.arrow):
        assert np.isfinite(mesh.positions).all()
        assert np.isfinite(mesh.normals).all()
        assert mesh.indices.max() < len(mesh.positions)

    span = float(asset.bow.extent[1])
    assert abs(span - bm.LIMB_SPAN_M) < 1e-3, f"limb span {span}"
    assert np.allclose(asset.bow.centre, 0.0, atol=0.01 * bm.LIMB_SPAN_M)

    # limbs run along Y, the riser is deep in Z, and the bow is thin in X
    e = asset.bow.extent
    assert e[1] > e[2] > e[0]
    # the arrow's long axis is forward, +Z
    ae = asset.arrow.extent
    assert ae[2] > 5 * max(ae[0], ae[1]), f"arrow not aligned to +Z: {ae}"


@needs_asset
def test_the_real_asset_has_its_modelled_string_removed():
    asset = bm.load_asset(ASSET)
    assert "string stripped" in asset.how
    # the string ran the full height at a near-constant Z; nothing that thin
    # and that long should be left
    for tris in bm.split_components(asset.bow):
        pts = asset.bow.positions[np.unique(tris.ravel())]
        ext = pts.max(axis=0) - pts.min(axis=0)
        thin = max(ext[0], ext[2]) / bm.LIMB_SPAN_M <= 0.05
        assert not (ext[1] / bm.LIMB_SPAN_M >= 0.7 and thin), ext


def test_fitting_does_not_mirror_the_model():
    """The frame is built as a cross product, so it must stay right-handed —
    a det -1 rotation would flip the bow and nobody would notice until the
    lighting came out inside-out."""
    lines, n = ["o Bow"], 1
    body, n = _bar(-0.01, 0.01, -1.0, 1.0, 0.0, 0.40, n)
    (mesh,) = bm.parse_obj("\n".join(lines + body))
    asset = bm.fit_asset(bm.BowAsset(bow=mesh, arrow=None))

    before = float(np.linalg.det(np.cov(mesh.positions.T) + np.eye(3) * 1e-9))
    after = float(np.linalg.det(np.cov(asset.bow.positions.T) + np.eye(3) * 1e-9))
    assert math.copysign(1, before) == math.copysign(1, after)
    assert abs(float(asset.bow.extent[1]) - bm.LIMB_SPAN_M) < 1e-4


# -- loading errors -----------------------------------------------------------


def test_missing_and_unsupported_files_raise_asset_error(tmp_path):
    """Acceptance 27: the caller needs one exception type to fall back from."""
    with pytest.raises(bm.AssetError):
        bm.load_asset(tmp_path / "nope.obj")
    other = tmp_path / "bow.fbx"
    other.write_bytes(b"not an obj")
    with pytest.raises(bm.AssetError):
        bm.load_asset(other)
    empty = tmp_path / "empty.obj"
    empty.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(bm.AssetError):
        bm.load_asset(empty)


# -- placement and projection -------------------------------------------------


def test_projection_matches_the_game_pinhole():
    """The bow must share the world's camera, not its own — otherwise it sits
    at a different perspective from the targets it is aimed at."""
    mesh = _mesh("P")
    world = bm.to_world(mesh, (0.0, 0.0, 2.0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert np.allclose(world, mesh.positions + np.array([0, 0, 2.0]))

    pts = np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [0.0, 0.1, 1.0]], dtype=np.float32)
    px = bm.project(pts)
    cx, cy = config.WINDOW_SIZE[0] / 2.0, config.WINDOW_SIZE[1] / 2.0
    assert np.allclose(px[0], [cx, cy])
    assert px[1][0] == pytest.approx(cx + config.FOCAL_PX * 0.2 / 2.0)
    assert px[2][1] == pytest.approx(cy + config.FOCAL_PX * 0.1 / 1.0)


def test_projection_survives_a_vertex_behind_the_eye():
    """Clamped to the near plane rather than dropped, so a triangle that
    briefly crosses behind the camera stretches instead of vanishing."""
    pts = np.array([[0.1, 0.0, -3.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    px = bm.project(pts)
    assert np.isfinite(px).all()


# -- placing a pose (PLAN.md §4.7.8, acceptance 17-19) ------------------------

from fletchflow.input.bow_input import BowSnapshot, BowState  # noqa: E402
from fletchflow.input.mapping import BowPose, Mapper  # noqa: E402

CX, CY = config.WINDOW_SIZE[0] / 2.0, config.WINDOW_SIZE[1] / 2.0


def _pose(state=BowState.DRAWN, power=0.9, yaw_deg=0.0, pitch_deg=0.0,
          anchor=(520.0, 400.0), render_scale=1.0):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = (math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y))
    # Perpendicular to forward, with world +y pointing DOWN. Getting these
    # signs wrong gives an `up` that is not square to `forward` at all, which
    # a basis that quietly re-orthogonalizes would absorb without complaint.
    up = (math.sin(p) * math.sin(y), -math.cos(p), math.sin(p) * math.cos(y))
    assert abs(sum(a * b for a, b in zip(forward, up))) < 1e-9
    return BowPose(
        anchor=anchor, draw_point=(anchor[0] - 60, anchor[1] + 40), aim=(0.0, -1.0),
        power=power, state=state, fire=None, scale=1.0, sight=(CX, CY),
        bow_forward=forward, bow_up=up, aim_weight=1.0, render_scale=render_scale,
    )


def _point_line_distance(p, a, b) -> float:
    """Perpendicular distance from p to the infinite line through a and b."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    vx, vy = bx - ax, by - ay
    n = math.hypot(vx, vy)
    if n < 1e-9:
        return math.dist(p, (ax, ay))
    return abs(vx * (ay - p[1]) - vy * (ax - p[0])) / n


def _vanishing_point(forward):
    """Where a line of this direction converges on screen — origin-independent,
    which is why canting the arrow's start point cannot move it."""
    fx, fy, fz = forward
    return (CX + config.FOCAL_PX * fx / fz, CY + config.FOCAL_PX * fy / fz)


def test_pose_basis_is_a_proper_rotation():
    """det -1 would mirror the bow — invisible on a symmetric mesh right up
    until the arrow rest appears on the wrong side."""
    for yaw in (-40, -15, 0, 15, 40):
        for pitch in (-20, 0, 20):
            pose = _pose(yaw_deg=yaw, pitch_deg=pitch)
            rot = bm.pose_basis(pose.bow_up, pose.bow_forward)
            assert float(np.linalg.det(rot)) == pytest.approx(1.0, abs=1e-4)
            assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-4)


@needs_asset
def test_the_cant_leaves_the_arrow_direction_alone():
    """The cant is a lie told about the body only. If it reached the arrow,
    the arrow would visibly point somewhere other than the crosshair."""
    asset = bm.load_asset(ASSET)
    pose = _pose(yaw_deg=12.0, pitch_deg=-6.0)

    vp = _vanishing_point(pose.bow_forward)
    original = config.BOW_RENDER_CANT_DEG
    try:
        widths = []
        for cant in (0.0, 35.0, 55.0):
            config.BOW_RENDER_CANT_DEG = cant
            placed = bm.place_bow(pose, asset)
            assert placed.arrow is not None

            # The arrow's nock end rides with the canted body -- that is what
            # keeps the string inside the bow -- so its position is expected
            # to move. Its DIRECTION is what must not.
            #
            # A 3D line projects to a 2D line that passes through its own
            # vanishing point, so anchoring at the nock (which sits on the
            # arrow's axis) and extending through the far end must aim at vp.
            # The arrow's two most distant screen points are NOT its ends: it
            # points nearly at the camera, and the near end is magnified
            # enough that the fletching outspans the foreshortened shaft.
            pts = placed.arrow.tris_px.reshape(-1, 2)
            nock = np.asarray(placed.nock_px, dtype=np.float32)
            far = np.linalg.norm(pts - nock, axis=1)
            # Mean of the farthest few percent, not the single farthest point:
            # the head has width, so one extreme vertex sits off the axis and
            # tilts a line this short by several degrees.
            head = pts[far >= np.quantile(far, 0.95)].mean(axis=0)
            assert _point_line_distance(vp, nock, head) < 10.0, cant

            body = placed.bow.tris_px[:, :, 0]
            widths.append(float(body.max() - body.min()))
        # The cant must actually reach the body. Not asserted as monotonic:
        # it rotates the bow about its own up axis, so against a pose that is
        # already 12 deg off-axis it first swings the bow back toward edge-on
        # before opening it out again.
        assert len(set(round(w, 1) for w in widths)) == len(widths), widths
    finally:
        config.BOW_RENDER_CANT_DEG = original


@needs_asset
def test_the_arrow_vanishing_point_lands_on_the_sight():
    """Acceptance 18. The crosshair and the bow's forward axis both come from
    the same aim angles, so the arrow must converge on the crosshair — if
    these ever drift apart the player is aiming with a lying arrow."""
    mapper = Mapper()
    bow_m, draw_m = (0.0, 0.10, 0.45), (-0.10, 0.10, 0.80)
    pose = None
    for i in range(60):
        pose = mapper.map(BowSnapshot(
            timestamp_ms=(i + 1) * 33, state=BowState.DRAWN, anchor=(0.45, 0.5),
            draw_point=(0.5, 0.6), power=0.8, fired_power=None, scale=1.0,
            bow_position_m=bow_m, draw_position_m=draw_m,
        ))
    assert pose.sight is not None
    vp = _vanishing_point(pose.bow_forward)
    assert abs(vp[0] - pose.sight[0]) < 10.0, (vp, pose.sight)
    assert abs(vp[1] - pose.sight[1]) < 10.0, (vp, pose.sight)


@needs_asset
def test_the_string_runs_tip_to_nock_to_tip():
    """The nock rides with the canted body, so the string stays inside the
    bow instead of cutting across the limbs."""
    asset = bm.load_asset(ASSET)
    drawn = bm.place_bow(_pose(power=0.9), asset)
    top, bottom = drawn.tips_px
    assert top[1] < bottom[1], "tips are ordered top then bottom on screen"
    # the nock sits between the tips vertically and pulled off the tip line
    assert top[1] < drawn.nock_px[1] < bottom[1]

    relaxed = bm.place_bow(_pose(power=0.0), asset)
    pull_full = abs(drawn.nock_px[0] - (top[0] + bottom[0]) / 2.0)
    pull_rest = abs(relaxed.nock_px[0] - (top[0] + bottom[0]) / 2.0)
    assert pull_full > pull_rest, "a harder draw must pull the nock further back"


@needs_asset
def test_no_arrow_unless_drawn():
    """Same rule as the crosshair (§4.7.4): nothing is on the string yet."""
    asset = bm.load_asset(ASSET)
    for state in (BowState.DOCKED, BowState.HELD, BowState.RELEASED):
        assert bm.place_bow(_pose(state=state, power=0.0), asset).arrow is None
    assert bm.place_bow(_pose(state=BowState.DRAWN), asset).arrow is not None


@needs_asset
def test_render_scale_moves_the_bow_in_depth():
    """A player leaning in raises render_scale, which must put the bow NEARER
    and so bigger — the perspective way, not by scaling a sprite."""
    asset = bm.load_asset(ASSET)
    near = bm.place_bow(_pose(render_scale=1.25), asset)
    far = bm.place_bow(_pose(render_scale=0.80), asset)
    assert near.centre_m[2] < far.centre_m[2]

    def height(p):
        ys = p.bow.tris_px[:, :, 1]
        return float(ys.max() - ys.min())

    assert height(near) > height(far) * 1.3


@needs_asset
def test_every_state_and_angle_places_without_exception():
    """Acceptance 19: the renderer must never be the thing that crashes a run."""
    asset = bm.load_asset(ASSET)
    for state in BowState:
        for yaw in (-40.0, 0.0, 40.0):
            for pitch in (-25.0, 0.0, 25.0):
                for scale in (0.80, 1.0, 1.25):
                    placed = bm.place_bow(
                        _pose(state=state, yaw_deg=yaw, pitch_deg=pitch,
                              render_scale=scale), asset)
                    assert np.isfinite(placed.bow.tris_px).all()
                    assert all(math.isfinite(v) for v in placed.nock_px)


def test_pose_basis_preserves_forward_against_a_skewed_up():
    """Forward is the aim; up is only the roll about it.

    An `up` that is not square to `forward` has to be squared up, and which
    of the two gets adjusted is not a detail: the crosshair is computed from
    the same angles as forward (§4.7.4), so bending forward to suit the up
    vector would point the drawn arrow somewhere the player is not aiming.
    """
    forward = (0.2068, -0.1045, 0.9729)
    skewed_up = (0.0, -1.0, 0.0)          # ~6 deg off square at this pitch
    assert abs(sum(a * b for a, b in zip(forward, skewed_up))) > 0.05

    rot = bm.pose_basis(skewed_up, forward)
    expected = np.asarray(forward) / np.linalg.norm(forward)
    assert np.allclose(rot[:, 2], expected, atol=1e-6), rot[:, 2]
    assert float(np.linalg.det(rot)) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5)


def test_pose_basis_survives_up_parallel_to_forward():
    """No roll is recoverable, but it must not paint NaNs."""
    rot = bm.pose_basis((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    assert np.isfinite(rot).all()
    assert float(np.linalg.det(rot)) == pytest.approx(1.0, abs=1e-5)
