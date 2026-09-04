"""True-3D bow body rendered with moderngl into an offscreen framebuffer.

The mesh is generated procedurally each frame (no asset files): each limb is
a tube swept along the flex-dependent bezier, ring radius tapering from the
riser to the tips, plus a thicker grip tube. A minimal Lambert + rim shader
lights it as real geometry, the FBO is read back as RGBA, and the result is
blitted over the camera feed. Rebuilding ~1.6k vertices per frame in numpy
costs well under a millisecond.

Model space matches bow.py's geometry convention: the bow spans the Y axis,
+X is the aim direction, and screen orientation is applied by rotating the
model by the aim angle (Z rotation only, so the 2D-computed string tips line
up exactly with the rendered wood).

Construction can fail on machines without OpenGL 3.3 — callers must be
ready to fall back to the 2D body (see __main__.py).
"""

from __future__ import annotations

import math

import moderngl
import numpy as np
import pygame

from fletchflow import config
from fletchflow.input.mapping import BowPose
from fletchflow.render.bow import BowGeometry

FBO_SIZE = 512  # square RGBA target; bow span 340 px fits with flex margin

VERT_SHADER = """
#version 330
uniform mat3 u_rot;      // model -> view rotation (aim angle)
uniform vec2 u_scale;    // model px -> clip space
in vec3 in_pos;
in vec3 in_normal;
out vec3 v_normal;
void main() {
    vec3 p = u_rot * in_pos;
    v_normal = u_rot * in_normal;
    gl_Position = vec4(p.xy * u_scale, p.z * 0.001, 1.0);
}
"""

FRAG_SHADER = """
#version 330
uniform vec3 u_color;
in vec3 v_normal;
out vec4 f_color;
const vec3 LIGHT = normalize(vec3(-0.35, -0.45, 0.82));
void main() {
    vec3 n = normalize(v_normal);
    float lambert = max(dot(n, LIGHT), 0.0);
    float rim = pow(1.0 - abs(n.z), 2.0) * 0.25;
    vec3 color = u_color * (0.35 + 0.75 * lambert) + vec3(rim);
    f_color = vec4(color, 1.0);
}
"""

WOOD_COLOR = (0.478, 0.302, 0.149)  # matches the 2D LIMB_MID tone
GRIP_COLOR = (0.216, 0.169, 0.145)

RINGS_PER_LIMB = 25
RING_SIDES = 8


class BowBodyRenderer3D:
    def __init__(self) -> None:
        self._ctx = moderngl.create_context(standalone=True)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._prog = self._ctx.program(
            vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER
        )
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((FBO_SIZE, FBO_SIZE), 4)],
            depth_attachment=self._ctx.depth_renderbuffer((FBO_SIZE, FBO_SIZE)),
        )
        # Vertex layout is constant; only positions/normals change per frame
        self._index_data = _tube_indices()
        self._vbo = self._ctx.buffer(reserve=_vertex_count() * 6 * 4, dynamic=True)
        self._ibo = self._ctx.buffer(self._index_data.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 3f", "in_pos", "in_normal")],
            index_buffer=self._ibo,
        )
        self._prog["u_scale"] = (2.0 / FBO_SIZE, -2.0 / FBO_SIZE)  # y-down
        self._cache_key: tuple[int, int, int] | None = None
        self._cache_image: pygame.Surface | None = None

    def draw(self, surface: pygame.Surface, pose: BowPose, geom: BowGeometry) -> None:
        # Re-render only when orientation or flex meaningfully changed
        # (0.5° / 0.5 px quantization); otherwise reuse the cached surface.
        # The 60 Hz render loop outpaces ~30 Hz tracking, so at least half
        # of all draws are free, and a steady hold costs just the blit.
        # geom.flex/belly are already scale-multiplied (bow.py's compute_geometry);
        # the GL mesh is built at unit scale and blown up at composite time
        # instead, so divide back out before keying/building.
        angle = math.atan2(geom.aim[1], geom.aim[0])
        unscaled_flex = geom.flex / geom.scale
        key = (round(math.degrees(angle) * 2), round(unscaled_flex * 2), round(geom.scale * 20))
        if key != self._cache_key:
            self._cache_key = key
            self._cache_image = self._render_body(angle, geom)

        img = self._cache_image
        surface.blit(
            img,
            (geom.anchor[0] - img.get_width() / 2, geom.anchor[1] - img.get_height() / 2),
        )

    def _render_body(self, angle: float, geom: BowGeometry) -> pygame.Surface:
        unscaled_flex = geom.flex / geom.scale
        unscaled_belly = geom.belly / geom.scale
        verts = _build_mesh(unscaled_flex, unscaled_belly)
        self._vbo.write(verts.tobytes())

        c, s = math.cos(angle), math.sin(angle)
        # Rotate model +X onto the aim direction (screen coords, y down).
        # GLSL mat3 uniforms are COLUMN-major: this tuple is columns, so the
        # first three values are column 0 — (c, s) puts +X onto (cos, sin)
        self._prog["u_rot"] = (c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)

        self._fbo.use()
        self._ctx.clear(0.0, 0.0, 0.0, 0.0)
        self._prog["u_color"] = WOOD_COLOR
        self._vao.render(mode=moderngl.TRIANGLES, vertices=self._limb_index_count)
        self._prog["u_color"] = GRIP_COLOR
        self._vao.render(
            mode=moderngl.TRIANGLES,
            vertices=len(self._index_data) - self._limb_index_count,
            first=self._limb_index_count,
        )

        raw = self._fbo.read(components=4)
        image = pygame.image.frombuffer(raw, (FBO_SIZE, FBO_SIZE), "RGBA")
        image = pygame.transform.flip(image, False, True)  # GL origin: bottom-left
        # Match the display pixel format once here — an unconverted RGBA
        # surface costs ~8 ms PER BLIT in software conversion
        image = image.convert_alpha()
        # The GL mesh itself is built at unit scale; blow the composited
        # image up/down here so its tips line up with bow.py's scaled 2D tips.
        if abs(geom.scale - 1.0) > 0.02:
            size = (round(FBO_SIZE * geom.scale), round(FBO_SIZE * geom.scale))
            image = pygame.transform.smoothscale(image, size)
        return image

    @property
    def _limb_index_count(self) -> int:
        return _segment_index_count(RINGS_PER_LIMB) * 2


# -- procedural mesh -------------------------------------------------------


def _segment_index_count(rings: int) -> int:
    return (rings - 1) * RING_SIDES * 6


def _vertex_count() -> int:
    return (RINGS_PER_LIMB * 2 + RINGS_PER_LIMB) * RING_SIDES


def _tube_indices() -> np.ndarray:
    """Index buffer for three tubes: two limbs + grip (constant topology)."""
    indices = []
    base = 0
    for _ in range(3):
        for ring in range(RINGS_PER_LIMB - 1):
            for side in range(RING_SIDES):
                a = base + ring * RING_SIDES + side
                b = base + ring * RING_SIDES + (side + 1) % RING_SIDES
                c = a + RING_SIDES
                d = b + RING_SIDES
                indices += [a, c, b, b, c, d]
        base += RINGS_PER_LIMB * RING_SIDES
    return np.array(indices, dtype="u4")


_RING_ANGLES = np.linspace(0, 2 * math.pi, RING_SIDES, endpoint=False)
_COS_A = np.cos(_RING_ANGLES)[None, :]  # (1, R)
_SIN_A = np.sin(_RING_ANGLES)[None, :]


def _sweep_tube(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Rings of vertices (pos+normal interleaved) around a planar polyline.
    Fully vectorized — the per-ring Python loop cost 6 ms/frame."""
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    # cross([tx, ty, 0], [0, 0, 1]) = (ty, -tx, 0): in-plane ring direction
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])

    r = radii[:, None]  # (N, 1)
    ring_nx = normals[:, 0:1] * _COS_A          # (N, R)
    ring_ny = normals[:, 1:2] * _COS_A
    ring_nz = np.broadcast_to(_SIN_A, ring_nx.shape)

    out = np.empty((len(points), RING_SIDES, 6), dtype="f4")
    out[:, :, 0] = points[:, 0:1] + ring_nx * r
    out[:, :, 1] = points[:, 1:2] + ring_ny * r
    out[:, :, 2] = ring_nz * r
    out[:, :, 3] = ring_nx
    out[:, :, 4] = ring_ny
    out[:, :, 5] = ring_nz
    return out.reshape(-1, 6)


def _build_mesh(flex: float, belly: float) -> np.ndarray:
    """Positions+normals for both limbs and the grip, in model px units."""
    half = config.BOW_SPAN_PX / 2.0
    t = np.linspace(0.0, 1.0, RINGS_PER_LIMB)[:, None]
    u = 1.0 - t
    parts = []
    for sign in (1.0, -1.0):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([belly, sign * half * 0.55])
        p2 = np.array([-flex, sign * half])
        curve = u * u * p0 + 2 * u * t * p1 + t * t * p2
        radii = np.linspace(9.0, 3.5, RINGS_PER_LIMB)
        parts.append(_sweep_tube(curve, radii))
    # Grip: short straight tube over the riser
    grip_y = np.linspace(-32.0, 32.0, RINGS_PER_LIMB)
    grip = np.column_stack([np.full_like(grip_y, 1.0), grip_y])
    parts.append(_sweep_tube(grip, np.full(RINGS_PER_LIMB, 11.0)))
    return np.concatenate(parts)
