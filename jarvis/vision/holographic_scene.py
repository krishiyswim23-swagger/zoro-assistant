"""Stage 6: the holographic object's state + procedural point-cloud generation.

Deliberately pure/no graphics dependencies (no pygame, no OpenGL) so the
interesting logic — how gestures change rotation/zoom, which object is
active, and how "zooming in" changes the point cloud to *look* like you're
zooming into atoms — can be unit-tested without a display or a camera.
jarvis/vision/holographic_web.py streams this module's state to the browser
page (jarvis/vision/web/hologram.js) that actually draws it.

The "atoms" here are a stylized visual effect, not a physically accurate
molecule — the point is the zoom-in feel Tony Stark's UI has, not chemistry.
"""

from __future__ import annotations

import math

from jarvis.vision.gesture_engine import GestureEvent

OBJECTS = ("face", "sphere", "molecule", "cube")

_ROTATION_SENSITIVITY = 300.0  # degrees of rotation per unit of normalized hand movement
_MIN_ZOOM = 0.3
_MAX_ZOOM = 6.0

# --- face mesh -----------------------------------------------------------

_HEAD_RX, _HEAD_RY, _HEAD_RZ = 0.85, 1.05, 0.9
_HEAD_PHI_MIN, _HEAD_PHI_MAX = math.radians(8), math.radians(172)  # just short of the poles
_HEAD_N_THETA = 56  # goes all the way around (0..2*pi); a full solid head, not just a front slice
_HEAD_N_PHI = 36

_EYE_CENTERS = ((-0.32, 0.32), (0.32, 0.32))  # (x, y) in head-local units
_EYE_RADIUS_X, _EYE_RADIUS_Y = 0.14, 0.09
_EYE_SOCKET_DEPTH = 0.05
_MOUTH_CENTER_Y = -0.5
_MOUTH_HALF_WIDTH = 0.26
_NOSE_HALF_WIDTH = 0.06
_NOSE_Y_RANGE = (-0.15, 0.25)  # between the eyes and the mouth
_NOSE_RIDGE_HEIGHT = 0.06

_MIN_BRIGHTNESS = 0.12  # floor so shadowed/silhouette surface stays faintly visible, not pure black
_SOCKET_DARKEN_FACTOR = 0.15  # how much dimmer eye sockets / an open mouth read, vs. plain skin


def _normalize3(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vec
    length = math.sqrt(x * x + y * y + z * z)
    return (x / length, y / length, z / length)


_LIGHT_DIR = _normalize3((0.25, 0.35, 0.9))  # mostly frontal, slightly above — a "screen glow" look


def generate_sphere_points(n: int = 200, radius: float = 1.0) -> list[tuple[float, float, float]]:
    """An evenly distributed point cloud over a sphere's surface (Fibonacci sphere)."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, radius, 0.0)]

    golden_angle = math.pi * (3 - math.sqrt(5))
    points = []
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        ring_radius = math.sqrt(max(0.0, 1 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * ring_radius
        z = math.sin(theta) * ring_radius
        points.append((x * radius, y * radius, z * radius))
    return points


def generate_cube_wireframe_points(size: float = 1.0, subdivisions: int = 10) -> list[tuple[float, float, float]]:
    """Points sampled along a cube's 12 edges, for a wireframe-style point cloud."""
    half = size / 2
    corners = [
        (-half, -half, -half), (half, -half, -half), (half, half, -half), (-half, half, -half),
        (-half, -half, half), (half, -half, half), (half, half, half), (-half, half, half),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

    points = []
    for a, b in edges:
        ax, ay, az = corners[a]
        bx, by, bz = corners[b]
        for step in range(subdivisions + 1):
            t = step / subdivisions
            points.append((ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t))
    return points


def _shell_points(center: tuple[float, float, float], radius: float, n: int) -> list[tuple[float, float, float]]:
    cx, cy, cz = center
    return [(cx + x, cy + y, cz + z) for x, y, z in generate_sphere_points(n=n, radius=radius)]


_DEFAULT_ATOMS = ((0.0, 0.0, 0.0), (0.6, 0.6, 0.6), (-0.6, 0.6, -0.6), (0.6, -0.6, -0.6), (-0.6, -0.6, 0.6))


def generate_molecule_points(
    lod_level: int, atoms: tuple[tuple[float, float, float], ...] = _DEFAULT_ATOMS
) -> list[tuple[float, float, float]]:
    """A handful of 'atom' points; deeper LOD levels add finer shells around
    each one, so zooming in reveals more and more structure."""
    points = list(atoms)
    if lod_level >= 1:
        for atom in atoms:
            points.extend(_shell_points(atom, radius=0.15, n=20))
    if lod_level >= 2:
        for atom in atoms:
            points.extend(_shell_points(atom, radius=0.28, n=60))
    return points


def _head_vertex(theta: float, phi: float, mouth_openness: float, eye_openness: float) -> tuple[float, float, float, float]:
    """One vertex of the head mesh: base ellipsoid position, nudged in/out
    for facial features, shaded by how directly its (pre-nudge) surface
    normal faces a fixed light direction."""
    nx = math.sin(phi) * math.sin(theta)
    ny = math.cos(phi)
    nz = math.sin(phi) * math.cos(theta)

    x, y, z = nx * _HEAD_RX, ny * _HEAD_RY, nz * _HEAD_RZ
    brightness = max(_MIN_BRIGHTNESS, nx * _LIGHT_DIR[0] + ny * _LIGHT_DIR[1] + nz * _LIGHT_DIR[2])
    is_frontal = nz > 0.15  # only the front-facing surface carries facial features

    if is_frontal:
        mouth_gap_height = 0.03 + 0.22 * mouth_openness
        for eye_x, eye_y in _EYE_CENTERS:
            dx = (x - eye_x) / _EYE_RADIUS_X
            dy = (y - eye_y) / _EYE_RADIUS_Y
            eye_amount = max(0.0, 1.0 - (dx * dx + dy * dy)) * eye_openness  # 1 at eye center, fading out
            if eye_amount > 0:
                z -= _EYE_SOCKET_DEPTH * eye_amount  # sink in, like a real eye socket
                brightness *= 1.0 - (1.0 - _SOCKET_DARKEN_FACTOR) * eye_amount

        if abs(x) < _MOUTH_HALF_WIDTH:
            dx = x / _MOUTH_HALF_WIDTH
            dy = (y - _MOUTH_CENTER_Y) / mouth_gap_height
            mouth_amount = max(0.0, 1.0 - (dx * dx + dy * dy))
            if mouth_amount > 0:
                z -= _EYE_SOCKET_DEPTH * mouth_amount
                brightness *= 1.0 - (1.0 - _SOCKET_DARKEN_FACTOR) * mouth_amount

        if abs(x) < _NOSE_HALF_WIDTH and _NOSE_Y_RANGE[0] < y < _NOSE_Y_RANGE[1]:
            # Push the nose ridge toward the viewer; already-computed lighting
            # then makes it read as a raised highlight without extra logic.
            z += _NOSE_RIDGE_HEIGHT

    return (x, y, z, brightness)


def generate_head_mesh(
    mouth_openness: float = 0.0,
    blink: float = 0.0,
    n_theta: int = _HEAD_N_THETA,
    n_phi: int = _HEAD_N_PHI,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[int, int, int]]]:
    """A solid, shaded head — a full ellipsoid surface (a real triangle mesh,
    not scattered points, so it reads as a solid volume rather than a
    translucent point cloud) with eyes/mouth as shaded, sunken depressions
    and the nose as a subtle raised ridge, lit by a fixed light direction.

    `mouth_openness` (0..1) deepens/widens the mouth depression; `blink`
    (0..1) fills the eye sockets back in to simulate closed eyelids. Both
    are fed from jarvis.voice.speaking_state.SpeakingState / an idle blink
    timer in jarvis/vision/holographic_web.py.

    Returns (vertices, triangles): vertices are (x, y, z, brightness);
    triangles are index triples into vertices, for GL_TRIANGLES.
    """
    mouth_openness = max(0.0, min(1.0, mouth_openness))
    eye_openness = 1.0 - max(0.0, min(1.0, blink))  # 0 = eyes closed (filled in), 1 = eyes open (sunken)

    vertices: list[tuple[float, float, float, float]] = []
    for j in range(n_phi):
        phi = _HEAD_PHI_MIN + (_HEAD_PHI_MAX - _HEAD_PHI_MIN) * j / (n_phi - 1)
        for i in range(n_theta):
            theta = 2 * math.pi * i / n_theta  # full loop around; i == n_theta wraps back to i == 0
            vertices.append(_head_vertex(theta, phi, mouth_openness, eye_openness))

    triangles: list[tuple[int, int, int]] = []
    for j in range(n_phi - 1):
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            a = j * n_theta + i
            b = j * n_theta + i_next
            c = (j + 1) * n_theta + i_next
            d = (j + 1) * n_theta + i
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    return vertices, triangles


class HolographicScene:
    """Owns the currently displayed object's rotation/zoom/selection state and
    reacts to GestureEvents from GestureEngine."""

    def __init__(self) -> None:
        self._object_index = 0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 1.0
        self.selected = False
        self.mouth_openness = 0.0
        self.blink = 0.0
        self.processing = False

    @property
    def active_object(self) -> str:
        return OBJECTS[self._object_index]

    def handle_event(self, event: GestureEvent) -> None:
        if event.type == "pinch_start":
            self.selected = True
        elif event.type == "pinch_end":
            self.selected = False
        elif event.type in ("move", "drag"):
            self.rotation_y = (self.rotation_y + event.data["dx"] * _ROTATION_SENSITIVITY) % 360
            self.rotation_x = (self.rotation_x + event.data["dy"] * _ROTATION_SENSITIVITY) % 360
        elif event.type == "spread":
            self.zoom = min(_MAX_ZOOM, self.zoom * event.data.get("scale_factor", 1.05))
        elif event.type == "contract":
            self.zoom = max(_MIN_ZOOM, self.zoom * event.data.get("scale_factor", 0.95))
        elif event.type == "swipe_left":
            self.cycle_object(-1)
        elif event.type == "swipe_right":
            self.cycle_object(1)

    def handle_events(self, events: list[GestureEvent]) -> None:
        for event in events:
            self.handle_event(event)

    def cycle_object(self, direction: int = 1) -> None:
        self._object_index = (self._object_index + direction) % len(OBJECTS)

    def set_mouth_openness(self, value: float) -> None:
        self.mouth_openness = max(0.0, min(1.0, value))

    def set_blink(self, value: float) -> None:
        self.blink = max(0.0, min(1.0, value))

    def set_processing(self, value: bool) -> None:
        self.processing = bool(value)

    def lod_level(self) -> int:
        """How 'zoomed into the atoms' the view is, as a discrete point-cloud density level."""
        if self.zoom < 1.5:
            return 0
        if self.zoom < 3.0:
            return 1
        return 2

    def points(self) -> list[tuple[float, float, float]]:
        if self.active_object == "face":
            vertices, _triangles = self._head_mesh()
            return [(x, y, z) for x, y, z, _brightness in vertices]
        if self.active_object == "sphere":
            return generate_sphere_points(n=200 * (self.lod_level() + 1))
        if self.active_object == "cube":
            return generate_cube_wireframe_points()
        return generate_molecule_points(self.lod_level())

    def mesh(self) -> tuple[list[tuple[float, float, float, float]], list[tuple[int, int, int]]] | None:
        """Solid shaded triangle mesh for the current object, if it has one
        (only the face does so far) — None means the renderer should fall
        back to drawing `points()` as a plain, uncolored point cloud."""
        if self.active_object == "face":
            return self._head_mesh()
        return None

    def _head_mesh(self) -> tuple[list[tuple[float, float, float, float]], list[tuple[int, int, int]]]:
        return generate_head_mesh(self.mouth_openness, self.blink)

    def render_state(self) -> dict:
        return {
            "object": self.active_object,
            "rotation": (self.rotation_x, self.rotation_y),
            "zoom": self.zoom,
            "selected": self.selected,
            "lod_level": self.lod_level(),
            "points": self.points(),
            "mesh": self.mesh(),
            "processing": self.processing,
        }
