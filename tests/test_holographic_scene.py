import unittest

from jarvis.vision.gesture_engine import GestureEvent
from jarvis.vision.holographic_scene import (
    _HEAD_N_PHI,
    _HEAD_N_THETA,
    _HEAD_RZ,
    OBJECTS,
    HolographicScene,
    generate_cube_wireframe_points,
    generate_head_mesh,
    generate_molecule_points,
    generate_sphere_points,
)


class SphereGenerationTest(unittest.TestCase):
    def test_point_count_matches_request(self) -> None:
        self.assertEqual(len(generate_sphere_points(n=100)), 100)

    def test_points_lie_on_the_requested_radius(self) -> None:
        radius = 2.0
        for x, y, z in generate_sphere_points(n=50, radius=radius):
            self.assertAlmostEqual((x**2 + y**2 + z**2) ** 0.5, radius, places=5)

    def test_empty_for_zero_points(self) -> None:
        self.assertEqual(generate_sphere_points(n=0), [])


class CubeGenerationTest(unittest.TestCase):
    def test_all_points_within_half_extent(self) -> None:
        half = 0.5
        for x, y, z in generate_cube_wireframe_points(size=1.0):
            for coord in (x, y, z):
                self.assertLessEqual(abs(coord), half + 1e-9)


class MoleculeGenerationTest(unittest.TestCase):
    def test_deeper_lod_adds_more_points(self) -> None:
        level0 = generate_molecule_points(0)
        level1 = generate_molecule_points(1)
        level2 = generate_molecule_points(2)
        self.assertLess(len(level0), len(level1))
        self.assertLess(len(level1), len(level2))

    def test_level0_is_just_the_atom_centers(self) -> None:
        atoms = ((0, 0, 0), (1, 1, 1))
        self.assertEqual(generate_molecule_points(0, atoms=atoms), list(atoms))


def _vertices(mouth_openness: float = 0.0, blink: float = 0.0):
    vertices, _triangles = generate_head_mesh(mouth_openness=mouth_openness, blink=blink)
    return vertices


def _eye_and_mouth_vertices(vertices):
    """Frontal vertices near an eye or mouth center, where socket-darkening applies."""
    from jarvis.vision.holographic_scene import _EYE_CENTERS, _MOUTH_CENTER_Y

    near = []
    for x, y, z, b in vertices:
        if z <= 0.3:
            continue
        if any(abs(x - ex) < 0.05 and abs(y - ey) < 0.03 for ex, ey in _EYE_CENTERS):
            near.append((x, y, z, b))
        elif abs(x) < 0.05 and abs(y - _MOUTH_CENTER_Y) < 0.03:
            near.append((x, y, z, b))
    return near


class HeadMeshGenerationTest(unittest.TestCase):
    def test_returns_the_expected_vertex_and_triangle_counts(self) -> None:
        vertices, triangles = generate_head_mesh()
        self.assertEqual(len(vertices), _HEAD_N_THETA * _HEAD_N_PHI)
        self.assertEqual(len(triangles), _HEAD_N_THETA * (_HEAD_N_PHI - 1) * 2)

    def test_triangle_indices_are_all_valid(self) -> None:
        vertices, triangles = generate_head_mesh()
        for a, b, c in triangles:
            for index in (a, b, c):
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(vertices))

    def test_brightness_is_within_valid_range(self) -> None:
        for _x, _y, _z, brightness in _vertices():
            self.assertGreaterEqual(brightness, 0.0)
            self.assertLessEqual(brightness, 1.0)

    def test_nose_ridge_pushes_a_vertex_beyond_the_base_head_depth(self) -> None:
        max_z = max(v[2] for v in _vertices())
        self.assertGreater(max_z, _HEAD_RZ)

    def test_eye_and_mouth_vertices_are_darker_than_plain_skin(self) -> None:
        vertices = _vertices(mouth_openness=0.0, blink=0.0)
        feature_brightness = [b for _x, _y, _z, b in _eye_and_mouth_vertices(vertices)]
        # a "cheek" spot clear of the eyes/nose/mouth bands, but still frontal
        cheek_target = (0.55, 0.0)
        cheek_vertex = min(
            (v for v in vertices if v[2] > 0.3),
            key=lambda v: (v[0] - cheek_target[0]) ** 2 + (v[1] - cheek_target[1]) ** 2,
        )
        self.assertTrue(feature_brightness)
        self.assertLess(max(feature_brightness), cheek_vertex[3])

    def test_opening_the_mouth_sinks_it_in_further(self) -> None:
        from jarvis.vision.holographic_scene import _MOUTH_CENTER_Y

        def mouth_center_z(mouth_openness: float) -> float:
            vertices = _vertices(mouth_openness=mouth_openness)
            candidates = [(y, z) for x, y, z in ((v[0], v[1], v[2]) for v in vertices) if abs(x) < 0.02]
            closest = min(candidates, key=lambda p: abs(p[0] - _MOUTH_CENTER_Y))
            return closest[1]

        self.assertLess(mouth_center_z(1.0), mouth_center_z(0.0))

    def test_blinking_fills_the_eye_sockets_back_in(self) -> None:
        vertices_open = _vertices(blink=0.0)
        vertices_closed = _vertices(blink=1.0)
        eye_open = [b for _x, _y, _z, b in _eye_and_mouth_vertices(vertices_open)]
        eye_closed = [b for _x, _y, _z, b in _eye_and_mouth_vertices(vertices_closed)]
        self.assertLess(sum(eye_open) / len(eye_open), sum(eye_closed) / len(eye_closed))

    def test_out_of_range_inputs_are_clamped(self) -> None:
        self.assertEqual(generate_head_mesh(mouth_openness=-1.0), generate_head_mesh(mouth_openness=0.0))
        self.assertEqual(generate_head_mesh(mouth_openness=2.0), generate_head_mesh(mouth_openness=1.0))
        self.assertEqual(generate_head_mesh(blink=-1.0), generate_head_mesh(blink=0.0))
        self.assertEqual(generate_head_mesh(blink=2.0), generate_head_mesh(blink=1.0))


class HolographicSceneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = HolographicScene()

    def test_starts_on_first_object_unselected(self) -> None:
        self.assertEqual(self.scene.active_object, OBJECTS[0])
        self.assertFalse(self.scene.selected)

    def test_starts_showing_the_face_with_a_closed_mouth(self) -> None:
        self.assertEqual(self.scene.active_object, "face")
        self.assertEqual(self.scene.mouth_openness, 0.0)

    def test_set_mouth_openness_clamps_to_valid_range(self) -> None:
        self.scene.set_mouth_openness(0.5)
        self.assertEqual(self.scene.mouth_openness, 0.5)
        self.scene.set_mouth_openness(2.0)
        self.assertEqual(self.scene.mouth_openness, 1.0)
        self.scene.set_mouth_openness(-1.0)
        self.assertEqual(self.scene.mouth_openness, 0.0)

    def test_set_blink_clamps_to_valid_range(self) -> None:
        self.scene.set_blink(0.5)
        self.assertEqual(self.scene.blink, 0.5)
        self.scene.set_blink(2.0)
        self.assertEqual(self.scene.blink, 1.0)
        self.scene.set_blink(-1.0)
        self.assertEqual(self.scene.blink, 0.0)

    def test_points_reflect_mouth_openness_while_on_the_face(self) -> None:
        self.scene.set_mouth_openness(0.0)
        closed_points = self.scene.points()
        self.scene.set_mouth_openness(1.0)
        open_points = self.scene.points()
        self.assertNotEqual(closed_points, open_points)

    def test_pinch_toggles_selection(self) -> None:
        self.scene.handle_event(GestureEvent("pinch_start", {"hand": "Right"}))
        self.assertTrue(self.scene.selected)
        self.scene.handle_event(GestureEvent("pinch_end", {"hand": "Right"}))
        self.assertFalse(self.scene.selected)

    def test_move_rotates_the_scene(self) -> None:
        self.scene.handle_event(GestureEvent("move", {"hand": "Right", "dx": 0.1, "dy": -0.05}))
        self.assertNotEqual(self.scene.rotation_y, 0.0)
        self.assertNotEqual(self.scene.rotation_x, 0.0)

    def test_spread_increases_zoom_and_contract_decreases_it(self) -> None:
        self.scene.handle_event(GestureEvent("spread", {"scale_factor": 2.0}))
        self.assertGreater(self.scene.zoom, 1.0)

        zoom_after_spread = self.scene.zoom
        self.scene.handle_event(GestureEvent("contract", {"scale_factor": 0.5}))
        self.assertLess(self.scene.zoom, zoom_after_spread)

    def test_zoom_is_clamped(self) -> None:
        for _ in range(20):
            self.scene.handle_event(GestureEvent("spread", {"scale_factor": 3.0}))
        self.assertLessEqual(self.scene.zoom, 6.0)

        for _ in range(20):
            self.scene.handle_event(GestureEvent("contract", {"scale_factor": 0.1}))
        self.assertGreaterEqual(self.scene.zoom, 0.3)

    def test_swipes_cycle_through_objects_and_wrap_around(self) -> None:
        seen = [self.scene.active_object]
        for _ in range(len(OBJECTS)):
            self.scene.handle_event(GestureEvent("swipe_right", {}))
            seen.append(self.scene.active_object)
        self.assertEqual(seen[0], seen[-1])  # wrapped back to the start
        self.assertEqual(len(set(seen[:-1])), len(OBJECTS))  # visited every object once

    def test_lod_level_rises_with_zoom(self) -> None:
        self.assertEqual(self.scene.lod_level(), 0)
        self.scene.zoom = 2.0
        self.assertEqual(self.scene.lod_level(), 1)
        self.scene.zoom = 4.0
        self.assertEqual(self.scene.lod_level(), 2)

    def test_render_state_shape(self) -> None:
        state = self.scene.render_state()
        self.assertEqual(
            set(state.keys()),
            {"object", "rotation", "zoom", "selected", "lod_level", "points", "mesh", "processing"},
        )
        self.assertIsInstance(state["points"], list)

    def test_set_processing_updates_render_state(self) -> None:
        self.assertFalse(self.scene.render_state()["processing"])
        self.scene.set_processing(True)
        self.assertTrue(self.scene.render_state()["processing"])

    def test_face_has_a_mesh_but_other_objects_dont(self) -> None:
        self.assertEqual(self.scene.active_object, "face")
        self.assertIsNotNone(self.scene.mesh())

        self.scene.cycle_object(1)
        self.assertNotEqual(self.scene.active_object, "face")
        self.assertIsNone(self.scene.mesh())

    def test_blink_changes_the_face_mesh(self) -> None:
        self.scene.set_blink(0.0)
        eyes_open_vertices, _ = self.scene.mesh()
        self.scene.set_blink(1.0)
        eyes_closed_vertices, _ = self.scene.mesh()
        self.assertNotEqual(eyes_open_vertices, eyes_closed_vertices)


if __name__ == "__main__":
    unittest.main()
