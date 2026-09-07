import unittest

from jarvis.vision.gesture_engine import GestureEngine

_ZERO = (0.0, 0.0, 0.0)


def _hand(handedness: str, thumb_index_distance: float, palm_x: float = 0.5, palm_y: float = 0.5) -> dict:
    """Build a 21-landmark hand with a controlled thumb/index gap and palm position.

    Only indices 0 (wrist), 4 (thumb tip), 5/9/13/17 (palm proxy), and 8 (index
    tip) matter to gesture_engine; the rest are filler.
    """
    landmarks = [_ZERO] * 21
    landmarks[0] = (palm_x, palm_y, 0.0)
    for i in (5, 9, 13, 17):
        landmarks[i] = (palm_x, palm_y, 0.0)
    landmarks[4] = (palm_x, palm_y, 0.0)
    landmarks[8] = (palm_x + thumb_index_distance, palm_y, 0.0)
    return {"handedness": handedness, "landmarks": landmarks}


class GestureEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = GestureEngine()

    def test_pinch_start_and_end_with_hysteresis(self) -> None:
        events = self.engine.process_frame({"hands": [_hand("Right", 0.2)]}, timestamp=0.0)
        self.assertEqual(events, [])

        events = self.engine.process_frame({"hands": [_hand("Right", 0.02)]}, timestamp=0.1)
        self.assertEqual([e.type for e in events], ["pinch_start"])

        # small wiggle within the hysteresis band shouldn't re-trigger anything
        events = self.engine.process_frame({"hands": [_hand("Right", 0.06)]}, timestamp=0.2)
        self.assertEqual(events, [])

        events = self.engine.process_frame({"hands": [_hand("Right", 0.09)]}, timestamp=0.3)
        self.assertEqual([e.type for e in events], ["pinch_end"])

    def test_move_event_reports_delta_when_not_pinching(self) -> None:
        self.engine.process_frame({"hands": [_hand("Right", 0.2, palm_x=0.5)]}, timestamp=0.0)
        events = self.engine.process_frame({"hands": [_hand("Right", 0.2, palm_x=0.6)]}, timestamp=0.1)

        move_events = [e for e in events if e.type == "move"]
        self.assertEqual(len(move_events), 1)
        self.assertAlmostEqual(move_events[0].data["dx"], 0.1, places=5)

    def test_drag_event_when_pinching_and_moving(self) -> None:
        self.engine.process_frame({"hands": [_hand("Right", 0.02, palm_x=0.5)]}, timestamp=0.0)
        events = self.engine.process_frame({"hands": [_hand("Right", 0.02, palm_x=0.55)]}, timestamp=0.1)

        types = [e.type for e in events]
        self.assertIn("drag", types)
        self.assertNotIn("move", types)

    def test_fast_horizontal_move_triggers_swipe_right(self) -> None:
        self.engine.process_frame({"hands": [_hand("Right", 0.2, palm_x=0.1)]}, timestamp=0.0)
        events = self.engine.process_frame({"hands": [_hand("Right", 0.2, palm_x=0.9)]}, timestamp=0.1)

        self.assertIn("swipe_right", [e.type for e in events])

    def test_swipe_respects_cooldown(self) -> None:
        self.engine.process_frame({"hands": [_hand("Right", 0.2, palm_x=0.1)]}, timestamp=0.0)
        first = self.engine.process_frame({"hands": [_hand("Right", 0.2, palm_x=0.9)]}, timestamp=0.1)
        self.assertIn("swipe_right", [e.type for e in first])

        second = self.engine.process_frame({"hands": [_hand("Right", 0.2, palm_x=0.1)]}, timestamp=0.15)
        self.assertNotIn("swipe_left", [e.type for e in second])

    def test_two_hands_spreading_apart_emits_spread(self) -> None:
        hands = [_hand("Left", 0.2, palm_x=0.4), _hand("Right", 0.2, palm_x=0.6)]
        self.engine.process_frame({"hands": hands}, timestamp=0.0)

        hands2 = [_hand("Left", 0.2, palm_x=0.2), _hand("Right", 0.2, palm_x=0.8)]
        events = self.engine.process_frame({"hands": hands2}, timestamp=0.1)

        self.assertIn("spread", [e.type for e in events])

    def test_two_hands_coming_together_emits_contract(self) -> None:
        hands = [_hand("Left", 0.2, palm_x=0.1), _hand("Right", 0.2, palm_x=0.9)]
        self.engine.process_frame({"hands": hands}, timestamp=0.0)

        hands2 = [_hand("Left", 0.2, palm_x=0.4), _hand("Right", 0.2, palm_x=0.6)]
        events = self.engine.process_frame({"hands": hands2}, timestamp=0.1)

        self.assertIn("contract", [e.type for e in events])

    def test_losing_a_hand_clears_its_state(self) -> None:
        self.engine.process_frame({"hands": [_hand("Right", 0.02)]}, timestamp=0.0)  # pinching
        self.engine.process_frame({"hands": []}, timestamp=0.1)  # hand disappears

        # same hand reappears not pinching; if state hadn't been cleared this
        # could misfire as a "drag" instead of a fresh "move"
        events = self.engine.process_frame({"hands": [_hand("Right", 0.2, palm_x=0.5)]}, timestamp=0.2)
        self.assertEqual(events, [])  # no previous position yet for this reappearance either


if __name__ == "__main__":
    unittest.main()
