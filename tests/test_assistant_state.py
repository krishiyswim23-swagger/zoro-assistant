import unittest

from jarvis.voice.assistant_state import IDLE, LISTENING, TALKING, THINKING, AssistantState


class AssistantStateTest(unittest.TestCase):
    def test_starts_idle(self) -> None:
        self.assertEqual(AssistantState().get(), IDLE)

    def test_set_and_get_round_trip(self) -> None:
        state = AssistantState()
        state.set(THINKING)
        self.assertEqual(state.get(), THINKING)
        state.set(TALKING)
        self.assertEqual(state.get(), TALKING)
        state.set(LISTENING)
        self.assertEqual(state.get(), LISTENING)
        state.set(IDLE)
        self.assertEqual(state.get(), IDLE)


if __name__ == "__main__":
    unittest.main()
