import unittest

from jarvis.voice.speaking_state import SpeakingState


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class SpeakingStateTest(unittest.TestCase):
    def test_not_speaking_by_default(self) -> None:
        state = SpeakingState(clock=_FakeClock())
        self.assertFalse(state.is_speaking())
        self.assertEqual(state.mouth_openness(), 0.0)

    def test_mouth_stays_shut_when_not_speaking_even_after_a_pulse(self) -> None:
        clock = _FakeClock()
        state = SpeakingState(clock=clock)
        state.pulse()
        self.assertEqual(state.mouth_openness(), 0.0)

    def test_pulse_opens_mouth_and_it_decays_back_to_closed(self) -> None:
        clock = _FakeClock()
        state = SpeakingState(clock=clock)
        state.set_speaking(True)
        state.pulse()

        self.assertEqual(state.mouth_openness(), 1.0)

        clock.now += 0.125  # halfway through the decay window
        self.assertAlmostEqual(state.mouth_openness(), 0.5, places=3)

        clock.now += 0.125  # fully decayed
        self.assertAlmostEqual(state.mouth_openness(), 0.0, places=3)

    def test_second_pulse_reopens_the_mouth(self) -> None:
        clock = _FakeClock()
        state = SpeakingState(clock=clock)
        state.set_speaking(True)
        state.pulse()

        clock.now += 0.25  # fully decayed
        self.assertAlmostEqual(state.mouth_openness(), 0.0, places=3)

        state.pulse()
        self.assertEqual(state.mouth_openness(), 1.0)

    def test_stopping_speech_immediately_closes_the_mouth(self) -> None:
        clock = _FakeClock()
        state = SpeakingState(clock=clock)
        state.set_speaking(True)
        state.pulse()
        self.assertEqual(state.mouth_openness(), 1.0)

        state.set_speaking(False)
        self.assertEqual(state.mouth_openness(), 0.0)

    def test_fallback_wobble_when_speaking_without_any_pulses(self) -> None:
        clock = _FakeClock()
        state = SpeakingState(clock=clock)
        state.set_speaking(True)  # engine that never reports word boundaries

        seen = set()
        for step in range(10):
            clock.now += 0.1
            value = state.mouth_openness()
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            seen.add(round(value, 3))

        self.assertGreater(len(seen), 1)  # it's actually moving, not stuck at one value


if __name__ == "__main__":
    unittest.main()
