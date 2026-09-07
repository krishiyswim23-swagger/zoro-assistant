import threading
import unittest
from unittest.mock import patch

from jarvis.voice.speaking_state import SpeakingState
from jarvis.voice.speech_output import SpeechOutput


class _FakeEngine:
    def __init__(self) -> None:
        self.said: list[str] = []
        self._word_callback = None

    def setProperty(self, *args, **kwargs) -> None:
        pass

    def connect(self, topic, callback):
        if topic == "started-word":
            self._word_callback = callback
        return {"topic": topic, "cb": callback}

    def say(self, text: str) -> None:
        self.said.append(text)
        if self._word_callback is not None:
            self._word_callback("utterance-1", 0, len(text))  # simulate one word boundary

    def runAndWait(self) -> None:
        pass


class SpeechOutputTest(unittest.TestCase):
    @patch("pyttsx3.init")
    def test_say_toggles_speaking_state_around_playback(self, mock_init) -> None:
        fake_engine = _FakeEngine()
        mock_init.return_value = fake_engine
        speaking_state = SpeakingState()

        seen_while_speaking = {}

        def spy_say(text: str) -> None:
            seen_while_speaking["was_speaking"] = speaking_state.is_speaking()
            fake_engine.said.append(text)

        fake_engine.say = spy_say  # type: ignore[method-assign]

        output = SpeechOutput(speaking_state=speaking_state)
        output.say("hello there")

        self.assertTrue(seen_while_speaking["was_speaking"])
        self.assertFalse(speaking_state.is_speaking())  # cleared after runAndWait() returns

    @patch("pyttsx3.init")
    def test_word_boundary_callback_pulses_speaking_state(self, mock_init) -> None:
        fake_engine = _FakeEngine()
        mock_init.return_value = fake_engine
        speaking_state = SpeakingState()

        pulses = []
        speaking_state.pulse = lambda: pulses.append(True)  # type: ignore[method-assign]

        output = SpeechOutput(speaking_state=speaking_state)
        output.say("hello")

        self.assertTrue(pulses)  # the fake engine's say() simulates a word boundary

    @patch("pyttsx3.init")
    def test_works_without_a_speaking_state(self, mock_init) -> None:
        mock_init.return_value = _FakeEngine()
        output = SpeechOutput()  # no speaking_state at all
        output.say("hello")  # should not raise

    @patch("pyttsx3.init")
    def test_pulses_language_brain_activity_on_every_reply(self, mock_init) -> None:
        from unittest.mock import MagicMock

        mock_init.return_value = _FakeEngine()
        activity = MagicMock()
        output = SpeechOutput(brain_activity=activity)
        output.say("hello")
        activity.pulse.assert_called_once_with("language")

    def test_engine_is_created_and_driven_on_one_non_calling_thread(self) -> None:
        # pyttsx3's drivers are thread-affine (SAPI5 on Windows is a COM object),
        # so creating the engine on one thread and speaking from another is what
        # made JARVIS go silent under --hologram. Everything must land on the
        # same thread, and it must not be the caller's.
        seen: dict[str, int] = {}
        engine = _FakeEngine()

        def record_say(text: str) -> None:
            seen["say"] = threading.get_ident()

        engine.say = record_say  # type: ignore[method-assign]

        def init_on_this_thread():
            seen["init"] = threading.get_ident()
            return engine

        with patch("pyttsx3.init", side_effect=init_on_this_thread):
            output = SpeechOutput()
            output.say("hello")

        self.assertEqual(seen["init"], seen["say"])
        self.assertNotEqual(seen["init"], threading.get_ident())

    @patch("pyttsx3.init")
    def test_available_when_engine_starts(self, mock_init) -> None:
        mock_init.return_value = _FakeEngine()
        output = SpeechOutput()
        self.assertTrue(output.is_available())
        self.assertIsNone(output.unavailable_reason)

    @patch("pyttsx3.init", side_effect=OSError("no espeak"))
    def test_engine_failure_is_reported_not_swallowed(self, _mock_init) -> None:
        with patch("builtins.print") as mock_print:
            output = SpeechOutput()

        self.assertFalse(output.is_available())
        self.assertIn("no espeak", output.unavailable_reason)
        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("Text-to-speech is unavailable", printed)
        self.assertIn("To fix:", printed)

    @patch("pyttsx3.init")
    def test_playback_failure_is_reported_once(self, mock_init) -> None:
        engine = _FakeEngine()
        engine.runAndWait = lambda: (_ for _ in ()).throw(RuntimeError("driver died"))
        mock_init.return_value = engine

        output = SpeechOutput()
        with patch("builtins.print") as mock_print:
            output.say("one")
            output.say("two")

        printed = [str(call.args[0]) for call in mock_print.call_args_list]
        warnings = [line for line in printed if "playback failed" in line]
        self.assertEqual(len(warnings), 1)  # reported on the first failure only


if __name__ == "__main__":
    unittest.main()
