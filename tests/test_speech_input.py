import unittest
from unittest.mock import MagicMock, patch

from jarvis.voice.speech_input import SpeechInput, VoiceUnavailableError


class SpeechInputTest(unittest.TestCase):
    def test_missing_microphone_backend_raises_voice_unavailable_not_crash(self) -> None:
        speech_input = SpeechInput()

        # sr.Microphone() imports pyaudio lazily, at construction time — this
        # simulates that dependency being missing (as it was, for the user,
        # on a fresh Windows/Python 3.14 install with no pyaudio wheel yet).
        with patch.object(speech_input._sr, "Microphone", side_effect=ImportError("No module named 'pyaudio'")):
            with self.assertRaises(VoiceUnavailableError):
                speech_input.listen()

    def test_no_microphone_device_raises_voice_unavailable(self) -> None:
        speech_input = SpeechInput()

        with patch.object(speech_input._sr, "Microphone", side_effect=OSError("no default input device")):
            with self.assertRaises(VoiceUnavailableError):
                speech_input.listen()

    def test_missing_pyaudio_wrapped_as_attribute_error_still_falls_back(self) -> None:
        # SpeechRecognition's own get_pyaudio() catches ModuleNotFoundError and
        # re-raises AttributeError("Could not find PyAudio; check installation")
        # instead — this is the exact shape hit on a real Windows/Python 3.14
        # machine with no pyaudio wheel available yet.
        speech_input = SpeechInput()

        with patch.object(
            speech_input._sr,
            "Microphone",
            side_effect=AttributeError("Could not find PyAudio; check installation"),
        ):
            with self.assertRaises(VoiceUnavailableError):
                speech_input.listen()

    def test_successful_listen_pulses_sensory_brain_activity(self) -> None:
        activity = MagicMock()
        speech_input = SpeechInput(brain_activity=activity)

        fake_source = MagicMock()
        fake_microphone = MagicMock()
        fake_microphone.__enter__ = MagicMock(return_value=fake_source)
        fake_microphone.__exit__ = MagicMock(return_value=False)

        with patch.object(speech_input._sr, "Microphone", return_value=fake_microphone):
            with patch.object(speech_input._recognizer, "adjust_for_ambient_noise"):
                with patch.object(speech_input._recognizer, "listen", return_value=MagicMock()):
                    with patch.object(speech_input._recognizer, "recognize_google", return_value="hello jarvis"):
                        text = speech_input.listen()

        self.assertEqual(text, "hello jarvis")
        activity.pulse.assert_called_once_with("sensory")


if __name__ == "__main__":
    unittest.main()
