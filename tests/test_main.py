import argparse
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import main as jarvis_main  # the root-level main.py entry point


class _FakeSpeechOutput:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


class _FakeRouter:
    def __init__(self, replies: dict | None = None, exit_on: set | None = None) -> None:
        self.replies = replies or {}
        self.exit_on = exit_on or set()
        self.handled: list[str] = []

    def handle(self, text: str) -> str:
        self.handled.append(text)
        if text in self.exit_on:
            raise jarvis_main.ExitRequested()
        return self.replies.get(text, "")


def _text_args() -> argparse.Namespace:
    return argparse.Namespace(voice=False, wake_word=None)


def _voice_args(wake_word: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(voice=True, wake_word=wake_word)


class _FakeSpeechInput:
    def __init__(self, text: str = "hello") -> None:
        self.text = text

    def listen(self) -> str:
        return self.text


class _SpyAssistantState(jarvis_main.AssistantState):
    """Records the order states were set in, on top of real get/set behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[str] = []

    def set(self, state: str) -> None:
        self.history.append(state)
        super().set(state)


class RunAssistantLoopTest(unittest.TestCase):
    def test_exit_phrase_breaks_loop_and_sets_stop_event(self) -> None:
        router = _FakeRouter(exit_on={"exit"})
        speech_output = _FakeSpeechOutput()
        stop_event = threading.Event()

        with patch("builtins.input", side_effect=["exit"]):
            jarvis_main.run_assistant_loop(_text_args(), router, speech_output, None, stop_event)

        self.assertTrue(stop_event.is_set())
        self.assertIn("Goodbye.", speech_output.said)

    def test_eof_breaks_loop_and_sets_stop_event(self) -> None:
        router = _FakeRouter()
        speech_output = _FakeSpeechOutput()
        stop_event = threading.Event()

        with patch("builtins.input", side_effect=EOFError):
            jarvis_main.run_assistant_loop(_text_args(), router, speech_output, None, stop_event)

        self.assertTrue(stop_event.is_set())

    def test_already_set_stop_event_prevents_any_input(self) -> None:
        router = _FakeRouter()
        speech_output = _FakeSpeechOutput()
        stop_event = threading.Event()
        stop_event.set()

        with patch("builtins.input") as mock_input:
            jarvis_main.run_assistant_loop(_text_args(), router, speech_output, None, stop_event)
            mock_input.assert_not_called()

    def test_replies_are_spoken(self) -> None:
        router = _FakeRouter(replies={"what time is it": "It's noon.", "exit": ""}, exit_on={"exit"})
        speech_output = _FakeSpeechOutput()
        stop_event = threading.Event()

        with patch("builtins.input", side_effect=["what time is it", "exit"]):
            jarvis_main.run_assistant_loop(_text_args(), router, speech_output, None, stop_event)

        self.assertIn("It's noon.", speech_output.said)

    def test_works_without_a_stop_event(self) -> None:
        router = _FakeRouter(exit_on={"exit"})
        speech_output = _FakeSpeechOutput()

        with patch("builtins.input", side_effect=["exit"]):
            jarvis_main.run_assistant_loop(_text_args(), router, speech_output, None)  # no stop_event at all

        self.assertIn("Goodbye.", speech_output.said)

    def test_assistant_state_goes_thinking_then_talking_then_idle_for_a_reply(self) -> None:
        router = _FakeRouter(replies={"hi": "hello", "exit": ""}, exit_on={"exit"})
        speech_output = _FakeSpeechOutput()
        stop_event = threading.Event()
        assistant_state = _SpyAssistantState()

        with patch("builtins.input", side_effect=["hi", "exit"]):
            jarvis_main.run_assistant_loop(
                _text_args(), router, speech_output, None, stop_event, assistant_state
            )

        thinking_index = assistant_state.history.index(jarvis_main.THINKING)
        talking_index = assistant_state.history.index(jarvis_main.TALKING)
        idle_index = assistant_state.history.index(jarvis_main.IDLE)
        self.assertLess(thinking_index, talking_index)
        self.assertLess(talking_index, idle_index)

    def test_assistant_state_goes_talking_on_exit_phrase(self) -> None:
        router = _FakeRouter(exit_on={"exit"})
        speech_output = _FakeSpeechOutput()
        stop_event = threading.Event()
        assistant_state = _SpyAssistantState()

        with patch("builtins.input", side_effect=["exit"]):
            jarvis_main.run_assistant_loop(
                _text_args(), router, speech_output, None, stop_event, assistant_state
            )

        self.assertIn(jarvis_main.TALKING, assistant_state.history)


class BuildParserTest(unittest.TestCase):
    def test_hologram_style_defaults_to_regions(self) -> None:
        args = jarvis_main.build_parser().parse_args([])
        self.assertEqual(args.hologram_style, "regions")

    def test_hologram_style_accepts_humanoid(self) -> None:
        args = jarvis_main.build_parser().parse_args(["--hologram-style", "humanoid"])
        self.assertEqual(args.hologram_style, "humanoid")

    def test_hologram_style_rejects_unknown_value(self) -> None:
        with self.assertRaises(SystemExit):
            jarvis_main.build_parser().parse_args(["--hologram-style", "bogus"])


class GetInputTest(unittest.TestCase):
    def test_voice_mode_sets_listening_state_before_capturing(self) -> None:
        assistant_state = _SpyAssistantState()
        speech_input = _FakeSpeechInput("what time is it")

        text = jarvis_main.get_input(_voice_args(), speech_input, assistant_state)

        self.assertEqual(text, "what time is it")
        self.assertIn(jarvis_main.LISTENING, assistant_state.history)

    def test_text_mode_does_not_touch_assistant_state(self) -> None:
        assistant_state = _SpyAssistantState()

        with patch("builtins.input", return_value="hi"):
            jarvis_main.get_input(_text_args(), None, assistant_state)

        self.assertEqual(assistant_state.history, [])


if __name__ == "__main__":
    unittest.main()
