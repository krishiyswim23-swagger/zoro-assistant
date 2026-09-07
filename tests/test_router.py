import unittest
from unittest.mock import MagicMock, patch

from jarvis.commands.system_commands import ApplicationNotFoundError
from jarvis.router import ExitRequested, Router


class FakeBrain:
    """Stand-in for AIBrain so router tests don't need an API key or network."""

    def __init__(self) -> None:
        self.reset_called = False
        self.last_prompt = None

    def is_available(self) -> bool:
        return True

    def think(self, text: str) -> str:
        self.last_prompt = text
        return f"brain:{text}"

    def reset(self) -> None:
        self.reset_called = True


class RouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.brain = FakeBrain()
        self.router = Router(self.brain)

    def test_time_intent(self) -> None:
        reply = self.router.handle("what time is it")
        self.assertTrue(reply.startswith("It's "))

    def test_date_intent(self) -> None:
        reply = self.router.handle("what is the date")
        self.assertTrue(reply.startswith("Today is "))

    def test_exit_raises(self) -> None:
        with self.assertRaises(ExitRequested):
            self.router.handle("exit")

    def test_reset_phrase_clears_brain_memory(self) -> None:
        self.router.handle("let's start over")
        self.assertTrue(self.brain.reset_called)

    def test_unknown_request_falls_back_to_brain(self) -> None:
        reply = self.router.handle("why is the sky blue")
        self.assertEqual(reply, "brain:why is the sky blue")

    @patch("jarvis.router.open_application")
    def test_open_dispatches_to_command(self, mock_open) -> None:
        mock_open.return_value = "Opening notepad."
        reply = self.router.handle("open notepad")
        mock_open.assert_called_once_with("notepad")
        self.assertEqual(reply, "Opening notepad.")

    @patch("jarvis.router.open_application")
    def test_open_unknown_app_falls_back_to_brain(self, mock_open) -> None:
        mock_open.side_effect = ApplicationNotFoundError("frobnicator")
        reply = self.router.handle("open frobnicator")
        self.assertIn("brain:", reply)

    @patch("jarvis.router.search_web")
    def test_search_dispatches_to_command(self, mock_search) -> None:
        mock_search.return_value = "Searching the web for cats."
        reply = self.router.handle("search for cats")
        mock_search.assert_called_once_with("cats")
        self.assertEqual(reply, "Searching the web for cats.")

    def test_pulses_association_on_every_request(self) -> None:
        activity = MagicMock()
        router = Router(self.brain, brain_activity=activity)
        router.handle("why is the sky blue")
        activity.pulse.assert_any_call("association")

    def test_pulses_reflex_arc_for_instant_commands_only(self) -> None:
        activity = MagicMock()
        router = Router(self.brain, brain_activity=activity)

        router.handle("what time is it")
        activity.pulse.assert_any_call("reflex_arc")

        activity.reset_mock()
        router.handle("why is the sky blue")  # falls through to the AI brain
        self.assertNotIn(("reflex_arc",), [call.args for call in activity.pulse.call_args_list])

    def test_works_without_a_brain_activity(self) -> None:
        self.router.handle("what time is it")  # must not raise


if __name__ == "__main__":
    unittest.main()
