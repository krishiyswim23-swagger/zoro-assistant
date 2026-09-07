import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from google.genai import errors

from jarvis.brain.ai_brain import AIBrain, BrainError, BrainUnavailableError
from jarvis.memory.store import MemoryStore


class _FakePart:
    def __init__(self, text: str | None = None) -> None:
        self.text = text


class _FakeContent:
    def __init__(self, parts: list) -> None:
        self.parts = parts


class _FakeCandidate:
    def __init__(self, content: _FakeContent) -> None:
        self.content = content


class _FakeFunctionCall:
    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.args = args


class _FakeResponse:
    def __init__(self, parts: list, function_calls: list | None = None) -> None:
        self.candidates = [_FakeCandidate(_FakeContent(parts))]
        self.function_calls = function_calls or []


class _FakeModels:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.models = _FakeModels(responses)


def _make_brain(**overrides) -> tuple[AIBrain, MemoryStore, str]:
    tmp_dir = tempfile.mkdtemp()
    db_path = str(Path(tmp_dir) / "memory.db")
    store = MemoryStore(db_path)
    kwargs = {"api_key": "", "model": "gemini-3.6-flash", "memory_store": store}
    kwargs.update(overrides)
    brain = AIBrain(**kwargs)
    return brain, store, tmp_dir


class AIBrainTest(unittest.TestCase):
    def test_unavailable_without_api_key(self) -> None:
        brain, store, _ = _make_brain()
        self.addCleanup(store.close)
        self.assertFalse(brain.is_available())
        with self.assertRaises(BrainUnavailableError):
            brain.think("hello")

    def test_reset_clears_turns(self) -> None:
        brain, store, _ = _make_brain()
        self.addCleanup(store.close)
        brain._turns = [[{"role": "user", "content": "hi"}]]
        brain.reset()
        self.assertEqual(brain._turns, [])

    def test_trim_turns_keeps_most_recent(self) -> None:
        brain, store, _ = _make_brain(max_history_turns=2)
        self.addCleanup(store.close)
        brain._turns = [[{"content": str(i)}] for i in range(5)]
        brain._trim_turns()
        self.assertEqual(len(brain._turns), 2)
        self.assertEqual(brain._turns[-1][0]["content"], "4")

    def test_think_runs_tool_then_returns_final_answer(self) -> None:
        brain, store, _ = _make_brain()
        self.addCleanup(store.close)

        tool_call_response = _FakeResponse(
            parts=[_FakePart()],
            function_calls=[_FakeFunctionCall("calculate", {"expression": "2 + 2"})],
        )
        final_response = _FakeResponse(parts=[_FakePart(text="The answer is 4.")])
        brain._client = _FakeClient([tool_call_response, final_response])

        reply = brain.think("what's 2+2?")

        self.assertEqual(reply, "The answer is 4.")
        turn = brain._turns[-1]
        # user request, model tool-call turn, tool-response turn, model final turn
        self.assertEqual(len(turn), 4)
        tool_response_content = turn[2]
        self.assertEqual(tool_response_content.parts[0].function_response.response["result"], "4")

    def test_think_reports_tool_errors_without_crashing(self) -> None:
        brain, store, _ = _make_brain()
        self.addCleanup(store.close)

        tool_call_response = _FakeResponse(
            parts=[_FakePart()],
            function_calls=[_FakeFunctionCall("calculate", {"expression": "not math"})],
        )
        final_response = _FakeResponse(
            parts=[_FakePart(text="That expression didn't parse, could you rephrase it?")]
        )
        brain._client = _FakeClient([tool_call_response, final_response])

        reply = brain.think("calculate 'not math'")

        self.assertIn("didn't parse", reply)
        tool_response_content = brain._turns[-1][2]
        self.assertIn("error", tool_response_content.parts[0].function_response.response)

    def test_think_stops_after_max_tool_iterations(self) -> None:
        brain, store, _ = _make_brain(max_tool_iterations=2)
        self.addCleanup(store.close)

        looping_response = _FakeResponse(
            parts=[_FakePart()],
            function_calls=[_FakeFunctionCall("calculate", {"expression": "1 + 1"})],
        )
        brain._client = _FakeClient([looping_response, looping_response])

        reply = brain.think("keep calculating forever")

        self.assertIn("step limit", reply)

    def test_rate_limit_error_gets_a_friendly_message(self) -> None:
        brain, store, _ = _make_brain()
        self.addCleanup(store.close)
        exc = errors.ClientError(
            429, {"error": {"message": "You exceeded your quota.", "status": "RESOURCE_EXHAUSTED"}}
        )
        self.assertIn("rate limit", brain._friendly_api_error(exc))
        self.assertNotIn("RESOURCE_EXHAUSTED", brain._friendly_api_error(exc))  # no raw JSON dumped

    def test_model_not_found_error_gets_a_friendly_message(self) -> None:
        brain, store, _ = _make_brain(model="not-a-real-model")
        self.addCleanup(store.close)
        exc = errors.ClientError(
            404, {"error": {"message": "model not found", "status": "NOT_FOUND"}}
        )
        self.assertIn("not-a-real-model", brain._friendly_api_error(exc))

    def test_other_api_errors_show_just_the_message(self) -> None:
        brain, store, _ = _make_brain()
        self.addCleanup(store.close)
        exc = errors.ServerError(
            500, {"error": {"message": "internal error", "status": "INTERNAL"}}
        )
        friendly = brain._friendly_api_error(exc)
        self.assertIn("internal error", friendly)
        self.assertNotIn("INTERNAL", friendly)

    def test_think_wraps_api_errors_as_brain_error(self) -> None:
        brain, store, _ = _make_brain()
        self.addCleanup(store.close)

        class _RaisingModels:
            def generate_content(self, **kwargs):
                raise errors.ClientError(
                    429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}
                )

        class _RaisingClient:
            models = _RaisingModels()

        brain._client = _RaisingClient()

        with self.assertRaises(BrainError) as ctx:
            brain.think("hi")
        self.assertIn("rate limit", str(ctx.exception))

    def test_think_pulses_prefrontal_and_tool_call_pulses_motor(self) -> None:
        activity = MagicMock()
        brain, store, _ = _make_brain(brain_activity=activity)
        self.addCleanup(store.close)

        tool_call_response = _FakeResponse(
            parts=[_FakePart()],
            function_calls=[_FakeFunctionCall("calculate", {"expression": "2 + 2"})],
        )
        final_response = _FakeResponse(parts=[_FakePart(text="4.")])
        brain._client = _FakeClient([tool_call_response, final_response])

        brain.think("what's 2+2?")

        activity.pulse.assert_any_call("prefrontal")
        activity.pulse.assert_any_call("motor")

    def test_weather_tool_call_also_pulses_predictive(self) -> None:
        activity = MagicMock()
        brain, store, _ = _make_brain(brain_activity=activity)
        self.addCleanup(store.close)

        tool_call_response = _FakeResponse(
            parts=[_FakePart()],
            function_calls=[_FakeFunctionCall("get_weather", {"location": "current"})],
        )
        final_response = _FakeResponse(parts=[_FakePart(text="It's sunny.")])
        brain._client = _FakeClient([tool_call_response, final_response])

        brain.think("what's the weather?")

        activity.pulse.assert_any_call("predictive")


if __name__ == "__main__":
    unittest.main()
