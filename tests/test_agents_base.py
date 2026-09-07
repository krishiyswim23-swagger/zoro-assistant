import time
import unittest

from jarvis.agents.base import Agent, AgentManager, COMPLETE, ERROR, IDLE, RUNNING


class _CountingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(agent_id="counter", name="Counter")
        self.calls = 0

    def run_once(self) -> str:
        self.calls += 1
        return f"ran {self.calls} time(s)"


class _FailingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(agent_id="failer", name="Failer")

    def run_once(self) -> str:
        raise RuntimeError("boom")


class AgentManagerTest(unittest.TestCase):
    def test_snapshot_shape_before_starting(self) -> None:
        manager = AgentManager()
        manager.register(_CountingAgent(), interval_seconds=10)
        snapshot = manager.snapshot()
        self.assertEqual(snapshot, [{"id": "counter", "name": "Counter", "status": IDLE, "detail": ""}])

    def test_agent_runs_and_reports_complete(self) -> None:
        agent = _CountingAgent()
        manager = AgentManager()
        manager.register(agent, interval_seconds=10)  # long interval — we only want the first run

        manager.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and agent.calls == 0:
                time.sleep(0.01)
        finally:
            manager.stop(timeout=2.0)

        self.assertGreaterEqual(agent.calls, 1)
        snapshot = manager.snapshot()[0]
        self.assertEqual(snapshot["status"], COMPLETE)
        self.assertIn("ran", snapshot["detail"])

    def test_failing_agent_reports_error_not_crash(self) -> None:
        manager = AgentManager()
        manager.register(_FailingAgent(), interval_seconds=10)

        manager.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and manager.snapshot()[0]["status"] not in (COMPLETE, ERROR):
                time.sleep(0.01)
        finally:
            manager.stop(timeout=2.0)

        snapshot = manager.snapshot()[0]
        self.assertEqual(snapshot["status"], ERROR)
        self.assertIn("boom", snapshot["detail"])

    def test_one_agent_failing_does_not_affect_another(self) -> None:
        counting = _CountingAgent()
        manager = AgentManager()
        manager.register(_FailingAgent(), interval_seconds=10)
        manager.register(counting, interval_seconds=10)

        manager.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and counting.calls == 0:
                time.sleep(0.01)
        finally:
            manager.stop(timeout=2.0)

        self.assertGreaterEqual(counting.calls, 1)

    def test_stop_actually_stops_further_runs(self) -> None:
        agent = _CountingAgent()
        manager = AgentManager()
        manager.register(agent, interval_seconds=0.05)  # short interval, so it would run again quickly if not stopped

        manager.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and agent.calls == 0:
            time.sleep(0.01)
        manager.stop(timeout=2.0)

        calls_at_stop = agent.calls
        time.sleep(0.2)  # long enough for another interval to have elapsed, if it were still running
        self.assertEqual(agent.calls, calls_at_stop)


if __name__ == "__main__":
    unittest.main()
