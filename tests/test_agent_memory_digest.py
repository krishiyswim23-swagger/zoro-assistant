import tempfile
import unittest
from pathlib import Path

from jarvis.agents.memory_digest import MemoryDigestAgent
from jarvis.memory.store import MemoryStore


class MemoryDigestAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        self.store = MemoryStore(str(Path(tmp_dir) / "memory.db"))
        self.addCleanup(self.store.close)
        self.agent = MemoryDigestAgent(self.store)

    def test_reports_zero_facts_and_no_projects_initially(self) -> None:
        result = self.agent.run_once()
        self.assertIn("0 fact", result)
        self.assertIn("no projects", result)

    def test_reports_updated_counts(self) -> None:
        self.store.remember("fact one")
        self.store.remember("fact two")
        self.store.create_project("robotics")

        result = self.agent.run_once()

        self.assertIn("2 fact", result)
        self.assertIn("1 project", result)

    def test_has_stable_id_and_name(self) -> None:
        self.assertEqual(self.agent.id, "memory_digest")
        self.assertEqual(self.agent.name, "Memory Digest")


if __name__ == "__main__":
    unittest.main()
