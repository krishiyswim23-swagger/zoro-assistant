import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from jarvis.memory.store import MemoryStore, ProjectNotFoundError


class MemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.mkdtemp()
        self.store = MemoryStore(str(Path(self._tmp_dir) / "memory.db"))
        self.addCleanup(self.store.close)

    def test_remember_and_recall(self) -> None:
        self.store.remember("the user's main project is a batteryless environmental sensor")
        self.store.remember("the user prefers tea over coffee")

        results = self.store.recall("sensor")

        self.assertEqual(len(results), 1)
        self.assertIn("sensor", results[0]["content"])

    def test_recall_no_match_returns_empty(self) -> None:
        self.store.remember("completely unrelated fact")
        self.assertEqual(self.store.recall("quantum physics"), [])

    def test_project_notes_round_trip(self) -> None:
        self.store.add_project_note("sensor research", "found an interesting battery-free approach")
        self.store.add_project_note("sensor research", "compared it against piezoelectric harvesting")

        notes = self.store.get_project_notes("sensor research")

        self.assertEqual(len(notes), 2)
        self.assertIn("piezoelectric", notes[1]["content"])
        self.assertIn("sensor research", self.store.list_projects())

    def test_get_notes_for_unknown_project_raises(self) -> None:
        with self.assertRaises(ProjectNotFoundError):
            self.store.get_project_notes("nonexistent project")

    def test_create_project_is_idempotent(self) -> None:
        first_id = self.store.create_project("robotics")
        second_id = self.store.create_project("robotics")
        self.assertEqual(first_id, second_id)

    def test_preferences_round_trip(self) -> None:
        self.assertIsNone(self.store.get_preference("response_style"))
        self.store.set_preference("response_style", "concise")
        self.assertEqual(self.store.get_preference("response_style"), "concise")
        self.store.set_preference("response_style", "detailed")
        self.assertEqual(self.store.get_preference("response_style"), "detailed")

    def test_persists_across_reconnect(self) -> None:
        self.store.remember("persisted fact")
        db_path = str(Path(self._tmp_dir) / "memory.db")
        self.store.close()

        reopened = MemoryStore(db_path)
        self.addCleanup(reopened.close)
        self.assertEqual(len(reopened.recall("persisted")), 1)

    def test_pulses_brain_activity_on_memory_operations(self) -> None:
        activity = MagicMock()
        store = MemoryStore(str(Path(self._tmp_dir) / "memory2.db"), brain_activity=activity)
        self.addCleanup(store.close)

        store.remember("a fact")
        activity.pulse.assert_called_with("memory")

        activity.reset_mock()
        store.recall("fact")
        activity.pulse.assert_called_with("memory")

    def test_works_without_a_brain_activity(self) -> None:
        self.store.remember("no brain activity wired up")  # must not raise

    def test_fact_count(self) -> None:
        self.assertEqual(self.store.fact_count(), 0)
        self.store.remember("first fact")
        self.store.remember("second fact")
        self.assertEqual(self.store.fact_count(), 2)


if __name__ == "__main__":
    unittest.main()
