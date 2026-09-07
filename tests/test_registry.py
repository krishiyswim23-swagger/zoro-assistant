import tempfile
import unittest
from pathlib import Path

from jarvis.memory.store import MemoryStore
from jarvis.tools.registry import TOOL_SPECS, build_dispatch


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.mkdtemp()
        self.store = MemoryStore(str(Path(self._tmp_dir) / "memory.db"))
        self.addCleanup(self.store.close)
        self.dispatch = build_dispatch(self.store)

    def test_every_tool_spec_has_a_dispatch_entry(self) -> None:
        spec_names = {spec["name"] for spec in TOOL_SPECS}
        self.assertEqual(spec_names, set(self.dispatch.keys()))

    def test_calculate_dispatch(self) -> None:
        self.assertEqual(self.dispatch["calculate"](expression="3 * 3"), "9")

    def test_memory_tools_share_the_same_store(self) -> None:
        self.dispatch["remember_fact"](content="the sky is blue")
        result = self.dispatch["recall_facts"](query="sky")
        self.assertIn("sky is blue", result)

    def test_project_tools_share_the_same_store(self) -> None:
        self.dispatch["add_project_note"](project="demo", note="first note")
        self.assertIn("demo", self.dispatch["list_projects"]())
        self.assertIn("first note", self.dispatch["get_project_notes"](project="demo"))


if __name__ == "__main__":
    unittest.main()
