import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis import config
from jarvis.tools import calculator, code_tool, file_tools, research
from jarvis.tools.calculator import CalculatorError
from jarvis.tools.file_tools import FileAccessError


class CalculatorTest(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        self.assertEqual(calculator.calculate("2 + 2"), "4")

    def test_functions_and_constants(self) -> None:
        self.assertEqual(calculator.calculate("round(sqrt(16))"), "4")

    def test_rejects_arbitrary_code(self) -> None:
        with self.assertRaises(CalculatorError):
            calculator.calculate("__import__('os').system('echo hi')")

    def test_rejects_bare_name_access(self) -> None:
        with self.assertRaises(CalculatorError):
            calculator.calculate("os")


class FileToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.mkdtemp()
        self._original_root = config.FILES_ROOT
        config.FILES_ROOT = Path(self._tmp_dir)
        (Path(self._tmp_dir) / "notes.txt").write_text("hello world")
        (Path(self._tmp_dir) / "sub").mkdir()
        (Path(self._tmp_dir) / "sub" / "report.txt").write_text("nested file")

    def tearDown(self) -> None:
        config.FILES_ROOT = self._original_root
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_search_files_finds_match_in_root(self) -> None:
        self.assertEqual(file_tools.search_files("notes"), ["notes.txt"])

    def test_search_files_recurses_into_subdirs(self) -> None:
        self.assertEqual(file_tools.search_files("report"), [str(Path("sub") / "report.txt")])

    def test_read_document_returns_contents(self) -> None:
        self.assertEqual(file_tools.read_document("notes.txt"), "hello world")

    def test_read_document_rejects_path_escape(self) -> None:
        with self.assertRaises(FileAccessError):
            file_tools.read_document("../../etc/passwd")

    def test_read_document_missing_file_raises(self) -> None:
        with self.assertRaises(FileAccessError):
            file_tools.read_document("does-not-exist.txt")


class CodeToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original = config.ENABLE_CODE_TOOL

    def tearDown(self) -> None:
        config.ENABLE_CODE_TOOL = self._original

    def test_disabled_by_default_raises(self) -> None:
        config.ENABLE_CODE_TOOL = False
        with self.assertRaises(code_tool.CodeToolError):
            code_tool.run_python("print(1)")

    def test_declined_confirmation_raises(self) -> None:
        config.ENABLE_CODE_TOOL = True
        with self.assertRaises(code_tool.CodeToolDeclined):
            code_tool.run_python("print(1)", confirm=lambda prompt: "n")

    def test_confirmed_execution_returns_output(self) -> None:
        config.ENABLE_CODE_TOOL = True
        output = code_tool.run_python("print('hi')", confirm=lambda prompt: "y")
        self.assertIn("hi", output)
        self.assertIn("exit code: 0", output)


class _FakeHTTPResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


class ResearchTest(unittest.TestCase):
    @patch("requests.get")
    @patch("requests.post")
    def test_web_research_returns_titles_and_excerpts(self, mock_post, mock_get) -> None:
        search_html = """
        <div class="result">
          <a class="result__a" href="https://example.com/a">Example A</a>
          <div class="result__snippet">Snippet A</div>
        </div>
        """
        mock_post.return_value = _FakeHTTPResponse(search_html)
        mock_get.return_value = _FakeHTTPResponse("<html><body><p>Full page text about A.</p></body></html>")

        result = research.web_research("example query", max_results=1)

        self.assertIn("Example A", result)
        self.assertIn("https://example.com/a", result)
        self.assertIn("Full page text about A.", result)

    @patch("requests.post")
    def test_no_results_returns_friendly_message(self, mock_post) -> None:
        mock_post.return_value = _FakeHTTPResponse("<html></html>")
        result = research.web_research("nothing matches this")
        self.assertIn("No web results found", result)


if __name__ == "__main__":
    unittest.main()
