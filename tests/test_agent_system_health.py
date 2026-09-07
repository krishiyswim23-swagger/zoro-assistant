import unittest
from unittest.mock import MagicMock, patch

from jarvis.agents.system_health import SystemHealthAgent


class SystemHealthAgentTest(unittest.TestCase):
    def test_reports_brain_available(self) -> None:
        brain = MagicMock()
        brain.is_available.return_value = True
        agent = SystemHealthAgent(brain)

        with patch.object(agent, "_check_microphone_library", return_value=True):
            with patch.object(agent, "_check_camera_library", return_value=True):
                result = agent.run_once()

        self.assertIn("brain ok", result)
        self.assertIn("voice libs ok", result)
        self.assertIn("vision libs ok", result)

    def test_reports_brain_unavailable_and_missing_libs(self) -> None:
        brain = MagicMock()
        brain.is_available.return_value = False
        agent = SystemHealthAgent(brain)

        with patch.object(agent, "_check_microphone_library", return_value=False):
            with patch.object(agent, "_check_camera_library", return_value=False):
                result = agent.run_once()

        self.assertIn("no API key", result)
        self.assertIn("voice libs not installed", result)
        self.assertIn("vision libs not installed", result)

    def test_never_opens_a_real_microphone_or_camera_device(self) -> None:
        # These checks must stay import-only — opening a real device could
        # contend with SpeechInput/HandTracker if either has one open on
        # another thread. Verified here by asserting neither underlying
        # library's device-open entry points get called.
        brain = MagicMock()
        brain.is_available.return_value = True
        agent = SystemHealthAgent(brain)

        with patch("speech_recognition.Microphone") as mock_mic, patch("cv2.VideoCapture") as mock_cap:
            agent.run_once()

        mock_mic.assert_not_called()
        mock_cap.assert_not_called()

    def test_has_stable_id_and_name(self) -> None:
        agent = SystemHealthAgent(MagicMock())
        self.assertEqual(agent.id, "system_health")
        self.assertEqual(agent.name, "System Health")


if __name__ == "__main__":
    unittest.main()
