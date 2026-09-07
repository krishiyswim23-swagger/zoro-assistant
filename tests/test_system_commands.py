import unittest

from jarvis.commands import system_commands


class SystemCommandsTest(unittest.TestCase):
    def test_get_time_format(self) -> None:
        result = system_commands.get_time()
        self.assertRegex(result, r"^\d{1,2}:\d{2} [AP]M$")

    def test_get_date_format(self) -> None:
        result = system_commands.get_date()
        self.assertRegex(result, r"^[A-Za-z]+, [A-Za-z]+ \d{1,2}, \d{4}$")

    def test_open_unknown_app_raises(self) -> None:
        with self.assertRaises(system_commands.ApplicationNotFoundError):
            system_commands.open_application("definitely-not-a-real-app-xyz")


if __name__ == "__main__":
    unittest.main()
