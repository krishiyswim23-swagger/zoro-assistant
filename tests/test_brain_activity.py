import unittest

from jarvis.voice.brain_activity import BrainActivity, PLANNED_REGIONS, REGIONS


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class BrainActivityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _FakeClock()
        self.activity = BrainActivity(clock=self.clock)

    def test_idle_region_has_zero_level(self) -> None:
        self.assertEqual(self.activity.level("prefrontal"), 0.0)

    def test_pulse_sets_level_to_amount(self) -> None:
        self.activity.pulse("prefrontal", 0.8)
        self.assertAlmostEqual(self.activity.level("prefrontal"), 0.8)

    def test_pulse_defaults_to_full_amount(self) -> None:
        self.activity.pulse("motor")
        self.assertAlmostEqual(self.activity.level("motor"), 1.0)

    def test_level_decays_linearly_to_zero(self) -> None:
        self.activity.pulse("language", 1.0)
        self.clock.now = 0.75  # half of the 1.5s decay window
        self.assertAlmostEqual(self.activity.level("language"), 0.5)
        self.clock.now = 1.5
        self.assertAlmostEqual(self.activity.level("language"), 0.0)
        self.clock.now = 10.0  # long after — stays at zero, doesn't go negative
        self.assertEqual(self.activity.level("language"), 0.0)

    def test_amount_is_clamped_to_valid_range(self) -> None:
        self.activity.pulse("sensory", 5.0)
        self.assertEqual(self.activity.level("sensory"), 1.0)
        self.activity.pulse("sensory", -5.0)
        self.assertEqual(self.activity.level("sensory"), 0.0)

    def test_unknown_region_is_ignored_not_an_error(self) -> None:
        self.activity.pulse("not_a_real_region", 1.0)  # must not raise
        self.assertEqual(self.activity.level("not_a_real_region"), 0.0)

    def test_planned_region_never_lights_up(self) -> None:
        self.assertIn("cerebellum", PLANNED_REGIONS)
        self.activity.pulse("cerebellum", 1.0)
        self.assertEqual(self.activity.level("cerebellum"), 0.0)

    def test_snapshot_covers_every_region(self) -> None:
        self.activity.pulse("memory", 0.6)
        snapshot = self.activity.snapshot()
        self.assertEqual(set(snapshot.keys()), set(REGIONS))
        self.assertAlmostEqual(snapshot["memory"], 0.6)

    def test_pulsing_one_region_does_not_affect_another(self) -> None:
        self.activity.pulse("prefrontal", 1.0)
        self.assertEqual(self.activity.level("motor"), 0.0)


if __name__ == "__main__":
    unittest.main()
