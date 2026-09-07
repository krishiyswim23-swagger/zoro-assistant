import unittest

from jarvis.vision.blink_scheduler import BlinkScheduler


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class BlinkSchedulerTest(unittest.TestCase):
    def test_eyes_stay_open_before_the_first_scheduled_blink(self) -> None:
        clock = _FakeClock()
        scheduler = BlinkScheduler(clock=clock, seed=1)
        self.assertEqual(scheduler.blink_amount(), 0.0)

    def _advance_to_frame(self, clock: _FakeClock, scheduler: BlinkScheduler, target_time: float, step: float = 0.01):
        """Poll blink_amount() every `step` seconds up to target_time, like a
        real render loop would, returning the last value read."""
        value = 0.0
        while clock.now < target_time:
            clock.now = min(clock.now + step, target_time)
            value = scheduler.blink_amount()
        return value

    def test_a_full_blink_cycle_rises_then_falls_then_reopens(self) -> None:
        clock = _FakeClock()
        scheduler = BlinkScheduler(clock=clock, seed=1)
        blink_starts_at = scheduler._next_blink_at  # noqa: SLF001

        rising = self._advance_to_frame(clock, scheduler, blink_starts_at + 0.03)
        self.assertGreater(rising, 0.0)

        peak = self._advance_to_frame(clock, scheduler, blink_starts_at + 0.075)
        self.assertGreater(peak, rising)

        reopened = self._advance_to_frame(clock, scheduler, blink_starts_at + 0.2)
        self.assertEqual(reopened, 0.0)

    def test_schedules_another_blink_after_completing_one(self) -> None:
        clock = _FakeClock()
        scheduler = BlinkScheduler(clock=clock, seed=1)

        first_scheduled_at = scheduler._next_blink_at  # noqa: SLF001
        self._advance_to_frame(clock, scheduler, first_scheduled_at + 0.2)  # trigger and complete the blink

        self.assertGreater(scheduler._next_blink_at, first_scheduled_at)  # noqa: SLF001

    def test_deterministic_given_the_same_seed(self) -> None:
        a = BlinkScheduler(clock=_FakeClock(), seed=42)
        b = BlinkScheduler(clock=_FakeClock(), seed=42)
        self.assertEqual(a._next_blink_at, b._next_blink_at)  # noqa: SLF001

    def test_blink_amount_never_exceeds_valid_range(self) -> None:
        clock = _FakeClock()
        scheduler = BlinkScheduler(clock=clock, seed=1)
        for _ in range(200):
            clock.now += 0.05
            value = scheduler.blink_amount()
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
