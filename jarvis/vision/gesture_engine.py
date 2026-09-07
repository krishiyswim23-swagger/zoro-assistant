"""Stage 6: turns a stream of hand landmark frames into discrete gesture events.

Deliberately has no dependency on mediapipe, OpenCV, or a camera — it's pure
geometry over plain data, so it can be unit-tested without any hardware and
reused by both the real hand_tracker (camera input) and anything that wants
to feed it synthetic frames (tests, a replay tool, etc.).

A "frame" is: {"hands": [{"handedness": "Left" | "Right", "landmarks": [(x, y, z), ...21 points]}]}
using MediaPipe's Hands landmark layout (0=wrist, 4=thumb tip, 8=index tip,
5/9/13/17=the base knuckles used here as a palm-center proxy), with x/y
normalized to [0, 1] across the camera frame (x increases rightward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_THUMB_TIP = 4
_INDEX_TIP = 8
_PALM_LANDMARKS = (0, 5, 9, 13, 17)


@dataclass(frozen=True)
class GestureEvent:
    type: str
    data: dict = field(default_factory=dict)


def _palm_center(landmarks: list[tuple[float, float, float]]) -> tuple[float, float]:
    xs = [landmarks[i][0] for i in _PALM_LANDMARKS]
    ys = [landmarks[i][1] for i in _PALM_LANDMARKS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class GestureEngine:
    """Feed it landmark frames in order; get back the gesture events they produce.

    Gestures recognized:
        pinch_start / pinch_end  -- thumb tip and index tip of one hand touching
        move                     -- a non-pinching hand's palm moved (dx, dy)
        drag                     -- a pinching hand's palm moved (dx, dy) -- e.g. to move/rotate a selected object
        swipe_left / swipe_right -- a fast horizontal palm movement while not pinching
        spread / contract        -- two hands' palms moving apart / together
    """

    def __init__(
        self,
        pinch_start_threshold: float = 0.05,
        pinch_release_threshold: float = 0.08,
        swipe_velocity_threshold: float = 1.2,
        swipe_cooldown: float = 0.6,
        scale_delta_threshold: float = 0.01,
        move_threshold: float = 0.002,
    ) -> None:
        self._pinch_start_threshold = pinch_start_threshold
        self._pinch_release_threshold = pinch_release_threshold
        self._swipe_velocity_threshold = swipe_velocity_threshold
        self._swipe_cooldown = swipe_cooldown
        self._scale_delta_threshold = scale_delta_threshold
        self._move_threshold = move_threshold

        self._pinching: dict[str, bool] = {}
        self._last_palm: dict[str, tuple[float, float, float]] = {}
        self._last_swipe_time: dict[str, float] = {}
        self._last_two_hand_distance: float | None = None

    def process_frame(self, frame: dict, timestamp: float) -> list[GestureEvent]:
        hands = frame.get("hands", [])
        events: list[GestureEvent] = []
        seen = set()

        for hand in hands:
            handedness = hand["handedness"]
            seen.add(handedness)
            landmarks = hand["landmarks"]
            events.extend(self._process_pinch(handedness, landmarks))
            events.extend(self._process_move_and_swipe(handedness, landmarks, timestamp))

        for stale in set(self._pinching) - seen:
            del self._pinching[stale]
        for stale in set(self._last_palm) - seen:
            del self._last_palm[stale]

        if len(hands) == 2:
            events.extend(self._process_two_hand_scale(hands))
        else:
            self._last_two_hand_distance = None

        return events

    def _process_pinch(self, handedness: str, landmarks: list) -> list[GestureEvent]:
        distance = _distance(landmarks[_THUMB_TIP][:2], landmarks[_INDEX_TIP][:2])
        was_pinching = self._pinching.get(handedness, False)

        if not was_pinching and distance < self._pinch_start_threshold:
            self._pinching[handedness] = True
            return [GestureEvent("pinch_start", {"hand": handedness})]
        if was_pinching and distance > self._pinch_release_threshold:
            self._pinching[handedness] = False
            return [GestureEvent("pinch_end", {"hand": handedness})]
        return []

    def _process_move_and_swipe(self, handedness: str, landmarks: list, timestamp: float) -> list[GestureEvent]:
        center = _palm_center(landmarks)
        previous = self._last_palm.get(handedness)
        self._last_palm[handedness] = (center[0], center[1], timestamp)
        if previous is None:
            return []

        prev_x, prev_y, prev_t = previous
        dt = timestamp - prev_t
        if dt <= 0:
            return []

        dx, dy = center[0] - prev_x, center[1] - prev_y
        if abs(dx) < self._move_threshold and abs(dy) < self._move_threshold:
            return []

        is_pinching = self._pinching.get(handedness, False)
        events = [GestureEvent("drag" if is_pinching else "move", {"hand": handedness, "dx": dx, "dy": dy})]

        velocity_x = dx / dt
        if not is_pinching and abs(velocity_x) > self._swipe_velocity_threshold:
            last_swipe = self._last_swipe_time.get(handedness, float("-inf"))
            if timestamp - last_swipe > self._swipe_cooldown:
                direction = "right" if velocity_x > 0 else "left"
                events.append(GestureEvent(f"swipe_{direction}", {"hand": handedness, "velocity": velocity_x}))
                self._last_swipe_time[handedness] = timestamp

        return events

    def _process_two_hand_scale(self, hands: list) -> list[GestureEvent]:
        centers = [_palm_center(hand["landmarks"]) for hand in hands]
        distance = _distance(centers[0], centers[1])
        events: list[GestureEvent] = []

        if self._last_two_hand_distance is not None:
            delta = distance - self._last_two_hand_distance
            if abs(delta) > self._scale_delta_threshold:
                scale_factor = distance / self._last_two_hand_distance if self._last_two_hand_distance else 1.0
                gesture_type = "spread" if delta > 0 else "contract"
                events.append(GestureEvent(gesture_type, {"distance": distance, "scale_factor": scale_factor}))

        self._last_two_hand_distance = distance
        return events
