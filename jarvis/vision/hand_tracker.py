"""Stage 6: camera capture + hand landmark detection.

Thin wrapper around OpenCV (camera capture) and MediaPipe's Hand Landmarker
task that turns each frame into the plain-data shape gesture_engine expects:
{"hands": [{"handedness": ..., "landmarks": [(x, y, z), ...]}]}.

Targets mediapipe's current "Tasks" API (`mediapipe.tasks.python.vision`),
not the older `mediapipe.solutions.hands` API — that one was removed
starting with mediapipe 0.10.30, and no mediapipe version has both it and
support for recent Python releases, so there's no version pin that keeps
the old API alive. The Tasks API needs a small model file, which is
downloaded once and cached under JARVIS_DATA_DIR.

Both mediapipe and OpenCV are imported lazily so that importing this module
— or the rest of the vision package — doesn't require them to be installed
unless you're actually running the camera pipeline (see requirements-vision.txt).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

from jarvis import config

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = config.DATA_DIR / "models" / "hand_landmarker.task"


class VisionUnavailableError(Exception):
    """Raised when the camera or the hand-tracking model can't be started."""


def _ensure_model_downloaded() -> str:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        except (OSError, urllib.error.URLError) as exc:
            raise VisionUnavailableError(
                f"couldn't download the hand-tracking model from {_MODEL_URL}: {exc}"
            ) from exc
    return str(_MODEL_PATH)


class HandTracker:
    def __init__(
        self,
        camera_index: int = 0,
        max_hands: int = 2,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise VisionUnavailableError(
                "opencv-python and mediapipe are required for vision — "
                "pip install -r requirements-vision.txt"
            ) from exc

        self._cv2 = cv2
        self._mp_image_ctor = mp.Image
        self._mp_image_format = mp.ImageFormat.SRGB
        self._start_time = time.monotonic()

        # Open the camera before touching the (heavier, slower-to-fail)
        # model, so a missing camera never leaves an unclosed landmarker
        # behind for the garbage collector to trip over during teardown.
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise VisionUnavailableError(f"couldn't open camera index {camera_index}")

        try:
            model_path = _ensure_model_downloaded()
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except VisionUnavailableError:
            self._capture.release()
            raise
        except Exception as exc:  # noqa: BLE001 - native lib load/model errors vary by platform
            self._capture.release()
            raise VisionUnavailableError(f"couldn't start the hand-tracking model: {exc}") from exc

    def read_frame(self):
        """Grab one camera frame and return (landmark_frame, bgr_image)."""
        ok, image = self._capture.read()
        if not ok:
            raise VisionUnavailableError("failed to read from camera")

        rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_ctor(image_format=self._mp_image_format, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            label = f"hand_{i}"
            if result.handedness and result.handedness[i]:
                label = result.handedness[i][0].category_name
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            hands.append({"handedness": label, "landmarks": landmarks})

        return {"hands": hands}, image

    def close(self) -> None:
        self._capture.release()
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
