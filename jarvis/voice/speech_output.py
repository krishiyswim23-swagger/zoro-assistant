"""Stage 2: text-to-speech output.

Wraps pyttsx3 (which drives Windows SAPI / macOS NSSpeechSynthesizer / espeak on
Linux) behind a small interface that degrades to printing text if no TTS
engine is available in the current environment (e.g. a headless server).

The engine lives on one dedicated worker thread that creates it and runs every
utterance, and `say()` hands text to that thread and waits. This isn't
indirection for its own sake — pyttsx3's drivers are thread-affine, and on
Windows especially: the SAPI5 driver is a COM object, so it must be created and
called on the same thread, and that thread must have called CoInitialize().
Creating the engine on the main thread and then speaking from the assistant
loop's background thread (which is what --hologram does) puts calls across a
COM apartment boundary, where they fail or simply produce no sound.

Optionally reports speaking activity to a jarvis.voice.speaking_state.
SpeakingState (Stage 6) via pyttsx3's utterance/word callbacks, so the
holographic face can animate its mouth in sync with actual speech.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading

logger = logging.getLogger(__name__)


def _install_hint() -> str:
    """Platform-specific advice for getting pyttsx3 to actually make sound."""
    if sys.platform.startswith("linux"):
        return (
            "on Linux pyttsx3 speaks through espeak — install it with "
            "`sudo apt install espeak` (and `libespeak1`), then try again"
        )
    if sys.platform == "darwin":
        return "on macOS this normally works out of the box — check `pip install pyttsx3` succeeded"
    if sys.platform == "win32":
        return (
            "on Windows pyttsx3 speaks through SAPI5 — check `pip install pyttsx3 pywin32` "
            "succeeded, and that Windows has a voice installed "
            "(Settings > Time & Language > Speech)"
        )
    return "check that `pip install pyttsx3` succeeded and that your output device isn't muted"


class SpeechOutput:
    def __init__(self, rate: int = 180, volume: float = 1.0, speaking_state=None, brain_activity=None) -> None:
        self._engine = None
        self._speaking_state = speaking_state
        self._brain_activity = brain_activity
        self.unavailable_reason: str | None = None

        self._queue: queue.Queue = queue.Queue()
        self._started = threading.Event()
        self._worker = threading.Thread(
            target=self._run_worker, args=(rate, volume), name="jarvis-tts", daemon=True
        )
        self._worker.start()
        self._started.wait()  # engine setup happens on the worker; report its result synchronously

    # -- worker thread ------------------------------------------------------

    def _run_worker(self, rate: int, volume: float) -> None:
        """Owns the engine for its entire lifetime: creates it, then speaks
        every queued utterance on this same thread."""
        self._engine = self._create_engine(rate, volume)
        self._started.set()

        if self._engine is None:
            self._drain_queue()
            return

        while True:
            text, done = self._queue.get()
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:  # pragma: no cover - environment dependent
                self._report_failure("Text-to-speech playback failed", exc)
            finally:
                done.set()

    def _create_engine(self, rate: int, volume: float):
        try:
            if sys.platform == "win32":  # pragma: no cover - Windows only
                # SAPI5 is COM; this thread needs its own apartment before
                # pyttsx3 can build a driver on it.
                try:
                    import pythoncom

                    pythoncom.CoInitialize()
                except ImportError:
                    logger.info("pywin32 not installed; relying on pyttsx3 to initialize COM itself.")

            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            self._connect_speaking_state(engine)
            return engine
        except Exception as exc:  # pragma: no cover - environment dependent
            # Loud on purpose: a silent fallback here is indistinguishable from
            # "JARVIS is broken" — you just never hear a word and nothing says why.
            self._report_failure("Text-to-speech is unavailable", exc)
            return None

    def _report_failure(self, headline: str, exc: Exception) -> None:
        if self.unavailable_reason is not None:
            # A broken driver fails on every utterance; say it once.
            logger.warning("%s; continuing in text mode.", headline, exc_info=True)
            return
        self.unavailable_reason = f"{type(exc).__name__}: {exc}"
        print(
            f"[!] {headline} ({self.unavailable_reason}) — "
            f"Zoro will print replies instead of speaking them.\n"
            f"    To fix: {_install_hint()}."
        )
        logger.warning("%s; continuing in text mode.", headline, exc_info=True)

    def _drain_queue(self) -> None:
        """Release anyone blocked in say() once we know nothing will be spoken."""
        while True:
            _text, done = self._queue.get()
            done.set()

    def _connect_speaking_state(self, engine) -> None:
        # Word-boundary pulses are a nice-to-have for tighter lip-sync timing;
        # say() below sets speaking True/False directly around the utterance,
        # so the face still animates (via SpeakingState's fallback wobble) even
        # on a driver where this connect() doesn't work at all.
        if self._speaking_state is None:
            return
        try:
            engine.connect("started-word", lambda name, location, length: self._speaking_state.pulse())
        except Exception:  # pragma: no cover - driver-dependent
            logger.info("This TTS driver doesn't support word-boundary callbacks; face will use a generic wobble.")

    # -- public API ---------------------------------------------------------

    def is_available(self) -> bool:
        """True if replies will actually be spoken aloud, not just printed."""
        return self._engine is not None

    def say(self, text: str) -> None:
        print(f"Zoro: {text}")
        if self._brain_activity is not None:
            self._brain_activity.pulse("language")
        if self._engine is None:
            return

        done = threading.Event()
        if self._speaking_state is not None:
            self._speaking_state.set_speaking(True)
        try:
            self._queue.put((text, done))
            done.wait()
        finally:
            if self._speaking_state is not None:
                self._speaking_state.set_speaking(False)


def _self_test() -> None:
    """`python -m jarvis.voice.speech_output` — check audio on its own, without
    starting the rest of JARVIS."""
    output = SpeechOutput()
    if not output.is_available():
        raise SystemExit(1)
    print("TTS engine started. Speaking a test phrase — you should hear this.")
    output.say("Zoro audio check. If you can hear this, speech output is working.")
    print("Done. If you heard nothing, check your output device and system volume.")


if __name__ == "__main__":
    _self_test()
