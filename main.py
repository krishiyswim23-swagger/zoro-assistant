#!/usr/bin/env python3
"""JARVIS entry point.

Stage 1 (typed commands) + Stage 2 (voice) + Stage 3 (AI brain) + Stage 4
(tools) + Stage 5 (memory) all meet here: read a request (typed or spoken),
let the router decide whether it's a known instant action or something for
the AI brain (which reasons, calls tools, and reads/writes memory as
needed), then speak/print the reply.

Stage 6 (vision/hologram) can run standalone via hologram.py, or alongside
this loop with --hologram: the assistant loop runs on a background thread
while the main thread runs the hologram's WebSocket/HTTP server (camera
capture, hand tracking and gesture interpretation stay in Python; the actual
rendering happens in the browser tab it opens), and a threading.Event lets
either side tell the other to shut down.

A shared BrainActivity (jarvis/voice/brain_activity.py) tracks which part of
the pipeline is actually doing something right now (routing, AI reasoning, a
tool call, memory, speech) and streams it to the hologram's neural-activity
visualization; an AgentManager (jarvis/agents/) runs real background tasks
(weather refresh, a memory digest, a system-health check) on their own
schedule, independent of whether anyone is talking to JARVIS, and their live
status feeds the same page's agent dashboard. Both run regardless of
--hologram — the hologram is just what displays them.

Usage:
    python main.py                                    # text mode
    python main.py --voice                            # microphone input
    python main.py --voice --wake-word "hey jarvis"
    python main.py --voice --hologram                 # talk to JARVIS while the hologram runs
    python main.py --voice --hologram --hologram-keyboard-demo  # no camera needed
"""

from __future__ import annotations

import argparse
import threading

from jarvis import config
from jarvis.agents.base import AgentManager
from jarvis.agents.memory_digest import MemoryDigestAgent
from jarvis.agents.system_health import SystemHealthAgent
from jarvis.agents.weather_watch import WeatherWatchAgent
from jarvis.brain.ai_brain import AIBrain
from jarvis.memory.store import MemoryStore
from jarvis.router import ExitRequested, Router
from jarvis.voice.assistant_state import IDLE, LISTENING, TALKING, THINKING, AssistantState
from jarvis.voice.brain_activity import BrainActivity
from jarvis.voice.speaking_state import SpeakingState
from jarvis.voice.speech_output import SpeechOutput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zoro desktop assistant")
    parser.add_argument("--voice", action="store_true", help="Use microphone input instead of typing")
    parser.add_argument(
        "--wake-word",
        default=None,
        help="Only act on utterances that start with this phrase, e.g. 'hey jarvis' (voice mode only)",
    )
    parser.add_argument(
        "--hologram",
        action="store_true",
        help="Also run the Stage 6 holographic interface alongside voice/text",
    )
    parser.add_argument("--hologram-camera", type=int, default=0, help="camera index for the hologram (default 0)")
    parser.add_argument("--hologram-fps", type=int, default=30, help="hologram render frame rate (default 30)")
    parser.add_argument(
        "--hologram-keyboard-demo",
        action="store_true",
        help="drive the hologram with arrow keys/space/tab instead of a camera",
    )
    parser.add_argument(
        "--hologram-style",
        choices=["regions", "humanoid"],
        default="regions",
        help=(
            "which hologram visualization to open: 'regions' (default) is the neural-activity "
            "brain graph; 'humanoid' is a particle humanoid bust with a boot-up sequence and a "
            "LISTENING/THINKING/TALKING status readout"
        ),
    )
    return parser


def get_input(args: argparse.Namespace, speech_input, assistant_state: AssistantState | None = None) -> str:
    if args.voice and speech_input is not None:
        from jarvis.voice.speech_input import VoiceUnavailableError

        if assistant_state is not None:
            assistant_state.set(LISTENING)
        try:
            text = speech_input.listen()
        except VoiceUnavailableError as exc:
            print(f"[voice unavailable: {exc}]")
            return input("You: ")

        print(f"You said: {text}")
        if args.wake_word:
            wake = args.wake_word.lower()
            if not text.lower().startswith(wake):
                return ""  # no wake word, ignore this utterance
            text = text[len(args.wake_word) :].strip(" ,")
        return text

    return input("You: ")


def run_assistant_loop(
    args: argparse.Namespace,
    router: Router,
    speech_output: SpeechOutput,
    speech_input,
    stop_event: threading.Event | None = None,
    assistant_state: AssistantState | None = None,
) -> None:
    """The core request/response loop. Runs standalone, or on a background
    thread alongside the hologram's render loop (see main() below) — either
    way, setting `stop_event` is how the two sides tell each other to stop.

    `assistant_state`, if given, is kept up to date (idle/thinking/talking)
    so the hologram's face can visibly reflect what's actually happening —
    a "thinking" pulse while router.handle() is working (which can take a
    few seconds with tool calls or a rate-limited API), not just a static
    face between replies.
    """
    speech_output.say("Zoro online. How can I help?")

    while stop_event is None or not stop_event.is_set():
        try:
            text = get_input(args, speech_input, assistant_state)
        except (EOFError, KeyboardInterrupt):
            if assistant_state is not None:
                assistant_state.set(TALKING)
            speech_output.say("Goodbye.")
            break

        if not text:
            continue

        if assistant_state is not None:
            assistant_state.set(THINKING)

        try:
            reply = router.handle(text)
        except ExitRequested:
            if assistant_state is not None:
                assistant_state.set(TALKING)
            speech_output.say("Goodbye.")
            break

        if reply:
            if assistant_state is not None:
                assistant_state.set(TALKING)
            speech_output.say(reply)

        if assistant_state is not None:
            assistant_state.set(IDLE)

    if stop_event is not None:
        stop_event.set()


def _run_with_hologram(
    args: argparse.Namespace,
    router: Router,
    speech_output: SpeechOutput,
    speech_input,
    speaking_state: SpeakingState,
    assistant_state: AssistantState,
    brain_activity: BrainActivity,
    agent_manager: AgentManager,
) -> None:
    from jarvis.vision.hand_tracker import VisionUnavailableError
    from jarvis.vision.holographic_web import RendererUnavailableError
    from jarvis.vision.holographic_web import run as run_hologram
    from jarvis.vision.holographic_web import run_keyboard_demo

    stop_event = threading.Event()
    assistant_thread = threading.Thread(
        target=run_assistant_loop,
        args=(args, router, speech_output, speech_input, stop_event, assistant_state),
        daemon=True,
    )
    assistant_thread.start()

    try:
        if args.hologram_keyboard_demo:
            run_keyboard_demo(
                fps=args.hologram_fps,
                stop_event=stop_event,
                speaking_state=speaking_state,
                assistant_state=assistant_state,
                brain_activity=brain_activity,
                agent_manager=agent_manager,
                wake_word=args.wake_word,
                style=args.hologram_style,
            )
        else:
            run_hologram(
                camera_index=args.hologram_camera,
                fps=args.hologram_fps,
                stop_event=stop_event,
                speaking_state=speaking_state,
                assistant_state=assistant_state,
                brain_activity=brain_activity,
                agent_manager=agent_manager,
                wake_word=args.wake_word,
                style=args.hologram_style,
            )
    except (VisionUnavailableError, RendererUnavailableError) as exc:
        print(f"Hologram unavailable ({exc}); continuing in voice/text-only mode.")
        assistant_thread.join()  # the hologram never started, so just let the assistant loop run its course
    else:
        # The broadcast loop returned (Escape pressed in the browser tab) —
        # tell the assistant thread to stop too. It may be mid-listen()/
        # input(), so this is best-effort; the thread is a daemon, so the
        # process still exits cleanly either way once main() returns below.
        stop_event.set()
        assistant_thread.join(timeout=2.0)


def main() -> None:
    args = build_parser().parse_args()

    brain_activity = BrainActivity()
    memory_store = MemoryStore(config.MEMORY_DB_PATH, brain_activity=brain_activity)
    brain = AIBrain(
        config.GEMINI_API_KEY,
        config.JARVIS_MODEL,
        memory_store,
        max_history_turns=config.MAX_HISTORY_TURNS,
        max_tool_iterations=config.MAX_TOOL_ITERATIONS,
        brain_activity=brain_activity,
    )
    router = Router(brain, brain_activity=brain_activity)
    speaking_state = SpeakingState()
    assistant_state = AssistantState()
    speech_output = SpeechOutput(speaking_state=speaking_state, brain_activity=brain_activity)

    speech_input = None
    if args.voice:
        from jarvis.voice.speech_input import SpeechInput, VoiceUnavailableError

        try:
            speech_input = SpeechInput(brain_activity=brain_activity)
        except VoiceUnavailableError as exc:
            print(f"Voice input unavailable ({exc}); falling back to typed input.")

    if not brain.is_available():
        print("Note: no GEMINI_API_KEY set — the AI brain is disabled; only known commands will work.")

    # Real background agents — these run on their own schedule regardless of
    # --hologram; the hologram (if running) just displays their live status.
    agent_manager = AgentManager()
    agent_manager.register(WeatherWatchAgent(), interval_seconds=600)
    agent_manager.register(MemoryDigestAgent(memory_store), interval_seconds=60)
    agent_manager.register(SystemHealthAgent(brain), interval_seconds=30)
    agent_manager.start()

    try:
        if args.hologram:
            _run_with_hologram(
                args, router, speech_output, speech_input, speaking_state, assistant_state, brain_activity, agent_manager
            )
        else:
            run_assistant_loop(args, router, speech_output, speech_input, assistant_state=assistant_state)
    finally:
        agent_manager.stop()


if __name__ == "__main__":
    main()
