"""Stage 6: serves jarvis/vision/web/ (the Three.js neural-activity/agent-
dashboard page) over a local HTTP server and streams live state to it over a
WebSocket: jarvis.vision.holographic_scene's rotation/zoom/selection state
(for gesture-driven camera control), jarvis.voice.brain_activity's per-
region firing levels (what JARVIS's pipeline is actually doing right now),
and jarvis.agents' live background-task status.

Python owns camera capture, hand tracking, gesture interpretation, and all
of the above state; hologram.js only draws it. `run_keyboard_demo()`'s
arrow-keys/space/tab/escape controls are the one thing that flows the other
way: the page itself listens for those keys and reports them back over the
same WebSocket as {"type": "key", ...} messages, since there's no local
window for Python to read a keyboard from.
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

_WEB_DIR = Path(__file__).resolve().parent / "web"
_DEFAULT_HTTP_PORT = 8765
_DEFAULT_WS_PORT = 8766


class RendererUnavailableError(Exception):
    """Raised when the `websockets` package isn't installed."""


class _KeyState:
    """Tracks keys the browser reports as held (for continuous controls like
    rotation) plus one-shot "just pressed" edges (for toggle controls like
    space/tab), fed by {"type": "key", ...} WebSocket messages sent from
    hologram.js."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._edges: list[str] = []
        self.escape_requested = False

    def handle_message(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except ValueError:
            return
        if message.get("type") != "key":
            return
        key = message.get("key")
        if not isinstance(key, str):
            return
        down = bool(message.get("down"))
        with self._lock:
            if down:
                if key not in self._held:
                    self._edges.append(key)
                self._held.add(key)
                if key == "Escape":
                    self.escape_requested = True
            else:
                self._held.discard(key)

    def is_held(self, key: str) -> bool:
        with self._lock:
            return key in self._held

    def take_edges(self) -> list[str]:
        with self._lock:
            edges, self._edges = self._edges, []
            return edges


def _start_http_server(port: int) -> ThreadingHTTPServer:
    handler_cls = functools.partial(SimpleHTTPRequestHandler, directory=str(_WEB_DIR))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _status_label(assistant_state, wake_word: str | None) -> str:
    from jarvis.voice.assistant_state import LISTENING, TALKING, THINKING

    current = assistant_state.get() if assistant_state is not None else None
    if current == THINKING:
        return "THINKING..."
    if current == TALKING:
        return "TALKING..."
    if current == LISTENING:
        return f'LISTENING FOR "{wake_word.upper()}"' if wake_word else "LISTENING..."
    if wake_word:
        return f'LISTENING FOR "{wake_word.upper()}"'
    return "ZORO — CONNECTED"


def _assistant_state_raw(assistant_state) -> str:
    from jarvis.voice.assistant_state import IDLE

    return assistant_state.get() if assistant_state is not None else IDLE


def _scene_payload(scene, *, regions: dict, agents: list, status_label: str, assistant_state_raw: str = "idle") -> str:
    state = scene.render_state()
    rotation_x, rotation_y = state["rotation"]
    return json.dumps(
        {
            "rotationX": rotation_x,
            "rotationY": rotation_y,
            "zoom": state["zoom"],
            "selected": state["selected"],
            "mouthOpenness": scene.mouth_openness,
            "blink": scene.blink,
            "processing": state["processing"],
            "regions": regions,
            "agents": agents,
            "statusLabel": status_label,
            "assistantState": assistant_state_raw,
        }
    )


def _page_for_style(style: str) -> str:
    return "humanoid.html" if style == "humanoid" else "index.html"


async def _run_async(
    scene,
    *,
    update_scene: Callable[[], None],
    fps: int,
    stop_event,
    speaking_state,
    assistant_state,
    blink_scheduler,
    key_state: _KeyState,
    http_port: int,
    ws_port: int,
    open_browser: bool,
    brain_activity=None,
    agent_manager=None,
    wake_word: str | None = None,
    on_ready: Callable[[int, int], None] | None = None,
    style: str = "regions",
) -> None:
    import websockets

    from jarvis.voice.assistant_state import THINKING

    connections: set = set()

    async def handler(websocket) -> None:
        connections.add(websocket)
        try:
            async for message in websocket:
                key_state.handle_message(message)
        finally:
            connections.discard(websocket)

    httpd = _start_http_server(http_port)
    actual_http_port = httpd.server_address[1]

    try:
        async with websockets.serve(handler, "127.0.0.1", ws_port) as server:
            actual_ws_port = server.sockets[0].getsockname()[1]
            if on_ready is not None:
                on_ready(actual_http_port, actual_ws_port)
            if open_browser:
                page = _page_for_style(style)
                webbrowser.open(f"http://127.0.0.1:{actual_http_port}/{page}?ws_port={actual_ws_port}")

            frame_period = 1.0 / fps
            loop = asyncio.get_running_loop()
            while (stop_event is None or not stop_event.is_set()) and not key_state.escape_requested:
                await loop.run_in_executor(None, update_scene)
                if speaking_state is not None:
                    scene.set_mouth_openness(speaking_state.mouth_openness())
                if assistant_state is not None:
                    scene.set_processing(assistant_state.get() == THINKING)
                scene.set_blink(blink_scheduler.blink_amount())

                if brain_activity is not None:
                    brain_activity.pulse("brainstem")  # "the process is alive", independent of activity

                if connections:
                    payload = _scene_payload(
                        scene,
                        regions=brain_activity.snapshot() if brain_activity is not None else {},
                        agents=agent_manager.snapshot() if agent_manager is not None else [],
                        status_label=_status_label(assistant_state, wake_word),
                        assistant_state_raw=_assistant_state_raw(assistant_state),
                    )
                    websockets.broadcast(connections, payload)

                await asyncio.sleep(frame_period)
    finally:
        httpd.shutdown()
        httpd.server_close()

    if stop_event is not None:
        stop_event.set()


def run(
    camera_index: int = 0,
    fps: int = 30,
    stop_event=None,
    speaking_state=None,
    assistant_state=None,
    http_port: int = _DEFAULT_HTTP_PORT,
    ws_port: int = _DEFAULT_WS_PORT,
    open_browser: bool = True,
    brain_activity=None,
    agent_manager=None,
    wake_word: str | None = None,
    style: str = "regions",
) -> None:
    """Camera-driven hologram: track hands, turn gestures into scene changes,
    and stream the result to the browser page over a WebSocket.

    `stop_event` (a threading.Event) is optional — pass one when running this
    alongside another loop (see main.py's --hologram flag) so either side can
    ask the other to shut down: this loop exits as soon as it's set (or the
    page sends an Escape key event), and sets it itself when it stops so the
    other loop knows to stop too.

    `speaking_state` (a jarvis.voice.speaking_state.SpeakingState) is also
    optional — pass the same instance main.py's SpeechOutput is using so the
    holographic face's mouth animates while JARVIS is actually talking.

    `assistant_state` (a jarvis.voice.assistant_state.AssistantState) is also
    optional — pass main.py's shared instance so the face pulses visibly
    while the assistant is waiting on the AI brain, not just while talking.

    `brain_activity` (a jarvis.voice.brain_activity.BrainActivity) is also
    optional — pass main.py's shared instance so the neural-activity
    visualization reflects what JARVIS's pipeline is actually doing.

    `agent_manager` (a jarvis.agents.base.AgentManager) is also optional —
    pass main.py's shared instance so the live agent dashboard reflects real
    background-task status.

    `wake_word`, if given, is shown in the "LISTENING FOR ..." status label
    when idle — purely cosmetic here, the actual wake-word gating happens in
    main.py's get_input().

    `style` picks which browser page opens: "regions" (default) is the
    neural-activity brain graph, "humanoid" is the particle humanoid-bust
    visualization. Both read the same broadcast state.
    """
    try:
        import websockets  # noqa: F401 - checked eagerly so this fails the same way before opening the camera
    except ImportError as exc:
        raise RendererUnavailableError(
            "the 'websockets' package is required for the browser hologram — "
            "pip install -r requirements-vision.txt"
        ) from exc

    from jarvis.vision.blink_scheduler import BlinkScheduler
    from jarvis.vision.gesture_engine import GestureEngine
    from jarvis.vision.hand_tracker import HandTracker
    from jarvis.vision.holographic_scene import HolographicScene

    scene = HolographicScene()
    engine = GestureEngine()
    blink_scheduler = BlinkScheduler()
    key_state = _KeyState()

    with HandTracker(camera_index=camera_index) as tracker:

        def update_scene() -> None:
            frame, _image = tracker.read_frame()
            events = engine.process_frame(frame, time.time())
            scene.handle_events(events)

        asyncio.run(
            _run_async(
                scene,
                update_scene=update_scene,
                fps=fps,
                stop_event=stop_event,
                speaking_state=speaking_state,
                assistant_state=assistant_state,
                blink_scheduler=blink_scheduler,
                key_state=key_state,
                http_port=http_port,
                ws_port=ws_port,
                open_browser=open_browser,
                brain_activity=brain_activity,
                agent_manager=agent_manager,
                wake_word=wake_word,
                style=style,
            )
        )


def run_keyboard_demo(
    fps: int = 30,
    stop_event=None,
    speaking_state=None,
    assistant_state=None,
    http_port: int = _DEFAULT_HTTP_PORT,
    ws_port: int = _DEFAULT_WS_PORT,
    open_browser: bool = True,
    brain_activity=None,
    agent_manager=None,
    wake_word: str | None = None,
    style: str = "regions",
) -> None:
    """No camera required: the browser page itself captures arrow keys/space/
    tab/escape and reports them back over the WebSocket (arrow keys rotate,
    +/- zoom, space toggles selection, tab switches objects, escape quits) —
    useful for trying the hologram out on a machine without a webcam, or
    without MediaPipe installed.

    See `run()` above for what the other parameters are for.
    """
    try:
        import websockets  # noqa: F401
    except ImportError as exc:
        raise RendererUnavailableError(
            "the 'websockets' package is required for the browser hologram — "
            "pip install -r requirements-vision.txt"
        ) from exc

    from jarvis.vision.blink_scheduler import BlinkScheduler
    from jarvis.vision.holographic_scene import HolographicScene

    scene = HolographicScene()
    blink_scheduler = BlinkScheduler()
    key_state = _KeyState()

    def update_scene() -> None:
        for key in key_state.take_edges():
            if key == " ":
                scene.selected = not scene.selected
            elif key == "Tab":
                scene.cycle_object(1)
        if key_state.is_held("ArrowLeft"):
            scene.rotation_y -= 4
        if key_state.is_held("ArrowRight"):
            scene.rotation_y += 4
        if key_state.is_held("ArrowUp"):
            scene.rotation_x -= 4
        if key_state.is_held("ArrowDown"):
            scene.rotation_x += 4
        if key_state.is_held("=") or key_state.is_held("+"):
            scene.zoom = min(6.0, scene.zoom * 1.03)
        if key_state.is_held("-"):
            scene.zoom = max(0.3, scene.zoom * 0.97)

    asyncio.run(
        _run_async(
            scene,
            update_scene=update_scene,
            fps=fps,
            stop_event=stop_event,
            speaking_state=speaking_state,
            assistant_state=assistant_state,
            blink_scheduler=blink_scheduler,
            key_state=key_state,
            http_port=http_port,
            ws_port=ws_port,
            open_browser=open_browser,
            brain_activity=brain_activity,
            agent_manager=agent_manager,
            wake_word=wake_word,
            style=style,
        )
    )
