import asyncio
import json
import unittest

from jarvis.agents.base import Agent, AgentManager
from jarvis.vision import holographic_web
from jarvis.vision.blink_scheduler import BlinkScheduler
from jarvis.vision.holographic_scene import HolographicScene
from jarvis.voice.assistant_state import IDLE, LISTENING, TALKING, THINKING, AssistantState
from jarvis.voice.brain_activity import BrainActivity


class KeyStateTest(unittest.TestCase):
    def test_held_key_tracked_until_released(self) -> None:
        key_state = holographic_web._KeyState()
        key_state.handle_message(json.dumps({"type": "key", "key": "ArrowLeft", "down": True}))
        self.assertTrue(key_state.is_held("ArrowLeft"))
        key_state.handle_message(json.dumps({"type": "key", "key": "ArrowLeft", "down": False}))
        self.assertFalse(key_state.is_held("ArrowLeft"))

    def test_repeated_keydown_only_edges_once(self) -> None:
        key_state = holographic_web._KeyState()
        key_state.handle_message(json.dumps({"type": "key", "key": "Tab", "down": True}))
        key_state.handle_message(json.dumps({"type": "key", "key": "Tab", "down": True}))
        self.assertEqual(key_state.take_edges(), ["Tab"])
        self.assertEqual(key_state.take_edges(), [])  # drained

    def test_escape_sets_flag(self) -> None:
        key_state = holographic_web._KeyState()
        self.assertFalse(key_state.escape_requested)
        key_state.handle_message(json.dumps({"type": "key", "key": "Escape", "down": True}))
        self.assertTrue(key_state.escape_requested)

    def test_malformed_or_irrelevant_messages_ignored(self) -> None:
        key_state = holographic_web._KeyState()
        key_state.handle_message("not json")
        key_state.handle_message(json.dumps({"type": "other"}))
        key_state.handle_message(json.dumps({"type": "key", "key": 5, "down": True}))
        self.assertEqual(key_state.take_edges(), [])
        self.assertFalse(key_state.escape_requested)


class ScenePayloadTest(unittest.TestCase):
    def test_payload_shape_matches_hologram_js_expectations(self) -> None:
        scene = HolographicScene()
        scene.rotation_x = 12.5
        scene.rotation_y = 30.0
        scene.zoom = 2.0
        scene.selected = True
        scene.set_mouth_openness(0.4)
        scene.set_blink(0.1)
        scene.set_processing(True)

        payload = json.loads(
            holographic_web._scene_payload(
                scene, regions={"prefrontal": 0.5}, agents=[{"id": "a", "name": "A", "status": "idle", "detail": ""}],
                status_label="ZORO — CONNECTED",
                assistant_state_raw="talking",
            )
        )

        self.assertEqual(
            payload,
            {
                "rotationX": 12.5,
                "rotationY": 30.0,
                "zoom": 2.0,
                "selected": True,
                "mouthOpenness": 0.4,
                "blink": 0.1,
                "processing": True,
                "regions": {"prefrontal": 0.5},
                "agents": [{"id": "a", "name": "A", "status": "idle", "detail": ""}],
                "statusLabel": "ZORO — CONNECTED",
                "assistantState": "talking",
            },
        )

    def test_assistant_state_raw_defaults_to_idle(self) -> None:
        scene = HolographicScene()
        payload = json.loads(
            holographic_web._scene_payload(scene, regions={}, agents=[], status_label="ZORO — CONNECTED")
        )
        self.assertEqual(payload["assistantState"], "idle")


class StatusLabelTest(unittest.TestCase):
    def test_none_assistant_state_defaults_to_connected(self) -> None:
        self.assertEqual(holographic_web._status_label(None, None), "ZORO — CONNECTED")

    def test_thinking_state(self) -> None:
        state = AssistantState()
        state.set(THINKING)
        self.assertEqual(holographic_web._status_label(state, None), "THINKING...")

    def test_talking_state(self) -> None:
        state = AssistantState()
        state.set(TALKING)
        self.assertEqual(holographic_web._status_label(state, None), "TALKING...")

    def test_idle_with_wake_word_shows_listening(self) -> None:
        state = AssistantState()
        state.set(IDLE)
        self.assertEqual(holographic_web._status_label(state, "hey jarvis"), 'LISTENING FOR "HEY JARVIS"')

    def test_idle_without_wake_word_shows_connected(self) -> None:
        state = AssistantState()
        state.set(IDLE)
        self.assertEqual(holographic_web._status_label(state, None), "ZORO — CONNECTED")

    def test_listening_state_without_wake_word(self) -> None:
        state = AssistantState()
        state.set(LISTENING)
        self.assertEqual(holographic_web._status_label(state, None), "LISTENING...")

    def test_listening_state_with_wake_word(self) -> None:
        state = AssistantState()
        state.set(LISTENING)
        self.assertEqual(holographic_web._status_label(state, "hey jarvis"), 'LISTENING FOR "HEY JARVIS"')


class AssistantStateRawTest(unittest.TestCase):
    def test_none_defaults_to_idle(self) -> None:
        self.assertEqual(holographic_web._assistant_state_raw(None), IDLE)

    def test_reflects_current_state(self) -> None:
        state = AssistantState()
        state.set(LISTENING)
        self.assertEqual(holographic_web._assistant_state_raw(state), LISTENING)


class PageForStyleTest(unittest.TestCase):
    def test_regions_style_uses_index_page(self) -> None:
        self.assertEqual(holographic_web._page_for_style("regions"), "index.html")

    def test_humanoid_style_uses_humanoid_page(self) -> None:
        self.assertEqual(holographic_web._page_for_style("humanoid"), "humanoid.html")

    def test_unknown_style_falls_back_to_index_page(self) -> None:
        self.assertEqual(holographic_web._page_for_style("something-else"), "index.html")


class RendererUnavailableTest(unittest.TestCase):
    def test_run_raises_when_websockets_not_installed(self) -> None:
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"websockets": None}):
            with self.assertRaises(holographic_web.RendererUnavailableError):
                holographic_web.run()

    def test_run_keyboard_demo_raises_when_websockets_not_installed(self) -> None:
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"websockets": None}):
            with self.assertRaises(holographic_web.RendererUnavailableError):
                holographic_web.run_keyboard_demo()


class _FlagStopEvent:
    """A minimal stand-in for threading.Event, usable from async test code."""

    def __init__(self) -> None:
        self._flag = False

    def is_set(self) -> bool:
        return self._flag

    def set(self) -> None:
        self._flag = True


class _EchoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(agent_id="echo", name="Echo")

    def run_once(self) -> str:
        return "did a thing"


class RunAsyncIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_broadcasts_scene_state_and_accepts_key_messages(self) -> None:
        import websockets

        scene = HolographicScene()
        scene.rotation_x = 7.0
        scene.zoom = 1.5
        key_state = holographic_web._KeyState()
        brain_activity = BrainActivity()
        brain_activity.pulse("prefrontal", 0.9)
        agent_manager = AgentManager()
        agent_manager.register(_EchoAgent(), interval_seconds=10)
        agent_manager.start()
        self.addCleanup(agent_manager.stop)

        ready = asyncio.Event()
        ports: dict[str, int] = {}

        def on_ready(http_port: int, ws_port: int) -> None:
            ports["http"] = http_port
            ports["ws"] = ws_port
            ready.set()

        stop_event = _FlagStopEvent()

        server_task = asyncio.create_task(
            holographic_web._run_async(
                scene,
                update_scene=lambda: None,
                fps=30,
                stop_event=stop_event,
                speaking_state=None,
                assistant_state=None,
                blink_scheduler=BlinkScheduler(),
                key_state=key_state,
                http_port=0,
                ws_port=0,
                open_browser=False,
                brain_activity=brain_activity,
                agent_manager=agent_manager,
                wake_word="hey jarvis",
                on_ready=on_ready,
            )
        )

        try:
            await asyncio.wait_for(ready.wait(), timeout=5.0)

            async with websockets.connect(f"ws://127.0.0.1:{ports['ws']}") as client:
                raw = await asyncio.wait_for(client.recv(), timeout=5.0)
                payload = json.loads(raw)
                self.assertEqual(payload["rotationX"], 7.0)
                self.assertEqual(payload["zoom"], 1.5)
                self.assertIn("mouthOpenness", payload)
                self.assertIn("blink", payload)
                self.assertAlmostEqual(payload["regions"]["prefrontal"], 0.9, places=1)
                self.assertIn("brainstem", payload["regions"])  # the heartbeat pulse
                self.assertEqual(payload["agents"][0]["id"], "echo")
                self.assertEqual(payload["statusLabel"], 'LISTENING FOR "HEY JARVIS"')
                self.assertEqual(payload["assistantState"], "idle")  # no assistant_state passed in

                await client.send(json.dumps({"type": "key", "key": "ArrowLeft", "down": True}))
                await asyncio.sleep(0.2)  # let the server loop process the incoming message
                self.assertTrue(key_state.is_held("ArrowLeft"))

                await client.send(json.dumps({"type": "key", "key": "Escape", "down": True}))
                await asyncio.wait_for(server_task, timeout=5.0)
                self.assertTrue(stop_event.is_set())
        finally:
            if not server_task.done():
                stop_event.set()
                await asyncio.wait_for(server_task, timeout=5.0)


if __name__ == "__main__":
    unittest.main()
