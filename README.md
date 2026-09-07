# JARVIS

A desktop AI assistant, built in stages: type or talk to it, it opens apps,
answers questions, reasons through multi-step research using tools, remembers
things across sessions, and can run a gesture-controlled holographic
interface — driven with your bare hands — alongside the conversation.

## Current status

| Stage | What it does | Status |
| --- | --- | --- |
| 1. Foundation | Typed commands: open apps, time/date, web search, lock screen | ✅ |
| 2. Voice | Microphone input (SpeechRecognition) + text-to-speech output (pyttsx3) | ✅ |
| 3. AI Brain | Anything that isn't a known command goes to Gemini for real reasoning, with conversation memory | ✅ |
| 4. Tools | Calculator, file search/reading, web research, weather, sandboxed Python — the brain calls these itself | ✅ |
| 5. Memory | Long-term facts, named projects, preferences — persisted to disk, survives restarts | ✅ |
| 6. Vision + Holographics | Camera hand-tracking → gestures → a gesture-controlled 3D neural-activity visualization + live agent dashboard, runnable standalone or alongside voice/text | ✅ (needs your own camera/GPU to see it move) |
| 7. Full JARVIS | Multi-step autonomous plans (research → compare → save to project) via chained tool calls in one request | ✅ |

Stages 4/5/7 turned out to be one piece of work: giving the AI brain tools
*is* what makes multi-step planning and persistent memory possible, so the
"plan → tools → memory → report" loop from the Stage 7 vision is really just
Stage 3's brain with a bigger toolbox — see the architecture section below.

## How it's wired

```
                 request (typed or spoken)
                              |
                              v
                        Router (jarvis/router.py)
                    /                          \
     instant OS action?                   everything else
      (open app, time,                           |
       date, search, lock)                       v
                |                    AIBrain (jarvis/brain/ai_brain.py)
                |                    Gemini, with a tool-use loop:
                |                       calculate          (jarvis/tools/calculator.py)
                |                       search_files /     (jarvis/tools/file_tools.py)
                |                       read_document
                |                       web_research       (jarvis/tools/research.py)
                |                       get_weather        (jarvis/tools/weather.py, Open-Meteo, no key needed)
                |                       run_python          (jarvis/tools/code_tool.py, opt-in + confirmed)
                |                       remember_fact /     (jarvis/tools/memory_tools.py
                |                       recall_facts /       -> jarvis/memory/store.py,
                |                       *_project* /          a SQLite file that survives restarts)
                |                       set/get_preference
                \                                 /
                 \                               /
                          reply
                              |
                              v
                 jarvis/voice/speech_output.py
              (spoken via TTS, always echoed as text)


  Stage 6, standalone (hologram.py) or alongside the loop above (main.py --hologram):

     camera --> jarvis/vision/hand_tracker.py (MediaPipe's Hand Landmarker task; downloads
                                                its small model file to JARVIS_DATA_DIR on first use)
            --> jarvis/vision/gesture_engine.py (pinch/move/drag/swipe/spread/contract)
            --> jarvis/vision/holographic_scene.py (rotation/zoom state, gesture-driven camera control)

     the rest of JARVIS --> jarvis/voice/brain_activity.py (per-region "firing" levels)
                        --> jarvis/agents/ (real background tasks + live status)
                                          \\
                                           v
                            jarvis/vision/holographic_web.py (WebSocket + static file server)
                                          |
                                          v
                            jarvis/vision/web/hologram.js (Three.js/WebGL, runs in your browser)
```

`hologram.py` (or `main.py --hologram`) opens the hologram in your default
browser: Python serves `jarvis/vision/web/` over a small local HTTP server
and streams state to the page over a WebSocket, 30 times a second by
default — gesture-driven rotation/zoom, `BrainActivity`'s per-region firing
levels, `AgentManager`'s live agent status, and an assistant status label.
All the actual drawing, lighting, and glow happen with real WebGL shaders
and Three.js's `UnrealBloomPass`, not legacy immediate-mode OpenGL. Camera
capture, hand tracking, and gesture interpretation stay in Python,
unchanged. When `--hologram` is passed to `main.py`, the assistant loop
above runs on a background thread while the main thread runs this server; a
`threading.Event` lets either side tell the other to stop — saying "exit"
(or Ctrl-D) ends the conversation and shuts the server down, and pressing
Escape in the browser tab does the same in reverse.

The page shows a **glowing neural-activity graph**, not a face: ten
brain-inspired regions (prefrontal, association, reflex arc, sensory
cortex, language, motor cortex, hippocampus, predictive, brainstem,
cerebellum), each a small cluster of connected, colored particles arranged
around a sphere (`hologram.js`'s `buildBrainGraph`/`buildEdges`). These
aren't decorative — each region is a real, existing part of JARVIS's
pipeline (see `jarvis/voice/brain_activity.py`'s module docstring for the
exact mapping) and lights up when that part of the pipeline actually runs:
asking a question lights up association then prefrontal (routing, then AI
reasoning); JARVIS replying lights up language; a tool call lights up
motor (and predictive specifically for the weather tool); remembering or
recalling something lights up memory (shown as "HIPPOCAMPUS"). Brainstem
pulses once per broadcast frame regardless of activity — a baseline
"the process is alive" heartbeat. Cerebellum is marked "planned" and never
lights up — there's no real cerebellum-equivalent implemented yet.

A sidebar panel lists every region's LIVE/PLANNED status; each region also
gets an on-screen label (camera-projected onto its cluster's position each
frame) showing its name and current firing %. A status readout at the top
shows `CONNECTED`, `THINKING...`, `TALKING...`, or `LISTENING FOR "..."`
(from `AssistantState` plus `--wake-word`, if set) — all computed in
`holographic_web._status_label`.

Along the bottom, a live **agent dashboard** shows `jarvis/agents/`' real
background tasks, each running on its own thread and schedule regardless of
whether anyone is talking to JARVIS (`jarvis/agents/base.py`'s
`AgentManager`): `WeatherWatchAgent` refreshes the weather periodically,
`MemoryDigestAgent` reports fact/project counts, and `SystemHealthAgent`
checks whether the AI brain has an API key and the voice/vision packages
are installed (deliberately without ever opening the mic or camera itself,
since another thread may already have one open). A card per agent shows its
name and latest real result — not a script, an actual return value from
`run_once()`.

`--hologram-keyboard-demo` needs no camera: instead of hand gestures, the
browser page itself captures arrow keys (rotate) and Escape (quit) and
reports them back over the same WebSocket.

### An alternative look: `--hologram-style humanoid`

`jarvis/vision/web/humanoid.js`/`humanoid.html` is a second Three.js page —
a particle-built humanoid bust instead of the region graph, with a boot-up
"assembling" sequence, a `STATUS: IDLE / LISTENING / THINKING / TALKING`
readout, a glowing core across the face that ripples and brightens with
`mouthOpenness` while JARVIS talks, and layered particle "mountain range"
flourishes flanking the figure. It reads the exact same broadcast state as
the region graph (rotation/zoom from gestures, `mouthOpenness`,
`processing`, `assistantState`, `agents`) — `holographic_web.py` doesn't
know or care which page is open, it just serves whichever
`_page_for_style()` picks. Try it with:

```bash
python hologram.py --style humanoid
python hologram.py --style humanoid --keyboard-demo   # no camera needed
python main.py --voice --hologram --hologram-style humanoid
```

`AssistantState` gained a fourth state, `LISTENING`, set by `main.py`'s
`get_input()` right before it calls `speech_input.listen()` in voice mode —
the region-graph view folds it into its existing `LISTENING FOR "..."`
label, the humanoid view shows it as its own `STATUS: LISTENING`.

**On performance:** everything that animates per-frame in `humanoid.js` (the
boot assembly, the idle shimmer, the breathing drift) runs in the vertex
shader off a few uniforms, so the CPU never re-uploads a particle buffer
after setup — ~6k particles cost about the same as none. Points are drawn as
soft round sprites by a fragment shader rather than the default square GL
points, which is most of what separates "glow" from "pixels". The page also
watches its own frame rate and steps quality down (render resolution first,
then the bloom pass) if it can't hold ~40fps, so a weaker GPU degrades
instead of stuttering.

If the *combined* `main.py --voice --hologram` mode is choppy on your
machine, the camera + MediaPipe hand tracking is the expensive part, not the
rendering — run with `--hologram-keyboard-demo` (no camera at all) to
confirm, and drop `--hologram-fps` if so.

`Router` only special-cases the handful of things JARVIS can *do* instantly
on your machine. Everything else — questions, research, "remember this",
"what's in my sensor project", running a snippet of code — goes to
`AIBrain`, which decides on its own whether to just answer or to call one or
more tools first. A request like *"research batteryless sensors, compare the
approaches, and save the result to my sensor project"* becomes: the model
calls `web_research`, reads what comes back, calls it again or reasons over
it, then calls `add_project_note` — all within one `think()` call, no extra
prompting needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env         # then fill in GEMINI_API_KEY (get one free at aistudio.google.com/apikey)
```

Voice mode needs a working microphone and speakers.

- **Windows**: speech output goes through SAPI5, which needs `pywin32`
  (`pip install pywin32`) — it's in `requirements.txt`. Make sure Windows
  has a voice installed under Settings > Time & Language > Speech.
- **Linux**: you may need `portaudio` (`sudo apt install portaudio19-dev`)
  for `sounddevice`, and `espeak` (`sudo apt install espeak`) for speech
  *output* — `pyttsx3` has no voice of its own on Linux, it drives espeak.

**If JARVIS never says a word**, check the speech path on its own:

```bash
python -m jarvis.voice.speech_output    # starts TTS and speaks one test phrase
```

`SpeechOutput` prints a loud, specific reason when its engine won't start or
playback fails (missing espeak, dead driver, and so on) instead of quietly
falling back to text — a silent fallback is indistinguishable from JARVIS
being broken.

For the holographic interface (Stage 6), also install the vision extras:

```bash
pip install -r requirements-vision.txt
```

These (OpenCV, MediaPipe, websockets) are kept separate from the core
install since OpenCV/MediaPipe are large and camera-dependent — everything
else runs fine without them. The hologram itself renders in your browser
(via a CDN-loaded Three.js), so no local graphics package is required.

## Running it

```bash
python main.py                                   # type to JARVIS
python main.py --voice                            # talk to JARVIS
python main.py --voice --wake-word "hey jarvis"   # only react after the wake word

python hologram.py                                # camera-driven hologram, standalone
python hologram.py --keyboard-demo                # no camera: arrows/space/tab/esc
python hologram.py --style humanoid               # the particle humanoid-bust visualization instead

python main.py --voice --hologram                                    # talk to JARVIS while the hologram runs
python main.py --voice --hologram --hologram-keyboard-demo           # same, no camera needed for the hologram
python main.py --voice --hologram --hologram-style humanoid          # humanoid bust instead of the region graph
python main.py --hologram --hologram-camera 1 --hologram-fps 24      # tweak which camera / frame rate
```

Without `GEMINI_API_KEY` set, JARVIS still runs — known commands (open
app, time, date, search, lock) keep working, but free-form questions and
tool use get a message telling you the AI brain isn't configured.

### Example session

```
You: open chrome
JARVIS: Opening chrome.
You: what time is it
JARVIS: It's 5:25 PM.
You: what's the weather like right now
JARVIS: Weather for New York, New York, United States: Right now: partly
        cloudy, 18.5°C (feels like 17.0°C), wind 12.0 km/h.
You: remember that my main project is a batteryless environmental sensor
JARVIS: Got it — I'll remember that.
You: research recent approaches to batteryless environmental sensors, compare
     them, and add an interesting gap to my sensor project's notes
JARVIS: [searches the web, reads a few sources, compares them, saves a note]
        I found three main approaches — RF energy harvesting, piezoelectric,
        and solar-thermal hybrids. One gap that stood out: most designs
        assume constant ambient RF, but there's little work on adaptive
        harvesting when the signal is intermittent. I've saved that to your
        sensor project.
You: forget this conversation
JARVIS: Alright, fresh start — what's next?
```

## Configuration

All of these go in `.env` (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | *(none)* | Required for the AI brain / tools. Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Without it, only instant OS commands work. |
| `JARVIS_MODEL` | `gemini-3.6-flash` | Model used for reasoning. See [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for current options. |
| `JARVIS_MAX_HISTORY_TURNS` | `20` | How many past request/response turns stay in short-term memory. |
| `JARVIS_MAX_TOOL_ITERATIONS` | `8` | Safety cap on chained tool calls per request. |
| `JARVIS_FILES_ROOT` | `~/Documents` (or `~`) | `search_files`/`read_document` are confined to this directory tree. |
| `JARVIS_ENABLE_CODE_TOOL` | `false` | Lets the brain run Python it wrote (always with a y/N confirmation first). Off by default — see "A note on safety" below. |
| `JARVIS_DATA_DIR` | `~/.jarvis` | Where the memory database (`memory.db`) lives. |

Free-tier Gemini models can have quite tight daily request caps (the exact
number varies by model and changes over time). If you hit one, JARVIS tells
you plainly ("I've hit the AI service's rate limit...") rather than dumping
the raw API error — wait a bit, or point `JARVIS_MODEL` at a model with more
free-tier headroom.

## A note on safety

- **File tools are sandboxed.** `search_files`/`read_document` can't read or
  list anything outside `JARVIS_FILES_ROOT`, even if asked to.
- **`web_research` fetches real, untrusted pages.** The brain is instructed
  to treat what comes back as data, not instructions — but always keep that
  in mind if you extend it.
- **The code tool is off by default** and, once enabled, always shows you
  the exact code and asks `[y/N]` before running it — precisely because a
  page fetched by `web_research` could otherwise try to talk the model into
  running something harmful. Only enable it if you're comfortable reviewing
  what it wants to run each time.
- **`get_weather` sends your public IP to ipapi.co** to guess your location,
  but only when you ask about "the weather" without naming a place — asking
  about a named city skips that lookup entirely (goes straight to the
  geocoder instead).

## A note on `--hologram` and threading

`pyttsx3`'s drivers are thread-affine, and `--hologram` runs the assistant
loop (and its `speech_output.say()` calls) on a background thread, since the
render loop needs the main thread. That combination used to break speech
outright — most severely on Windows, where the SAPI5 driver is a COM object
that must be created and called on one thread that has called
`CoInitialize()`. An engine built on the main thread and then driven from
the assistant thread crosses a COM apartment boundary, where calls fail or
simply make no sound; the old code caught that exception and logged it at
INFO, which nothing configures a handler for, so JARVIS just never spoke and
never said why.

`SpeechOutput` now owns a single dedicated `jarvis-tts` worker thread that
initializes COM (on Windows), creates the engine, and runs every utterance;
`say()` queues text for that thread and waits for it. So the engine is
always created and driven on the same thread no matter who calls `say()`,
which is what `test_engine_is_created_and_driven_on_one_non_calling_thread`
pins down.

## Testing

```bash
python -m unittest discover -s tests
```

Everything except the browser page itself is covered: system commands,
router intents, the AI brain's tool-calling loop, the toolbox (calculator,
file sandboxing, code-tool gating, mocked web research and weather
lookups), persistent memory, `SpeakingState`'s and `BlinkScheduler`'s timing
(both take an injectable clock, so tests don't wait on real time),
`BrainActivity`'s per-region decay, the agent framework (`AgentManager`
actually starting/stopping real background threads and reporting
idle/running/complete/error status) and each concrete agent, and — for
Stage 6 — the gesture engine, the holographic scene's rotation/zoom state,
and `holographic_web.py`'s WebSocket/HTTP server itself (a real client
connects over a real local socket and checks the broadcast payload,
including the region/agent/status-label/assistant-state fields, and which
page `_page_for_style()` picks for each `--hologram-style`), all pure-logic
and camera/display-free. `jarvis/vision/hand_tracker.py` needs a real
camera, and `jarvis/vision/web/hologram.js`/`humanoid.js` need a real
browser/GPU, so verify those by actually running `python hologram.py`
(add `--style humanoid` for the other view).

## Extending JARVIS

- **New instant command**: add a regex + handler to `jarvis/router.py`'s
  `_PATTERNS`/`_dispatch`, and the action itself in `jarvis/commands/`.
- **New app alias**: add an entry to the per-OS tables in
  `jarvis/commands/system_commands.py`.
- **New tool**: write the function, add its spec (name/description/JSON
  Schema `parameters`) + dispatch entry in `jarvis/tools/registry.py`. The
  brain picks it up automatically — no other wiring needed.
- **New gesture**: add detection logic to `jarvis/vision/gesture_engine.py`
  (it's pure geometry over landmark data — easy to unit-test) and react to
  the new `GestureEvent.type` in `jarvis/vision/holographic_scene.py`.
- **New holographic object**: add a point-cloud generator function to
  `jarvis/vision/holographic_scene.py` and a name to `OBJECTS`.
- **New background agent**: subclass `Agent` in `jarvis/agents/base.py`
  (implement `run_once()`, returning a short result string) and register it
  with an interval in `main.py`'s `AgentManager` — its live status shows up
  in the dashboard automatically.
- **New brain-activity region**: add the key to `REGIONS` in
  `jarvis/voice/brain_activity.py` (and to `PLANNED_REGIONS` if there's no
  real implementation behind it yet), then call `.pulse("your_region")`
  wherever the real work happens; mirror the key/color/label in
  `hologram.js`'s `REGIONS` array.

## Where this could go next

The seven stages here cover voice, reasoning, tools, memory, and a
gesture-driven 3D interface — the core of the "JARVIS" vision. Natural next
steps if you want to keep pushing it further: semantic (embedding-based)
memory recall instead of keyword matching, letting the AI brain narrate what
it's doing on the holographic display in real time, face/pose tracking
alongside hands, and a proper planning trace UI so you can see each tool call
the brain made for a multi-step request instead of just the final answer.
