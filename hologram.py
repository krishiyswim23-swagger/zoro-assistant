#!/usr/bin/env python3
"""Stage 6 entry point: the gesture-controlled holographic interface.

Runs as its own program (not inside main.py) because it owns a real-time
render loop and a camera capture loop, which don't mix cleanly with the
text/voice request-response loop in main.py.

Usage:
    python hologram.py                  # camera + hand tracking
    python hologram.py --camera 1       # use a different camera index
    python hologram.py --keyboard-demo  # no camera needed: arrow keys/space/tab/esc

Opens the hologram in your default browser (a Three.js/WebGL page) and prints
the URL in case it doesn't; requires the vision extras: pip install -r
requirements-vision.txt
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zoro holographic interface")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--fps", type=int, default=30, help="target render frame rate (default 30)")
    parser.add_argument(
        "--keyboard-demo",
        action="store_true",
        help="drive the hologram with arrow keys/space/tab instead of a camera",
    )
    parser.add_argument(
        "--style",
        choices=["regions", "humanoid"],
        default="regions",
        help=(
            "which hologram visualization to open: 'regions' (default) is the neural-activity "
            "brain graph; 'humanoid' is a particle humanoid bust with a boot-up sequence and a "
            "LISTENING/THINKING/TALKING status readout"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from jarvis.vision.hand_tracker import VisionUnavailableError
    from jarvis.vision.holographic_web import RendererUnavailableError, run, run_keyboard_demo

    try:
        if args.keyboard_demo:
            run_keyboard_demo(fps=args.fps, style=args.style)
        else:
            run(camera_index=args.camera, fps=args.fps, style=args.style)
    except (VisionUnavailableError, RendererUnavailableError) as exc:
        print(f"Can't start the hologram: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
