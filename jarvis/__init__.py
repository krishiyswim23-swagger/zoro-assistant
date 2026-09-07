"""JARVIS — a desktop AI assistant.

Package layout:
    jarvis.commands  -- Stage 1: system actions (open apps, time/date, web search)
    jarvis.voice      -- Stage 2: speech input/output
    jarvis.brain      -- Stage 3: general-purpose AI reasoning
    jarvis.router     -- decides whether a request is a known command or goes to the brain
"""

__version__ = "0.3.0"
