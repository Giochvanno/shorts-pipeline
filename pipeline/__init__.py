"""Shorts pipeline core: idempotent steps that turn an idea into a YouTube Short.

The core is a plain library. It reads ideas and writes results; it knows nothing
about Telegram, FastAPI, MCP or agents. Those are outer layers built on top.
"""
