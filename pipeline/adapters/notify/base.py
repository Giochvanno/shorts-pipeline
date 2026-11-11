"""Notify port: how the pipeline tells a human that something needs attention."""

from typing import Protocol


class Notifier(Protocol):
    def send(self, text: str) -> None: ...
