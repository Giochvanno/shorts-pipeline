"""Alerts to the terminal (stderr). Telegram replaces this in stage 2."""

import sys


class ConsoleNotifier:
    def send(self, text: str) -> None:
        print(f"ALERT  {text}", file=sys.stderr)
