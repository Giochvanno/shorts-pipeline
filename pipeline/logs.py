"""Structured logs: one JSON object per line, easy to grep and to aggregate in the soak report.

{"ts": "...", "level": "INFO", "event": "step_failed", "video_id": 3, "step": "voice", ...}
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER_NAME = "pipeline"

# A library must not print on its own: without a configured handler, records are dropped
# (instead of Python's last-resort stderr output). The CLI attaches the JSON file handler.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

# Attributes every LogRecord has; anything else came from `extra=` and is ours.
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_to_file(path: Path) -> logging.Handler:
    """Attach a JSON-lines file handler; returns it so the caller can remove it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger = get_logger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler
