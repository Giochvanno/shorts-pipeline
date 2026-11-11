"""Pipeline state in SQLite: which step of which video is pending, running, done or failed.

Why a database and not files: a crash at any moment must leave a consistent
picture, and a restart must continue from the first unfinished step instead of
starting over (and paying again).

Allowed step transitions (anything else raises InvalidTransitionError):

    pending -> running -> done
                  |
                  v
               failed -> running   (retry)

    any -> pending                 (reset_from: "redo the script", "redo visuals")

Every transition is one UPDATE whose WHERE clause checks the current status,
so a step can never be, say, finished twice.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Self

INTERRUPTED = "interrupted: process stopped while the step was running"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class StateError(Exception):
    """Base class for state errors."""


class UnknownVideoError(StateError):
    pass


class InvalidTransitionError(StateError):
    pass


@dataclass(frozen=True, slots=True)
class StepRecord:
    video_id: int
    name: str
    position: int
    status: StepStatus
    attempts: int
    error: str | None
    started_at: str | None
    finished_at: str | None


# Each entry is applied once, in order. PRAGMA user_version stores how many are applied.
# Never edit an applied migration: add a new one (tables for artifacts, costs, ... come later).
MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE videos (
        id          INTEGER PRIMARY KEY,
        series_id   TEXT NOT NULL,
        topic       TEXT NOT NULL,
        created_at  TEXT NOT NULL
    );
    CREATE TABLE steps (
        video_id     INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        name         TEXT NOT NULL,
        position     INTEGER NOT NULL,
        status       TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'running', 'done', 'failed')),
        attempts     INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        error        TEXT,
        started_at   TEXT,
        finished_at  TEXT,
        PRIMARY KEY (video_id, name),
        UNIQUE (video_id, position)
    );
    CREATE INDEX steps_by_status ON steps(status);
    """,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class StateStore:
    """All reads and writes of pipeline state go through this class."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # WAL: a crash mid-write never corrupts the file,
        # and the bot can read while the runner writes.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    # --- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        version: int = self._conn.execute("PRAGMA user_version").fetchone()[0]
        return version

    def _migrate(self) -> None:
        for number in range(self.schema_version, len(MIGRATIONS)):
            # One transaction per migration: it is applied fully or not at all.
            self._conn.executescript(
                f"BEGIN;\n{MIGRATIONS[number]}\nPRAGMA user_version = {number + 1};\nCOMMIT;"
            )

    # --- videos -------------------------------------------------------------------

    def create_video(self, series_id: str, topic: str, steps: Sequence[str]) -> int:
        """Create a video with all its steps pending, in the given order."""
        if not steps:
            raise ValueError("a video needs at least one step")
        if len(set(steps)) != len(steps):
            raise ValueError(f"duplicate step names: {list(steps)}")
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO videos (series_id, topic, created_at) VALUES (?, ?, ?)",
                (series_id, topic, _now()),
            )
            video_id = cur.lastrowid
            assert video_id is not None
            self._conn.executemany(
                "INSERT INTO steps (video_id, name, position) VALUES (?, ?, ?)",
                [(video_id, name, i) for i, name in enumerate(steps)],
            )
        return video_id

    def steps(self, video_id: int) -> list[StepRecord]:
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE video_id = ? ORDER BY position", (video_id,)
        ).fetchall()
        if not rows:
            raise UnknownVideoError(f"no video with id {video_id}")
        return [_to_record(row) for row in rows]

    def step(self, video_id: int, name: str) -> StepRecord:
        for record in self.steps(video_id):
            if record.name == name:
                return record
        raise StateError(f"video {video_id} has no step {name!r}")

    def next_step(self, video_id: int) -> StepRecord | None:
        """First step that is not done, or None when the video is finished.

        This is what makes restarts resume from the failed step.
        """
        return next((s for s in self.steps(video_id) if s.status != StepStatus.DONE), None)

    def video_status(self, video_id: int) -> StepStatus:
        """Derived from steps, never stored, so it cannot disagree with them."""
        statuses = {s.status for s in self.steps(video_id)}
        for status in (StepStatus.RUNNING, StepStatus.FAILED):
            if status in statuses:
                return status
        if statuses == {StepStatus.DONE}:
            return StepStatus.DONE
        return StepStatus.PENDING

    # --- step transitions ---------------------------------------------------------

    def start_step(self, video_id: int, name: str) -> StepRecord:
        """pending/failed -> running. Only the next step in order may start."""
        upcoming = self.next_step(video_id)
        if upcoming is None or upcoming.name != name:
            expected = upcoming.name if upcoming else "nothing (video is done)"
            raise InvalidTransitionError(f"cannot start {name!r}: next step is {expected}")
        self._transition(
            video_id,
            name,
            allowed_from=(StepStatus.PENDING, StepStatus.FAILED),
            sql="status = 'running', attempts = attempts + 1, error = NULL,"
            " started_at = :now, finished_at = NULL",
        )
        return self.step(video_id, name)

    def finish_step(self, video_id: int, name: str) -> StepRecord:
        """running -> done."""
        self._transition(
            video_id,
            name,
            allowed_from=(StepStatus.RUNNING,),
            sql="status = 'done', finished_at = :now",
        )
        return self.step(video_id, name)

    def fail_step(self, video_id: int, name: str, error: str) -> StepRecord:
        """running -> failed, with the reason kept for the alert and the soak report."""
        self._transition(
            video_id,
            name,
            allowed_from=(StepStatus.RUNNING,),
            sql="status = 'failed', error = :error, finished_at = :now",
            error=error,
        )
        return self.step(video_id, name)

    def reset_from(self, video_id: int, name: str) -> None:
        """Make `name` and every later step pending again; earlier steps stay done.

        Used by the Telegram buttons "redo script" / "redo visuals".
        """
        target = self.step(video_id, name)
        with self._conn:
            self._conn.execute(
                "UPDATE steps SET status = 'pending', error = NULL,"
                " started_at = NULL, finished_at = NULL"
                " WHERE video_id = ? AND position >= ?",
                (video_id, target.position),
            )

    def recover_interrupted(self) -> int:
        """Call once at startup: a step still 'running' means the process died mid-step.

        With a single worker nothing can legitimately be running at startup,
        so those steps become failed and will be retried.
        """
        with self._conn:
            cur = self._conn.execute(
                "UPDATE steps SET status = 'failed', error = ?, finished_at = ?"
                " WHERE status = 'running'",
                (INTERRUPTED, _now()),
            )
        return cur.rowcount

    def _transition(
        self,
        video_id: int,
        name: str,
        *,
        allowed_from: tuple[StepStatus, ...],
        sql: str,
        error: str | None = None,
    ) -> None:
        placeholders = ", ".join(f":from{i}" for i in range(len(allowed_from)))
        params: dict[str, object] = {
            "video_id": video_id,
            "name": name,
            "now": _now(),
            "error": error,
            **{f"from{i}": s.value for i, s in enumerate(allowed_from)},
        }
        with self._conn:
            cur = self._conn.execute(
                f"UPDATE steps SET {sql}"
                f" WHERE video_id = :video_id AND name = :name AND status IN ({placeholders})",
                params,
            )
        if cur.rowcount == 1:
            return
        current = self.step(video_id, name)  # raises if the video or step does not exist
        allowed = "/".join(allowed_from)
        raise InvalidTransitionError(
            f"step {name!r} of video {video_id} is {current.status}, expected {allowed}"
        )


def _to_record(row: sqlite3.Row) -> StepRecord:
    return StepRecord(
        video_id=row["video_id"],
        name=row["name"],
        position=row["position"],
        status=StepStatus(row["status"]),
        attempts=row["attempts"],
        error=row["error"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )
