"""StateStore: statuses, restarts from the failed step, crash recovery, guards."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from pipeline.state import (
    INTERRUPTED,
    MIGRATIONS,
    InvalidTransitionError,
    StateError,
    StateStore,
    StepStatus,
    UnknownVideoError,
)

STEPS = ("idea", "script", "voice", "render")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "pipeline.db"


@pytest.fixture
def store(db_path: Path) -> Iterator[StateStore]:
    # Closing matters on Windows: an open SQLite file cannot be deleted by tmp_path cleanup.
    with StateStore(db_path) as s:
        yield s


@pytest.fixture
def video(store: StateStore) -> int:
    return store.create_video("fact-of-day", "Why the sky is blue", STEPS)


def _complete(store: StateStore, video_id: int, *names: str) -> None:
    for name in names:
        store.start_step(video_id, name)
        store.finish_step(video_id, name)


# --- happy path -------------------------------------------------------------------


def test_new_video_has_all_steps_pending_in_order(store: StateStore, video: int) -> None:
    steps = store.steps(video)

    assert [s.name for s in steps] == list(STEPS)
    assert {s.status for s in steps} == {StepStatus.PENDING}
    assert store.video_status(video) == StepStatus.PENDING


def test_full_run_marks_video_done(store: StateStore, video: int) -> None:
    _complete(store, video, *STEPS)

    assert store.next_step(video) is None
    assert store.video_status(video) == StepStatus.DONE
    record = store.step(video, "render")
    assert record.attempts == 1
    assert record.started_at is not None
    assert record.finished_at is not None


# --- restarts: the point of this module -------------------------------------------


def test_restart_resumes_from_failed_step_not_from_start(store: StateStore, video: int) -> None:
    _complete(store, video, "idea", "script")
    store.start_step(video, "voice")
    store.fail_step(video, "voice", "TTS timeout")

    upcoming = store.next_step(video)

    assert upcoming is not None
    assert upcoming.name == "voice"
    assert upcoming.error == "TTS timeout"
    assert store.video_status(video) == StepStatus.FAILED
    # Earlier steps are untouched, so they will not run (or be paid for) again.
    assert store.step(video, "script").attempts == 1


def test_retry_counts_attempts_and_clears_error(store: StateStore, video: int) -> None:
    for _ in range(2):
        store.start_step(video, "idea")
        store.fail_step(video, "idea", "boom")

    retried = store.start_step(video, "idea")

    assert retried.attempts == 3
    assert retried.error is None
    assert retried.status == StepStatus.RUNNING


def test_state_survives_process_restart(db_path: Path) -> None:
    with StateStore(db_path) as first:
        video_id = first.create_video("fact-of-day", "Octopus hearts", STEPS)
        _complete(first, video_id, "idea")

    with StateStore(db_path) as second:  # a new process opening the same file
        upcoming = second.next_step(video_id)

    assert upcoming is not None
    assert upcoming.name == "script"


def test_crash_mid_step_is_recovered_as_failed(db_path: Path) -> None:
    with StateStore(db_path) as before_crash:
        video_id = before_crash.create_video("fact-of-day", "Honey never spoils", STEPS)
        before_crash.start_step(video_id, "idea")
        # ...the process dies here: the step stays "running" forever.

    with StateStore(db_path) as after_restart:
        recovered = after_restart.recover_interrupted()
        step = after_restart.step(video_id, "idea")
        retried = after_restart.start_step(video_id, "idea")

    assert recovered == 1
    assert step.status == StepStatus.FAILED
    assert step.error == INTERRUPTED
    assert retried.attempts == 2


def test_reset_from_reruns_later_steps_only(store: StateStore, video: int) -> None:
    _complete(store, video, *STEPS)

    store.reset_from(video, "voice")  # e.g. "redo visuals" button

    statuses = {s.name: s.status for s in store.steps(video)}
    assert statuses == {
        "idea": StepStatus.DONE,
        "script": StepStatus.DONE,
        "voice": StepStatus.PENDING,
        "render": StepStatus.PENDING,
    }
    upcoming = store.next_step(video)
    assert upcoming is not None
    assert upcoming.name == "voice"


# --- guards: illegal moves are errors, not silent corruption ----------------------


def test_cannot_finish_a_step_that_never_started(store: StateStore, video: int) -> None:
    with pytest.raises(InvalidTransitionError, match="pending"):
        store.finish_step(video, "idea")


def test_cannot_finish_twice(store: StateStore, video: int) -> None:
    _complete(store, video, "idea")

    with pytest.raises(InvalidTransitionError):
        store.finish_step(video, "idea")


def test_cannot_skip_ahead(store: StateStore, video: int) -> None:
    with pytest.raises(InvalidTransitionError, match="next step is idea"):
        store.start_step(video, "voice")


def test_cannot_start_when_video_is_done(store: StateStore, video: int) -> None:
    _complete(store, video, *STEPS)

    with pytest.raises(InvalidTransitionError, match="video is done"):
        store.start_step(video, "idea")


def test_unknown_video_and_step(store: StateStore, video: int) -> None:
    with pytest.raises(UnknownVideoError):
        store.steps(video + 1)
    with pytest.raises(StateError, match="no step"):
        store.step(video, "publish")


@pytest.mark.parametrize("steps", [(), ("idea", "idea")], ids=["empty", "duplicates"])
def test_bad_step_lists_are_rejected(store: StateStore, steps: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="step"):
        store.create_video("fact-of-day", "x", steps)


def test_database_rejects_unknown_status_even_via_raw_sql(db_path: Path, video: int) -> None:
    # Second line of defence: the CHECK constraint in the schema itself.
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE steps SET status = 'almost-done' WHERE video_id = ?", (video,))
    finally:
        conn.close()


# --- migrations -------------------------------------------------------------------


def test_migrations_apply_once(db_path: Path) -> None:
    with StateStore(db_path) as s:
        assert s.schema_version == len(MIGRATIONS)
        video_id = s.create_video("fact-of-day", "x", STEPS)

    with StateStore(db_path) as reopened:  # must not re-run CREATE TABLE or lose data
        assert reopened.schema_version == len(MIGRATIONS)
        assert len(reopened.steps(video_id)) == len(STEPS)
