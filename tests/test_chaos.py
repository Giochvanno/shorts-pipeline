"""Chaos tests: fakes randomly fail, stall and return garbage; the pipeline must cope.

What is proven here, with fixed seeds so every run is reproducible:
1. Under heavy chaos at least 95 % of videos finish without a human, and the rest
   finish after one `resume` (the plan's stability target).
2. A good answer is never paid for twice: per video exactly one script, one voice
   track and one image per scene get through, however many retries happen around them.
3. Garbage never reaches the disk.
4. A crash mid-step resumes in a new process without re-calling finished work.

Render is replaced by a light stub that checks its inputs: it has no adapter, so
chaos cannot touch it, and real renders are covered by test_step_render.
"""

import random
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.adapters.chaos import ChaosConfig, ChaosImages, ChaosLlm, ChaosTts
from pipeline.adapters.images import ImageRequest, ImageResult
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.registry import Adapters, build_adapters
from pipeline.adapters.tts.fake import FakeTts
from pipeline.config import load_channel, load_topics
from pipeline.runner import DEFAULT_STEPS, RetryPolicy, Runner, StepFn
from pipeline.state import StateStore, StepStatus
from pipeline.steps import script, visuals, voice
from pipeline.steps.base import StepContext, StepResult
from pipeline.testing import fake_context

HEAVY = ChaosConfig(fail_rate=0.25, garbage_rate=0.15, seed=1)  # 40 % of calls go wrong
POLICY = RetryPolicy(max_attempts=4, base_delay_s=0.0, jitter=0.0)


def stub_render(ctx: StepContext) -> StepResult:
    scenes = script.load(ctx.run_dir).scenes
    frames = visuals.frame_paths(ctx.run_dir, len(scenes))
    voice.voice_file(ctx.run_dir)
    return StepResult(outputs=tuple(frames), cost_usd=0.0, cached=False)


STEPS: list[tuple[str, StepFn]] = [*DEFAULT_STEPS[:-1], ("render", stub_render)]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[StateStore]:
    with StateStore(tmp_path / "pipeline.db") as s:
        yield s


def _chaos_adapters(config: ChaosConfig) -> Adapters:
    return build_adapters(load_channel().models, fake=True, chaos=config, sleep=lambda _s: None)


def _runner(store: StateStore, tmp_path: Path, adapters: Adapters) -> Runner:
    return Runner(
        store=store,
        adapters=adapters,
        runs_dir=tmp_path / "runs",
        policy=POLICY,
        steps=STEPS,
        sleep=lambda _s: None,
        rng=random.Random(0),
    )


def _stats(adapters: Adapters) -> tuple[int, int, int]:
    assert isinstance(adapters.llm_heavy, ChaosLlm)
    assert isinstance(adapters.tts, ChaosTts)
    assert isinstance(adapters.images, ChaosImages)
    return adapters.llm_heavy.stats.passed, adapters.tts.stats.passed, adapters.images.stats.passed


# --- 1 + 2: twenty videos under heavy chaos ------------------------------------------


def test_twenty_videos_survive_heavy_chaos_and_never_pay_twice(
    store: StateStore, tmp_path: Path
) -> None:
    adapters = _chaos_adapters(HEAVY)
    runner = _runner(store, tmp_path, adapters)
    topics = load_topics("fact-of-day")[:20]

    first_pass: dict[int, StepStatus] = {}
    for topic in topics:
        video = runner.create_video("fact-of-day", topic)
        before = _stats(adapters)
        first_pass[video] = runner.run_video(video).status
        if first_pass[video] == StepStatus.DONE:
            llm, tts, img = (a - b for a, b in zip(_stats(adapters), before, strict=True))
            scenes = len(script.load(tmp_path / "runs" / str(video)).scenes)
            # Exactly one good script, one good voice, one good image per scene.
            assert (llm, tts, img) == (1, 1, scenes), f"video {video} paid twice"

    done = sum(status == StepStatus.DONE for status in first_pass.values())
    assert done / len(topics) >= 0.95, f"only {done}/{len(topics)} finished unattended"

    # The rest finish after one resume ("next night"), still without re-paying done work.
    for video, status in first_pass.items():
        if status != StepStatus.DONE:
            assert runner.run_video(video).status == StepStatus.DONE

    chaos = adapters.images
    assert isinstance(chaos, ChaosImages)
    assert chaos.stats.failures > 0 and chaos.stats.garbage > 0  # chaos really happened


def test_chaos_is_reproducible_for_a_seed(tmp_path: Path) -> None:
    def run_once(folder: Path) -> tuple[list[StepStatus], tuple[int, int, int]]:
        adapters = _chaos_adapters(HEAVY)
        with StateStore(folder / "pipeline.db") as store:
            runner = _runner(store, folder, adapters)
            statuses = [
                runner.run_video(runner.create_video("fact-of-day", topic)).status
                for topic in load_topics("fact-of-day")[:5]
            ]
        return statuses, _stats(adapters)

    assert run_once(tmp_path / "a") == run_once(tmp_path / "b")


# --- 3: garbage never reaches the disk -------------------------------------------------


@pytest.mark.parametrize("target", ["llm", "tts", "images"])
def test_pure_garbage_is_never_saved(store: StateStore, tmp_path: Path, target: str) -> None:
    clean = ChaosConfig(fail_rate=0.0, garbage_rate=0.0, seed=3)
    always_garbage = ChaosConfig(fail_rate=0.0, garbage_rate=1.0, seed=3)
    adapters = Adapters(
        llm_heavy=ChaosLlm(FakeLlm(), always_garbage if target == "llm" else clean, lambda _: None),
        llm_light=FakeLlm(),
        tts=ChaosTts(FakeTts(), always_garbage if target == "tts" else clean, lambda _: None),
        images=ChaosImages(
            FakeImages(), always_garbage if target == "images" else clean, lambda _: None
        ),
    )
    runner = _runner(store, tmp_path, adapters)
    video = runner.create_video("fact-of-day", "why octopuses have three hearts")

    report = runner.run_video(video)

    assert report.status == StepStatus.FAILED
    run_dir = tmp_path / "runs" / str(video)
    saved = {p.name for p in run_dir.rglob("*") if p.is_file()}
    forbidden = {"llm": "script.json", "tts": "voice.wav", "images": "scene_00.png"}[target]
    assert forbidden not in saved


# --- 4: crash in the middle of visuals --------------------------------------------------


class Crash(BaseException):
    """The process is killed: nothing in the pipeline may catch this."""


class CrashAfter(FakeImages):
    def __init__(self, good_calls: int) -> None:
        super().__init__()
        self.good_calls = good_calls

    def generate(self, request: ImageRequest) -> ImageResult:
        if len(self.calls) == self.good_calls:
            raise Crash
        return super().generate(request)


def test_crash_mid_visuals_resumes_without_recalling_finished_work(
    store: StateStore, tmp_path: Path
) -> None:
    crashing = Adapters(
        llm_heavy=FakeLlm(), llm_light=FakeLlm(), tts=FakeTts(), images=CrashAfter(good_calls=2)
    )
    video = _runner(store, tmp_path, crashing).create_video("fact-of-day", "how glass is made")
    with pytest.raises(Crash):
        _runner(store, tmp_path, crashing).run_video(video)

    # A new process with fresh adapters picks up where the dead one stopped.
    fresh_llm, fresh_tts, fresh_images = FakeLlm(), FakeTts(), FakeImages()
    fresh = Adapters(llm_heavy=fresh_llm, llm_light=fresh_llm, tts=fresh_tts, images=fresh_images)
    restarted = _runner(store, tmp_path, fresh)
    report = restarted.run_video(video)

    scenes = len(script.load(tmp_path / "runs" / str(video)).scenes)
    assert restarted.recovered == 1
    assert report.status == StepStatus.DONE
    assert fresh_llm.calls == []  # script was done: not asked again
    assert fresh_tts.calls == []  # voice was done: not asked again
    assert len(fresh_images.calls) == scenes - 2  # only the frames that were missing


def test_per_scene_retries_absorb_bad_images(tmp_path: Path) -> None:
    flaky = ChaosImages(
        FakeImages(), ChaosConfig(fail_rate=0.3, garbage_rate=0.2, seed=5), lambda _s: None
    )
    ctx = replace(fake_context(tmp_path / "run"), images=flaky)
    script.run(ctx)

    result = visuals.run(ctx)

    assert len(result.outputs) == len(script.load(ctx.run_dir).scenes)
    assert flaky.stats.failures + flaky.stats.garbage > 0  # some were rejected and retried
