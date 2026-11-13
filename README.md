# shorts-pipeline

[![CI](https://github.com/Giochvanno/shorts-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Giochvanno/shorts-pipeline/actions/workflows/ci.yml)

An AI pipeline that turns an idea into a ready-to-publish YouTube Short:
script → voice → visuals → captions → render → QA → human approval → upload.

The goal is **stability you can measure**, not a demo:

| Target (Shorts v1)                       | Status |
| ---------------------------------------- | ------ |
| ≥ 95 % of runs finish without a human    | —      |
| ≤ $0.50 per video                        | —      |
| ≤ 5 min of my time per day               | —      |

Real-provider numbers will come from `python -m pipeline soak --runs 50` in stage 2.

**Stage 1 (fakes, $0), measured by `tests/test_chaos.py`:** 20 of 20 videos finish
unattended while 40 % of all adapter calls fail or return garbage; every good answer is
paid for exactly once (120 images for 20 × 6 scenes out of 203 calls); a crash mid-step
resumes in a new process without re-calling finished work.

## Design in one paragraph

Every step is an idempotent module with its status in SQLite, so a crash resumes
from the failed step and a retry never pays twice (artifacts are cached by input
hash). Every external service sits behind a port with a real and a **fake**
adapter, so the whole pipeline runs in CI with no network and $0 spent.
The core is a plain library: bots, APIs and agents are outer layers on top.

## Quick start (Windows, PowerShell)

```powershell
# once: install uv (Python package and project manager)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/Giochvanno/shorts-pipeline.git
cd shorts-pipeline
uv sync                      # creates .venv with Python 3.12 + dev tools
uv run pytest                # tests

uv run python -m pipeline run --series fact-of-day               # -> runs/1/video.mp4
uv run python -m pipeline run --series fact-of-day --chaos 0.3   # same, with failing fakes
uv run python -m pipeline status                                 # where every video stands
uv run python -m pipeline resume --video 1                       # continue a failed video
uv run python -m pipeline run --series fact-of-day --real llm    # real Claude via `claude -p`
uv sync --extra stt                                              # once: faster-whisper (voice check)
uv run python -m pipeline run --series fact-of-day --real tts --real stt   # real voice, checked
uv run pytest -m live                                            # live checks (real services)
uv run python -m pipeline check-config   # validate config/*.yaml after editing
```

FFmpeg must be in PATH for the render step and its tests (`ffmpeg -version`).

Keys for paid providers go into `.env` in the repo root (git ignores it), one per line:
`ELEVENLABS_API_KEY=...`. Variables already set in the shell win over the file.

## Checks (same as CI)

```powershell
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # types (strict)
uv run pytest                # tests
```

## Layout

```
assets/fonts/        Montserrat ExtraBold + its OFL licence (hook and captions drawn by FFmpeg)
config/              channel.yaml (models, budget, upload) + series/*.yaml (style bibles)
pipeline/            core library
  config.py          typed config schema; hard rules live in code, not YAML
  state.py           SQLite: step statuses, restart from the failed step, crash recovery
  runner.py          step order, retries with backoff + jitter, alerts, JSON logs
  artifacts.py       atomic writes + cache keyed by input hash (never pay twice)
  costs.py           cost ledger: every adapter call recorded, budget checked before paid calls
  ass.py             word-by-word captions as ASS subtitles (libass), spoken word highlighted
  cli.py             run / resume / status / check-config
  steps/             idea, script, voice, visuals, captions, render, qa, publish
                     voice: synthesized, then checked by listening (whisper), redone if wrong
                     captions: script words + provider (or whisper) timings, aligned with a diff
  text.py            spoken form for comparing ("243" == "two hundred forty-three")
  env.py             reads secrets from .env (no dependency)
  adapters/          ports + fakes (llm, tts, images, notify) + chaos wrappers
                     http.py: stdlib HTTP behind a swappable transport (fake in tests)
                     llm/claude_cli.py: Claude via `claude -p` on the subscription
                     tts/elevenlabs.py: voice + per-character timings, cost from credits
tests/               unit, architecture, chaos and render tests
.github/workflows/   CI: lint + types on Linux, tests on Linux and Windows
```

`bot/`, `config/music.yaml` and the `Dockerfile` appear in the steps that need them.
