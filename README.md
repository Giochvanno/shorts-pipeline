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

Numbers will come from `python -m pipeline soak --runs 50` once it exists.

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
uv run python -m pipeline --version
uv run python -m pipeline check-config   # validate config/*.yaml after editing
```

FFmpeg must be in PATH for the render step and its tests (`ffmpeg -version`).

## Checks (same as CI)

```powershell
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # types (strict)
uv run pytest                # tests
```

## Layout

```
assets/fonts/        Montserrat ExtraBold + its OFL licence (text is drawn by FFmpeg)
config/              channel.yaml (models, budget, upload) + series/*.yaml (style bibles)
pipeline/            core library
  config.py          typed config schema; hard rules live in code, not YAML
  steps/             idea, script, voice, visuals, captions, render, qa, publish
  adapters/          llm, tts, images, stock, youtube — each real + fake
tests/               unit, architecture, chaos and render tests
.github/workflows/   CI: lint + types on Linux, tests on Linux and Windows
```

`bot/`, `config/music.yaml` and the `Dockerfile` appear in the steps that need them.
