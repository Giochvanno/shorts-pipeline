"""Live check against the real `claude -p` on your subscription. Never runs in CI.

    uv run pytest -m live

Uses the light model from config and a one-word answer, so it costs next to nothing.
"""

import shutil

import pytest

from pipeline.adapters.llm import LlmRequest
from pipeline.adapters.llm.claude_cli import ClaudeCliLlm
from pipeline.config import load_channel

pytestmark = pytest.mark.live


@pytest.fixture
def llm() -> ClaudeCliLlm:
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI is not installed")
    models = load_channel().models
    return ClaudeCliLlm(models.llm.light.primary.model, timeout_s=models.llm.timeout_s)


def test_real_claude_answers_and_reports_usage(llm: ClaudeCliLlm) -> None:
    response = llm.complete(
        LlmRequest(task="ping", system="Reply with exactly one word: ok", prompt="Go.")
    )

    assert "ok" in response.text.lower()
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.cost_usd >= 0
    print(f"\nlive claude_cli: {response.model}, cost ${response.cost_usd:.5f}")
