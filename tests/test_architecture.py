"""Architecture rules from the plan, enforced as tests.

1. The core (``pipeline/``) never imports outer layers: our bot, FastAPI, MCP.
2. Model IDs live only in ``config/``, never in code.

These tests read the source with ``ast`` and never execute it.
"""

import ast
import re
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "pipeline"

FORBIDDEN_IMPORTS = frozenset({"bot", "fastapi", "starlette", "uvicorn", "mcp"})

MODEL_ID_PATTERN = re.compile(
    r"claude-(opus|sonnet|haiku)|eleven_\w+_v\d|fal-ai/|black-forest-labs/",
    re.IGNORECASE,
)


def _core_files() -> list[Path]:
    return sorted(PACKAGE_DIR.rglob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _string_constants(tree: ast.Module) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_scan_actually_finds_core_files() -> None:
    # Guards against a silently green suite if PACKAGE_DIR ever points nowhere.
    assert (PACKAGE_DIR / "__init__.py") in _core_files()


def test_import_detector_catches_violations() -> None:
    tree = ast.parse("import fastapi.routing\nfrom mcp.server import Server\nfrom . import x")

    assert _top_level_imports(tree) == {"fastapi", "mcp"}


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.relative_to(PACKAGE_DIR).as_posix())
def test_core_does_not_import_outer_layers(path: Path) -> None:
    leaked = _top_level_imports(_parse(path)) & FORBIDDEN_IMPORTS

    assert not leaked, f"core module imports outer layer(s): {sorted(leaked)}"


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.relative_to(PACKAGE_DIR).as_posix())
def test_no_hardcoded_model_ids(path: Path) -> None:
    hits = [s for s in _string_constants(_parse(path)) if MODEL_ID_PATTERN.search(s)]

    assert not hits, f"model IDs belong in config/, found in code: {hits}"
