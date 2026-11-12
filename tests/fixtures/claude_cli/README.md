Real output of `claude -p --output-format json`, Claude Code 2.1.322 on Windows,
recorded 2026-11-12 (session ids zeroed). Contract tests parse these exact bytes,
so a format change in a future CLI version shows up as a failing test.

- `success.json`: `claude -p "Reply with exactly: ok" --output-format json --model claude-haiku-5-5`
- `error_unknown_model.*`: same with `--model no-such-model` (exit code 1); the banner line
  was printed before the JSON.
