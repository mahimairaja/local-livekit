# local-livekit

LiveKit Agents running fully offline. Showcase for local STT, TTS, LLM and noise cancellation while the matching upstream PRs are in review.

## Layout

- `livekit-plugins/livekit-plugins-<name>/`: one uv workspace member per plugin, laid out like `livekit/agents/livekit-plugins/*` (own `pyproject.toml`, namespace package `livekit/plugins/<name>/`, own `tests/`). A new plugin also needs an entry in root `[tool.uv.sources]` and `[tool.mypy] mypy_path`.
- `main.py`: one agent that wires every plugin together.
- Frontend/backend come later (agent-starter-react + livekit-api token helpers). Keep root free for `frontend/` and `backend/`.

## Stack

- STT and TTS: sherpa-onnx (ONNX only, no torch). Kokoro is the default TTS voice.
- LLM: `livekit-plugins-hf` (transformers, in-process). Smallest models that work: this repo proves the pipeline runs.
- Models are fetched with `Plugin.download_files()`, as silero and turn-detector do. Never commit model files.

## Rules

1. Check the docs before writing anything non-trivial. LiveKit: the `livekit-docs` MCP server (`.mcp.json`) and the skills in `.claude/skills/`. Other libraries: Context7. Read the upstream source when docs are thin.
2. Use the framework's built-in method over hand-rolled code (e.g. `livekit.api.AccessToken`, `tts.StreamAdapter`, `AudioEmitter`, `stt.SpeechEvent`).
3. Pin exact versions with `==`. Commit `uv.lock`. Pre-commit revs are exact tags.
4. Full typing. `mypy --strict` must pass. No `# type: ignore` without a reason on the same line.
5. Tests prove behaviour, not coverage. Use the framework's own test tools: `AgentSession` + `RunResult` assertions, fake STT/TTS in the style of upstream `tests/fake_stt.py`, pytest-asyncio.
6. No em dashes in code, docs or commits.
7. Never comment on or open issues/PRs in livekit/agents from this repo's work.

## Commands

```sh
uv sync                          # install
uv run pytest                    # tests
uv run pre-commit run --all-files  # ruff format, ruff check, mypy strict, uv lock --check
```

## Git

- Branches: `feat/`, `fix/`, `chore/`.
- Author: mahimairaja <mahimairaja3@gmail.com>. No AI attribution in commits or PRs.
