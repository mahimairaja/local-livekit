# local-livekit

[LiveKit Agents](https://github.com/livekit/agents) running fully offline. Every model in the voice pipeline runs on your machine: no API keys, no cloud.

| Stage | Runs on | Package |
| --- | --- | --- |
| Noise cancellation | RNNoise | `livekit-plugins-rnnoise` |
| Speech-to-text | sherpa-onnx, streaming zipformer | `livekit-plugins-sherpa` |
| Turn detection | LiveKit `TurnDetector` v1-mini (local CPU) | built into `livekit-agents` |
| LLM | Ollama | `livekit-plugins-openai` (`LLM.with_ollama`) |
| Text-to-speech | sherpa-onnx, Kokoro v1.0 | `livekit-plugins-sherpa` |

The plugins are laid out like `livekit/agents/livekit-plugins/*`, so each one can move upstream on its own. The matching upstream PRs take one to two months to review; this repo shows them working together in the meantime.

## Run it

```sh
brew install ollama && ollama pull llama3.2:3b   # or any model; set OLLAMA_MODEL
uv sync
uv run -m livekit.agents download-files          # once, while online
uv run python main.py console                    # talk through your mic and speakers
```

To use it from a browser instead, run a local server (`livekit-server --dev`) and `uv run python main.py dev` with `LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret`.

## Develop

```sh
uv run pytest                      # model tests download on first run; agent tests need Ollama
uv run pre-commit run --all-files  # ruff, mypy --strict, uv lock --check
```

## Credits

- `livekit-plugins-rnnoise` is [Apoorva Verma](https://github.com/apoorva-01)'s work from [livekit/agents#6477](https://github.com/livekit/agents/pull/6477).
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) by k2-fsa, [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad, [RNNoise](https://github.com/xiph/rnnoise) by Xiph.
