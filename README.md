# local-livekit

[LiveKit Agents](https://github.com/livekit/agents) running fully offline. Every model in the voice pipeline runs on your machine: no API keys, no cloud.

| Stage | Model | Size | Package |
| --- | --- | --- | --- |
| Noise cancellation | RNNoise | built in | `livekit-plugins-rnnoise` |
| Speech-to-text | streaming zipformer (sherpa-onnx) | 70M, int8 | `livekit-plugins-sherpa` |
| Turn detection | LiveKit `TurnDetector` v1-mini | local CPU | built into `livekit-agents` |
| LLM | SmolLM2-135M-Instruct (transformers) | 135M | `livekit-plugins-hf` |
| Text-to-speech | Kokoro v1.0 (sherpa-onnx) | 82M, int8 | `livekit-plugins-sherpa` |

Every model is the smallest that works: the point is a pipeline that runs anywhere, not answer quality. Swap the LLM with `LLM_MODEL` (e.g. `Qwen/Qwen2.5-0.5B-Instruct`).

The plugins are laid out like `livekit/agents/livekit-plugins/*`, so each one can move upstream on its own. The matching upstream PRs take one to two months to review; this repo shows them working together in the meantime.

## Run it

```sh
uv sync
LIVEKIT_HF_MODELS=HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac \
  uv run -m livekit.agents download-files   # once, online; the revision matches main.py
uv run python main.py console   # talk through your mic and speakers
```

To use it from a browser instead, run a local server (`livekit-server --dev`) and `uv run python main.py dev` with `LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret`.

## Develop

```sh
uv run pytest                      # downloads the models on first run, then runs offline
uv run pre-commit run --all-files  # ruff, mypy --strict, uv lock --check
```

## Credits

- `livekit-plugins-hf` is from [livekit/agents#7731](https://github.com/livekit/agents/pull/7731).
- `livekit-plugins-rnnoise` is [Apoorva Verma](https://github.com/apoorva-01)'s work from [livekit/agents#6477](https://github.com/livekit/agents/pull/6477).
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) by k2-fsa, [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) and [misaki](https://github.com/hexgrad/misaki) by hexgrad, [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) by Hugging Face, [RNNoise](https://github.com/xiph/rnnoise) by Xiph.
