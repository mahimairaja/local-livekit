# livekit-plugins-sherpa

Offline speech-to-text and text-to-speech for [LiveKit Agents](https://github.com/livekit/agents), powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). ONNX Runtime only: no torch, no network at runtime.

## STT

A true streaming recognizer (zipformer transducer, about 70M parameters). It emits interim transcripts while the user speaks and a final transcript with word timings once they pause.

```python
from livekit.plugins import sherpa

session = AgentSession(stt=sherpa.STT(), ...)
```

Fetch the model once while online:

```sh
uv run -m livekit.agents download-files
```

| Option | Default | Meaning |
| --- | --- | --- |
| `model` | `zipformer-en-2023-06-26` | Built-in model, pinned to a Hugging Face commit. `zipformer-en-20M-2023-02-17` is smaller but drops the first words of an utterance on 48 kHz room audio |
| `endpoint_silence` | `0.5` | Seconds of trailing silence that end an utterance |
| `num_threads` | `2` | ONNX Runtime threads |
| `recognizer` | `None` | Bring any `sherpa_onnx.OnlineRecognizer` (endpoint detection enabled) |

Capabilities: `streaming=True`, `interim_results=True`, `aligned_transcript="word"`. The final transcript is always emitted before `END_OF_SPEECH` for the same utterance.

Word timings come from sherpa-onnx token onsets. A word ends where the next one starts.

The default model is wideband English. It does not transcribe 8 kHz telephone audio. Use an 8 kHz model through `recognizer=` for SIP.

## TTS

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) v1.0 (int8), 24 kHz.

Phonemes come from [misaki](https://github.com/hexgrad/misaki), Kokoro's own G2P: the model's `lexicon-us-en.txt` is built from misaki's `us_gold` and `us_silver` dictionaries ([sherpa-onnx script](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/kokoro/v1.0/generate_lexicon_en.py)). Words outside the lexicon fall back to espeak-ng, as in misaki itself. Both ship with the model and run inside sherpa-onnx, so there is no Python G2P and no system dependency.

```python
session = AgentSession(tts=sherpa.TTS(voice="af_heart"), ...)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `voice` | `af_heart` | Any of the 54 Kokoro v1.0 voices (`sherpa.KokoroVoice`) |
| `speed` | `1.0` | Speaking rate |
| `num_threads` | `2` | ONNX Runtime threads |
| `engine` | `None` | Bring a `sherpa_onnx.OfflineTts` with the same voice table |

Capabilities: `streaming=False`. `AgentSession` wraps it in `tts.StreamAdapter`, which splits LLM output into sentences. Each sentence's audio is pushed as sherpa-onnx produces it, and an interruption stops generation at the next chunk.

`aligned_transcript` is off: sherpa-onnx does not expose Kokoro's phoneme durations, so there are no word timings.
