# livekit-plugins-sherpa

Offline speech-to-text and text-to-speech for [LiveKit Agents](https://github.com/livekit/agents), powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). ONNX Runtime only: no torch, no network at runtime.

## STT

A true streaming recognizer (zipformer transducer, about 70M parameters). It emits interim transcripts while the user speaks and a final transcript with word timings once they pause.

```python
from livekit.plugins import sherpa

session = AgentSession(stt=sherpa.STT(), ...)
```

Fetch models once while online. `download-files` fetches the defaults (the streaming STT and Kokoro); list others by name in `LIVEKIT_SHERPA_MODELS`:

```sh
uv run -m livekit.agents download-files
LIVEKIT_SHERPA_MODELS=moonshine-tiny-en-int8,kitten-nano-en-v0_2 uv run -m livekit.agents download-files
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

## Batch STT (Whisper, Moonshine, SenseVoice, Parakeet, ...)

`sherpa.BatchSTT` wraps a sherpa-onnx offline recognizer: it transcribes a whole utterance at once. `AgentSession` makes it live on its own, by wrapping it in `stt.StreamAdapter` with the session's VAD (a local Silero VAD by default) and transcribing each detected utterance.

```python
session = AgentSession(stt=sherpa.BatchSTT(model="moonshine-tiny-en-int8"), ...)

# any other sherpa-onnx offline model
recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=..., tokens=...)
session = AgentSession(stt=sherpa.BatchSTT(recognizer=recognizer), ...)
```

| Built-in model | Size | |
| --- | --- | --- |
| `moonshine-tiny-en-int8` (default) | ~27M, 120 MB | Moonshine tiny, English |
| `whisper-tiny.en-int8` | ~39M, 100 MB | Whisper tiny.en |

Capabilities: `streaming=False`, `interim_results=False`. No interim transcripts or word timings: the transcript arrives once the VAD ends the utterance.

## TTS

Kokoro by default, plus Kitten and Piper built in. Any other sherpa-onnx TTS model (VITS, Matcha, newer Kokoro) works through `engine=`.

| Built-in model | Size | Voices |
| --- | --- | --- |
| `kokoro-int8-multi-lang-v1_0` (default) | 82M, 114 MB, 24 kHz | 54, e.g. `af_heart` (`sherpa.KokoroVoice`) |
| `kitten-nano-en-v0_2` | ~15M, 24 MB, 24 kHz | 8, `expr-voice-2-m` ... `expr-voice-5-f` |
| `piper-en_US-amy-low` | 63 MB, 16 kHz | 1, `amy` |

Kokoro's phonemes come from [misaki](https://github.com/hexgrad/misaki), Kokoro's own G2P: the model's `lexicon-us-en.txt` is built from misaki's `us_gold` and `us_silver` dictionaries ([sherpa-onnx script](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/kokoro/v1.0/generate_lexicon_en.py)). Words outside the lexicon fall back to espeak-ng, as in misaki itself. Both ship with the model and run inside sherpa-onnx, so there is no Python G2P and no system dependency.

```python
session = AgentSession(tts=sherpa.TTS(voice="af_heart"), ...)
session = AgentSession(tts=sherpa.TTS(model="kitten-nano-en-v0_2", voice="expr-voice-3-f"), ...)

# any other sherpa-onnx TTS model: pick the voice by id, or pass its speaker names
engine = sherpa_onnx.OfflineTts(config)
session = AgentSession(tts=sherpa.TTS(engine=engine, voice=0), ...)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `voice` | the model's default | A speaker name from the model's table, or a speaker id |
| `model` | `kokoro-int8-multi-lang-v1_0` | Built-in model, pinned to a Hugging Face commit |
| `speed` | `1.0` | Speaking rate |
| `num_threads` | `2` | ONNX Runtime threads |
| `engine` | `None` | Bring any `sherpa_onnx.OfflineTts` |
| `speakers` | `None` | Speaker names of `engine`, in id order, so `voice` can be a name |

Capabilities: `streaming=False`. `AgentSession` wraps it in `tts.StreamAdapter`, which splits LLM output into sentences. Each sentence's audio is pushed as sherpa-onnx produces it, and an interruption stops generation at the next chunk.

`aligned_transcript` is off: sherpa-onnx does not expose phoneme durations, so there are no word timings.

## What is tested

Real models, in pytest: both streaming zipformers on LibriSpeech; Moonshine tiny and Whisper tiny.en on LibriSpeech, on Kokoro speech, and live through `StreamAdapter` with the Silero VAD; Kokoro, Kitten nano and Piper amy each spoken and transcribed back. Other sherpa-onnx models through `recognizer=` and `engine=` use the same interfaces but are not tested here.

Not covered: models outside sherpa-onnx (faster-whisper, whisper.cpp, XTTS and similar).
