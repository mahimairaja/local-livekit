# livekit-plugins-sherpa

Offline speech-to-text for [LiveKit Agents](https://github.com/livekit/agents), powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). ONNX Runtime only: no torch, no network at runtime.

## STT

A true streaming recognizer (zipformer transducer). It emits interim transcripts while the user speaks and a final transcript with word timings once they pause.

```python
from livekit.plugins import sherpa

session = AgentSession(stt=sherpa.STT(), ...)
```

Fetch the model once while online:

```sh
python main.py download-files
```

| Option | Default | Meaning |
| --- | --- | --- |
| `model` | `zipformer-en-2023-06-26` | Built-in model, pinned to a Hugging Face commit |
| `endpoint_silence` | `0.5` | Seconds of trailing silence that end an utterance |
| `num_threads` | `2` | ONNX Runtime threads |
| `recognizer` | `None` | Bring any `sherpa_onnx.OnlineRecognizer` (endpoint detection enabled) |

Capabilities: `streaming=True`, `interim_results=True`, `aligned_transcript="word"`. The final transcript is always emitted before `END_OF_SPEECH` for the same utterance.

Word timings come from sherpa-onnx token onsets. A word ends where the next one starts.

The default model is wideband English. It does not transcribe 8 kHz telephone audio. Use an 8 kHz model through `recognizer=` for SIP.
