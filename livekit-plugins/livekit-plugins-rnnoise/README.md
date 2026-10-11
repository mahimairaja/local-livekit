# livekit-plugins-rnnoise

Self-hosted noise cancellation for LiveKit Agents, running Xiph's [RNNoise](https://github.com/xiph/rnnoise) on-device. No API key, no LiveKit Cloud.

**Credit:** written by [Apoorva Verma](https://github.com/apoorva-01) in [livekit/agents#6477](https://github.com/livekit/agents/pull/6477), vendored here from commit `96af57c` so it can run alongside the other offline plugins while that PR is in review. Only typing and packaging were adjusted for this repo.

```python
from livekit.agents import room_io
from livekit.plugins import rnnoise

room_options = room_io.RoomOptions(
    audio_input=room_io.AudioInputOptions(noise_cancellation=rnnoise.RNNoise()),
)
```

Mono only. Frames are resampled to 48 kHz, denoised in 480-sample chunks, and returned at the input's rate and length.

The native library comes from `pyrnnoise`, which also declares `matplotlib` and `audiolab` for its CLI. They get installed but are never imported.
