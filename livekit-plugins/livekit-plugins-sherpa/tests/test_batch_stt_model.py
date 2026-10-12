"""sherpa.STT in batch mode: real sherpa-onnx offline models (Moonshine, Whisper).

Needs the model files once (downloaded on first run); skipped when offline without them.
"""

from __future__ import annotations

import wave

import jiwer
import numpy as np
import pytest
from huggingface_hub import hf_hub_download

from livekit import rtc
from livekit.agents import inference, stt, utils
from livekit.plugins import sherpa
from livekit.plugins.sherpa.models import (
    STT_MODELS,
    TTS_MODELS,
    download_model,
)

# A real recording: LibriSpeech, shipped with the streaming model's repo.
CLIP = STT_MODELS["zipformer-en-2023-06-26"]
CLIP_TEXT = (
    "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE "
    "THE SQUALID QUARTER OF THE BROTHELS"
)
MAX_CER = 0.15


def _norm(text: str) -> str:
    """Whisper and Moonshine punctuate and use mixed case; the reference does not."""
    return " ".join("".join(c for c in text.upper() if c.isalnum() or c.isspace()).split())


def _clip() -> rtc.AudioFrame:
    path = hf_hub_download(CLIP.repo_id, "test_wavs/0.wav", revision=CLIP.revision)
    with wave.open(path) as w:
        return rtc.AudioFrame(
            w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getnframes()
        )


BATCH_MODELS: list[sherpa.STTModels] = ["moonshine-tiny-en-int8", "whisper-tiny.en-int8"]


@pytest.fixture(scope="module", params=BATCH_MODELS)
def engine(request: pytest.FixtureRequest) -> sherpa.STT:
    name: sherpa.STTModels = request.param
    try:
        download_model(STT_MODELS[name])
        download_model(CLIP)
    except Exception as e:
        pytest.skip(f"model not available: {e}")
    return sherpa.STT(model=name)


def test_declares_batch_only(engine: sherpa.STT) -> None:
    assert not engine.capabilities.streaming
    assert engine.capabilities.offline_recognize


async def test_transcribes_a_real_recording(engine: sherpa.STT) -> None:
    event = await engine.recognize([_clip()])

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    heard = _norm(event.alternatives[0].text)
    assert jiwer.cer(CLIP_TEXT, heard) <= MAX_CER, heard


async def test_transcribes_kokoro(engine: sherpa.STT) -> None:
    try:
        download_model(TTS_MODELS["kokoro-int8-multi-lang-v1_0"])
    except Exception as e:
        pytest.skip(f"kokoro not available: {e}")
    audio = await sherpa.TTS().synthesize("The weather in Toronto is sunny today.").collect()

    event = await engine.recognize([audio])  # 24 kHz in: the plugin resamples to 16 kHz

    heard = _norm(event.alternatives[0].text)
    assert jiwer.cer("THE WEATHER IN TORONTO IS SUNNY TODAY", heard) <= 0.2, heard


async def test_live_audio_through_stream_adapter_and_vad(engine: sherpa.STT) -> None:
    # What AgentSession does with a batch STT: the VAD cuts utterances, the STT transcribes each.
    adapter = stt.StreamAdapter(stt=engine, vad=inference.VAD(model="silero"))
    stream = adapter.stream()
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    for frame in utils.audio.AudioByteStream(16000, 1, samples_per_channel=160).push(
        silence + bytes(_clip().data) + silence
    ):
        stream.push_frame(frame)
    stream.end_input()
    finals = [ev.alternatives[0].text async for ev in stream if ev.type == "final_transcript"]
    await stream.aclose()

    heard = _norm(" ".join(finals))
    assert jiwer.cer(CLIP_TEXT, heard) <= MAX_CER, heard
