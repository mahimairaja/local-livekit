"""The default STT model, on the LibriSpeech clips shipped with it.

Transcripts are compared by character error rate, so a swap to another small model does not
rewrite every assertion. Needs the model files once (downloaded on first run); skipped when
offline without them.
"""

from __future__ import annotations

import wave

import jiwer
import numpy as np
import pytest
from huggingface_hub import hf_hub_download

from livekit import rtc
from livekit.agents import stt, utils
from livekit.plugins import sherpa
from livekit.plugins.sherpa.models import STT_MODELS, STTModels, download_model

# Input as it arrives from a LiveKit room: 48 kHz, 10 ms frames. The plugin resamples to 16 kHz.
ROOM_RATE = 48000

TRANSCRIPTS = {
    "0.wav": "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE "
    "THE SQUALID QUARTER OF THE BROTHELS",
    "1.wav": "GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER "
    "A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONOURED BOSOM TO CONNECT HER PARENT "
    "FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN",
}
# test_wavs/8k.wav is left out on purpose: band-limited telephone audio, cleanly upsampled,
# decodes to nothing with these wideband models. Telephony needs an 8 kHz model.
MAX_CER = 0.1


MODEL: STTModels = "zipformer-en-2023-06-26"


@pytest.fixture(scope="module")
def engine() -> sherpa.STT:
    try:
        download_model(STT_MODELS[MODEL])
    except Exception as e:
        pytest.skip(f"sherpa model not available: {e}")
    return sherpa.STT(model=MODEL)


def _close(expected: str, heard: str) -> bool:
    return bool(jiwer.cer(expected, heard) <= MAX_CER)


def _clip(name: str) -> rtc.AudioFrame:
    model = STT_MODELS[MODEL]
    path = hf_hub_download(model.repo_id, f"test_wavs/{name}", revision=model.revision)
    with wave.open(path) as w:
        pcm = w.readframes(w.getnframes())
        frame = rtc.AudioFrame(pcm, w.getframerate(), w.getnchannels(), w.getnframes())
    resampler = rtc.AudioResampler(frame.sample_rate, ROOM_RATE)
    return rtc.combine_audio_frames([*resampler.push(frame), *resampler.flush()])


def _silence(seconds: float) -> rtc.AudioFrame:
    n = int(ROOM_RATE * seconds)
    return rtc.AudioFrame(np.zeros(n, dtype=np.int16).tobytes(), ROOM_RATE, 1, n)


def _room_frames(audio: rtc.AudioFrame) -> list[rtc.AudioFrame]:
    bstream = utils.audio.AudioByteStream(ROOM_RATE, 1, samples_per_channel=ROOM_RATE // 100)
    return [*bstream.push(audio.data), *bstream.flush()]


async def _stream(engine: sherpa.STT, audio: list[rtc.AudioFrame]) -> list[stt.SpeechEvent]:
    stream = engine.stream()
    for chunk in audio:
        for frame in _room_frames(chunk):
            stream.push_frame(frame)
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    return events


async def test_streams_an_utterance_into_one_final_transcript(engine: sherpa.STT) -> None:
    events = await _stream(engine, [_silence(0.5), _clip("0.wav"), _silence(1.0)])

    types = [ev.type for ev in events if ev.type != stt.SpeechEventType.INTERIM_TRANSCRIPT]
    assert types == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.RECOGNITION_USAGE,
    ]
    assert any(ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT for ev in events)

    final = next(ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
    assert _close(TRANSCRIPTS["0.wav"], final.alternatives[0].text)


async def test_word_timings_follow_the_audio(engine: sherpa.STT) -> None:
    lead = 0.5
    clip = _clip("0.wav")
    events = await _stream(engine, [_silence(lead), clip, _silence(1.0)])

    data = next(ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
    words = data.alternatives[0].words
    assert words is not None
    assert words == data.alternatives[0].text.split()
    starts = [w.start_time for w in words if utils.is_given(w.start_time)]
    assert len(starts) == len(words)
    assert starts == sorted(starts)
    assert lead < starts[0] < lead + 1.0
    assert starts[-1] < lead + clip.duration


async def test_silence_between_utterances_splits_them(engine: sherpa.STT) -> None:
    events = await _stream(
        engine, [_silence(0.5), _clip("0.wav"), _silence(1.0), _clip("1.wav"), _silence(1.0)]
    )

    finals = [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert _close(TRANSCRIPTS["0.wav"], finals[0].alternatives[0].text)
    # 1.wav has pauses of its own, so it may arrive as several finals.
    assert _close(TRANSCRIPTS["1.wav"], " ".join(ev.alternatives[0].text for ev in finals[1:]))
    second_start = finals[1].alternatives[0].start_time
    assert second_start > 0.5 + _clip("0.wav").duration + 1.0


async def test_recognize_transcribes_a_whole_buffer(engine: sherpa.STT) -> None:
    event = await engine.recognize([_silence(0.5), _clip("0.wav")])

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert _close(TRANSCRIPTS["0.wav"], event.alternatives[0].text)
