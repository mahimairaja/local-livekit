"""The whole voice pipeline from main.py, audio in to audio out, with no room and no network.

Kokoro speaks a question into the session's microphone input. The session has to hear it
(sherpa STT), end the turn (local turn detector), answer (local LLM) and speak (Kokoro).
Fake audio I/O follows livekit/agents' tests/fake_io.py.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import (
    AgentSession,
    ConversationItemAddedEvent,
    UserInputTranscribedEvent,
    utils,
)
from livekit.agents.voice.io import AudioInput, AudioOutput, AudioOutputCapabilities
from livekit.plugins import hf
from livekit.plugins.sherpa.models import STT_MODELS, TTS_MODELS, download_model
from main import LLM_MODEL, LLM_REVISION, LocalAgent, Models, build_session, load_models


class Microphone(AudioInput):
    def __init__(self) -> None:
        super().__init__(label="test-mic")
        self._ch = utils.aio.Chan[rtc.AudioFrame]()

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._ch.__anext__()

    async def speak(self, audio: rtc.AudioFrame) -> None:
        """Push audio in 10 ms frames at real-time pace, as a room would."""
        step = audio.sample_rate // 100
        for frame in utils.audio.AudioByteStream(audio.sample_rate, 1, step).push(audio.data):
            self._ch.send_nowait(frame)
            await asyncio.sleep(0.01)

    async def silence(self, seconds: float, sample_rate: int = 24000) -> None:
        n = int(sample_rate * seconds)
        await self.speak(rtc.AudioFrame(b"\x00\x00" * n, sample_rate, 1, n))


class Speaker(AudioOutput):
    """Collects what the agent says; each segment "plays" instantly."""

    def __init__(self) -> None:
        super().__init__(label="test-speaker", capabilities=AudioOutputCapabilities(pause=False))
        self.frames: list[rtc.AudioFrame] = []
        self._pushed = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self._pushed == 0.0:
            self.on_playback_started(created_at=0.0)
        self.frames.append(frame)
        self._pushed += frame.duration

    def flush(self) -> None:
        super().flush()
        pushed, self._pushed = self._pushed, 0.0
        self.on_playback_finished(playback_position=pushed, interrupted=False)

    def clear_buffer(self) -> None:
        pushed, self._pushed = self._pushed, 0.0
        self.on_playback_finished(playback_position=pushed, interrupted=True)


@pytest.fixture(scope="module")
def models() -> Models:
    try:
        download_model(STT_MODELS["zipformer-en-2023-06-26"])
        download_model(TTS_MODELS["kokoro-int8-multi-lang-v1_0"])
        hf.download_model(LLM_MODEL, revision=LLM_REVISION)
    except Exception as e:
        pytest.skip(f"models not available: {e}")
    return load_models()


async def test_spoken_question_gets_a_spoken_answer(models: Models) -> None:
    question = await models.tts.synthesize("What is the capital of France?").collect()
    mic, speaker = Microphone(), Speaker()
    heard: list[str] = []
    replies: list[str] = []

    session: AgentSession = build_session(models)
    session.input.audio = mic
    session.output.audio = speaker

    @session.on("user_input_transcribed")
    def _on_transcript(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            heard.append(ev.transcript)

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "assistant":
            replies.append(ev.item.text_content or "")

    await session.start(LocalAgent())
    # Wait for the greeting to be spoken, not a fixed time: first runs warm the models up.
    for _ in range(200):
        if replies and speaker.frames:
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(1.0)
    greeting_audio = sum(f.duration for f in speaker.frames)

    await mic.silence(0.5)
    await mic.speak(question)
    await mic.silence(1.5)
    for _ in range(300):
        if len(replies) >= 2:
            break
        await asyncio.sleep(0.1)
    await session.aclose()

    assert greeting_audio > 0.5, "the greeting was not spoken"
    assert any("FRANCE" in t.upper() for t in heard), heard
    assert len(replies) >= 2 and replies[-1].strip(), replies
    assert sum(f.duration for f in speaker.frames) > greeting_audio + 0.5, "no spoken answer"
