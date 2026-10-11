"""The guard that turns a silently denied microphone into a log line."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.voice.io import AudioInput
from main import MicSilenceGuard

RATE = 48000


class _Mic(AudioInput):
    def __init__(self, frames: list[rtc.AudioFrame]) -> None:
        super().__init__(label="test-mic")
        self._frames = iter(frames)

    async def __anext__(self) -> rtc.AudioFrame:
        try:
            return next(self._frames)
        except StopIteration:
            raise StopAsyncIteration from None


def _frames(seconds: float, level: int) -> list[rtc.AudioFrame]:
    n = RATE // 100
    pcm = np.full(n, level, dtype=np.int16).tobytes()
    return [rtc.AudioFrame(pcm, RATE, 1, n) for _ in range(int(seconds * 100))]


async def _drain(frames: list[rtc.AudioFrame]) -> list[rtc.AudioFrame]:
    return [f async for f in MicSilenceGuard(_Mic(frames))]


async def test_all_zero_input_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="local-agent"):
        out = await _drain(_frames(5.0, 0))

    assert len(out) == 500  # audio still flows through untouched
    warnings = [r for r in caplog.records if "microphone permission" in r.getMessage()]
    assert len(warnings) == 1


async def test_a_noise_floor_is_not_a_denied_mic(caplog: pytest.LogCaptureFixture) -> None:
    # One quiet sample per frame is enough: real mics are never exactly zero.
    with caplog.at_level(logging.WARNING, logger="local-agent"):
        await _drain(_frames(1.0, 0) + _frames(0.01, 3) + _frames(3.0, 0))

    assert not [r for r in caplog.records if "microphone permission" in r.getMessage()]
