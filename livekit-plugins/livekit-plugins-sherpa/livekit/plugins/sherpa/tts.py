from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, get_args

import numpy as np
import numpy.typing as npt
import sherpa_onnx

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    tts,
    utils,
)
from livekit.agents.utils import is_given

from .models import TTS_MODELS, KokoroModel, KokoroVoice, TTSModels, local_model_dir

VOICES: tuple[KokoroVoice, ...] = get_args(KokoroVoice)

ChunkCallback = Callable[[npt.NDArray[np.float32], float], int]


class GeneratedAudio(Protocol):
    samples: npt.NDArray[np.float32]
    sample_rate: int


class OfflineTts(Protocol):
    """The subset of `sherpa_onnx.OfflineTts` this plugin uses."""

    @property
    def sample_rate(self) -> int: ...
    def generate(
        self, text: str, sid: int = 0, speed: float = 1.0, callback: ChunkCallback | None = None
    ) -> GeneratedAudio: ...


@dataclass
class _TTSOptions:
    voice: KokoroVoice
    speed: float


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: KokoroVoice = "af_heart",
        speed: float = 1.0,
        model: TTSModels = "kokoro-int8-multi-lang-v1_0",
        num_threads: int = 2,
        engine: OfflineTts | None = None,
    ) -> None:
        """Fully offline TTS with Kokoro, run by sherpa-onnx.

        Non-streaming: the agent wraps it in `tts.StreamAdapter`, which splits the LLM output
        into sentences. Audio of a sentence is pushed as soon as sherpa-onnx produces it.

        Args:
            voice: Kokoro voice. Fetch the model first with `download-files`.
            speed: Speaking rate. Larger is faster.
            model: Built-in model to load.
            num_threads: ONNX Runtime threads.
            engine: Use an already built `sherpa_onnx.OfflineTts` (with the same voice table)
                instead of `model`.
        """
        self._engine = engine or _load_kokoro(TTS_MODELS[model], num_threads=num_threads)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._engine.sample_rate,
            num_channels=1,
        )
        self._model_name = model if engine is None else "custom"
        self._opts = _TTSOptions(voice=voice, speed=speed)
        # sherpa-onnx makes no promise that one engine can run two generations at once.
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "sherpa-onnx"

    def update_options(
        self,
        *,
        voice: NotGivenOr[KokoroVoice] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        if not self._input_text.strip():
            output_emitter.flush()
            return

        loop = asyncio.get_running_loop()
        chunks = asyncio.Queue[bytes | None]()
        cancelled = threading.Event()

        def on_chunk(samples: npt.NDArray[np.float32], progress: float) -> int:
            loop.call_soon_threadsafe(chunks.put_nowait, _to_pcm16(samples))
            return 1 if cancelled.is_set() else 0

        def generate() -> None:
            try:
                with self._tts._lock:
                    self._tts._engine.generate(
                        self._input_text,
                        sid=VOICES.index(self._opts.voice),
                        speed=self._opts.speed,
                        callback=on_chunk,
                    )
            finally:
                loop.call_soon_threadsafe(chunks.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(generate))
        try:
            while (chunk := await chunks.get()) is not None:
                output_emitter.push(chunk)
            await task
        finally:
            # Stops sherpa-onnx at its next chunk if the agent was interrupted.
            cancelled.set()
            await utils.aio.cancel_and_wait(task)
        output_emitter.flush()


def _load_kokoro(spec: KokoroModel, *, num_threads: int) -> OfflineTts:
    root = Path(local_model_dir(spec))
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(root / spec.model),
                voices=str(root / spec.voices),
                tokens=str(root / spec.tokens),
                lexicon=str(root / spec.lexicon),
                data_dir=str(root / spec.data_dir),
                lang=spec.language,
            ),
            num_threads=num_threads,
        ),
        max_num_sentences=1,
    )
    engine: OfflineTts = sherpa_onnx.OfflineTts(config)
    return engine


def _to_pcm16(samples: npt.NDArray[np.float32]) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
