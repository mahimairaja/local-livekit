"""Offline STT and TTS for LiveKit Agents, powered by sherpa-onnx."""

import os

from livekit.agents import Plugin

from .batch_stt import BatchSTT
from .log import logger
from .models import (
    BATCH_STT_MODELS,
    STT_MODELS,
    TTS_MODELS,
    BatchSTTModels,
    HubModel,
    KokoroVoice,
    STTModels,
    TTSModels,
    download_model,
)
from .stt import STT, SpeechStream
from .tts import TTS, ChunkedStream, Voice
from .version import __version__

__all__ = [
    "STT",
    "TTS",
    "BatchSTT",
    "BatchSTTModels",
    "ChunkedStream",
    "KokoroVoice",
    "STTModels",
    "SpeechStream",
    "TTSModels",
    "Voice",
    "__version__",
]

MODELS_ENV_VAR = "LIVEKIT_SHERPA_MODELS"
DEFAULT_MODELS = ("zipformer-en-2023-06-26", "kokoro-int8-multi-lang-v1_0")


def models_to_download() -> list[HubModel]:
    """Built-in models fetched by `download-files`: the defaults, or `LIVEKIT_SHERPA_MODELS`
    (comma separated names, e.g. "moonshine-tiny-en-int8,kitten-nano-en-v0_2")."""
    registry: dict[str, HubModel] = {
        name: spec
        for table in (STT_MODELS, BATCH_STT_MODELS, TTS_MODELS)
        for name, spec in table.items()
    }
    names = [n.strip() for n in os.environ.get(MODELS_ENV_VAR, "").split(",") if n.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown {MODELS_ENV_VAR} entries {unknown}; known: {list(registry)}")
    return [registry[n] for n in names or DEFAULT_MODELS]


class SherpaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        for model in models_to_download():
            download_model(model)


Plugin.register_plugin(SherpaPlugin())
