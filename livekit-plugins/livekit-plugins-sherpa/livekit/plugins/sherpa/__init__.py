"""Offline STT and TTS for LiveKit Agents, powered by sherpa-onnx."""

from livekit.agents import Plugin

from .log import logger
from .models import STT_MODELS, TTS_MODELS, KokoroVoice, STTModels, TTSModels, download_model
from .stt import STT, SpeechStream
from .tts import TTS, ChunkedStream
from .version import __version__

__all__ = [
    "STT",
    "TTS",
    "ChunkedStream",
    "KokoroVoice",
    "STTModels",
    "SpeechStream",
    "TTSModels",
    "__version__",
]


class SherpaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        for model in [*STT_MODELS.values(), *TTS_MODELS.values()]:
            download_model(model)


Plugin.register_plugin(SherpaPlugin())
