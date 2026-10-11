"""Offline STT and TTS for LiveKit Agents, powered by sherpa-onnx."""

from livekit.agents import Plugin

from .log import logger
from .models import STT_MODELS, STTModels, TransducerModel, download_model
from .stt import STT, SpeechStream
from .version import __version__

__all__ = ["STT", "STTModels", "SpeechStream", "TransducerModel", "__version__"]


class SherpaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        for model in STT_MODELS.values():
            download_model(model)


Plugin.register_plugin(SherpaPlugin())
