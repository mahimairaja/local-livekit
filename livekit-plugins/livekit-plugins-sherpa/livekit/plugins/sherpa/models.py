from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

STTModels = Literal["zipformer-en-2023-06-26"]


@dataclass(frozen=True)
class TransducerModel:
    """A streaming transducer model published on the Hugging Face Hub.

    The revision is a commit sha, so the files that get downloaded never change.
    """

    repo_id: str
    revision: str
    encoder: str
    decoder: str
    joiner: str
    tokens: str = "tokens.txt"
    language: str = "en"

    @property
    def files(self) -> tuple[str, str, str, str]:
        return (self.encoder, self.decoder, self.joiner, self.tokens)


STT_MODELS: dict[STTModels, TransducerModel] = {
    "zipformer-en-2023-06-26": TransducerModel(
        repo_id="csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        revision="672fbf1b30579d6585301139bb363f42a0ad4a24",
        encoder="encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        decoder="decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        joiner="joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    ),
}


def download_model(model: TransducerModel) -> None:
    for filename in model.files:
        hf_hub_download(model.repo_id, filename, revision=model.revision)


def resolve_model_file(model: TransducerModel, filename: str) -> str:
    """Return the local path of an already downloaded file. Never touches the network."""
    try:
        path: str = hf_hub_download(
            model.repo_id, filename, revision=model.revision, local_files_only=True
        )
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            f"sherpa model file {filename!r} from {model.repo_id} is not downloaded. "
            "Run `python main.py download-files` once while online."
        ) from e
    return path
