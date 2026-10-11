from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

STTModels = Literal["zipformer-en-2023-06-26"]
TTSModels = Literal["kokoro-int8-multi-lang-v1_0"]

# Speaker ids of the Kokoro v1.0 voice table, in order (see the model's `speaker2id` metadata).
KokoroVoice = Literal[
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
    "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora",
    "em_alex", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex",
    "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
    "zm_yunxia", "zm_yunyang", "em_santa",
]  # fmt: skip


@dataclass(frozen=True)
class HubModel:
    """Model files on the Hugging Face Hub, pinned to a commit so they never change."""

    repo_id: str
    revision: str
    files: tuple[str, ...]
    """Glob patterns of the files the plugin needs from the repo."""


@dataclass(frozen=True)
class TransducerModel(HubModel):
    encoder: str = ""
    decoder: str = ""
    joiner: str = ""
    tokens: str = "tokens.txt"
    language: str = "en"


@dataclass(frozen=True)
class KokoroModel(HubModel):
    model: str = ""
    voices: str = "voices.bin"
    tokens: str = "tokens.txt"
    lexicon: str = "lexicon-us-en.txt"
    data_dir: str = "espeak-ng-data"
    language: str = "en"


def _transducer(repo_id: str, revision: str, suffix: str) -> TransducerModel:
    encoder, decoder, joiner = (f"{part}-{suffix}" for part in ("encoder", "decoder", "joiner"))
    return TransducerModel(
        repo_id=repo_id,
        revision=revision,
        files=(encoder, decoder, joiner, "tokens.txt"),
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
    )


STT_MODELS: dict[STTModels, TransducerModel] = {
    "zipformer-en-2023-06-26": _transducer(
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "672fbf1b30579d6585301139bb363f42a0ad4a24",
        "epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    ),
}

TTS_MODELS: dict[TTSModels, KokoroModel] = {
    "kokoro-int8-multi-lang-v1_0": KokoroModel(
        repo_id="csukuangfj/kokoro-int8-multi-lang-v1_0",
        revision="2a360693d79b88b49b88e29aec2b53577f41f206",
        files=(
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "lexicon-us-en.txt",
            "espeak-ng-data/**",
        ),
        model="model.int8.onnx",
    ),
}


def download_model(model: HubModel) -> str:
    path: str = snapshot_download(
        model.repo_id, revision=model.revision, allow_patterns=list(model.files)
    )
    return path


def local_model_dir(model: HubModel) -> str:
    """Directory of an already downloaded model. Never touches the network."""
    try:
        path: str = snapshot_download(
            model.repo_id,
            revision=model.revision,
            allow_patterns=list(model.files),
            local_files_only=True,
        )
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            f"sherpa model {model.repo_id} is not downloaded. "
            "Run `python main.py download-files` once while online."
        ) from e
    return path
