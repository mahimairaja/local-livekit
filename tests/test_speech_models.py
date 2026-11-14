"""Which speech models main.py builds: sherpa by default, Hugging Face on request."""

from __future__ import annotations

from typing import Any

import pytest

import main
from livekit.plugins import hf, sherpa


def test_defaults_to_sherpa(monkeypatch: pytest.MonkeyPatch) -> None:
    # sherpa's constructors need the model files, so record the calls instead
    built: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(sherpa, "STT", lambda **kw: built.append(("stt", kw)))
    monkeypatch.setattr(sherpa, "TTS", lambda **kw: built.append(("tts", kw)))

    main.build_stt()
    main.build_tts()

    assert built == [("stt", {}), ("tts", {"voice": "af_heart"})]


def test_sunflower_gets_its_language_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "STT_MODEL", "Sunbird/SunflowerASR-51-african-languages")
    monkeypatch.setattr(main, "STT_LANGUAGE", "lug")

    model = main.build_stt()

    assert isinstance(model, hf.STT)
    assert model.model == "Sunbird/SunflowerASR-51-african-languages"
    assert model._opts.language == "lug"
    assert model._opts.language_token_ids["lug"] == 50332


def test_other_hf_stt_models_get_no_extra_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "STT_MODEL", "openai/whisper-large-v3-turbo")

    model = main.build_stt()

    assert isinstance(model, hf.STT)
    assert model._opts.language_token_ids == {}


def test_hf_tts_takes_model_and_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "TTS_MODEL", "Sunbird/orpheus-3b-tts-multilingual")
    monkeypatch.setattr(main, "TTS_VOICE", "salt_lug_0001")

    model = main.build_tts()

    assert isinstance(model, hf.TTS)
    assert model._opts.voice == "salt_lug_0001"
