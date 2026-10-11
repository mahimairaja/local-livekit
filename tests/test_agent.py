"""Turn-level tests of the agent in main.py, with the same local Ollama model it runs on.

The judge is the local model too, so nothing leaves the machine. Skipped when Ollama is
not running or the model is not pulled.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import AsyncIterator

import pytest

from livekit.agents import AgentSession, RunResult, llm
from livekit.plugins import openai
from main import OLLAMA_MODEL, OLLAMA_URL, LocalAgent


def _ollama_has_model() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/models", timeout=2) as resp:
            models = json.load(resp)["data"]
    except OSError:
        return False
    return any(m["id"] == OLLAMA_MODEL for m in models)


pytestmark = pytest.mark.skipif(
    not _ollama_has_model(), reason=f"Ollama with {OLLAMA_MODEL} is not running"
)


@pytest.fixture
async def model() -> AsyncIterator[llm.LLM]:
    async with openai.LLM.with_ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL) as m:
        yield m


async def test_answers_in_short_plain_speech(model: llm.LLM) -> None:
    async with AgentSession(llm=model) as session:
        await session.start(LocalAgent())
        result: RunResult[None] = await session.run(user_input="Why is the sky blue?")

        reply = result.expect.next_event().is_message(role="assistant")
        await reply.judge(model, intent="Explains why the sky looks blue.")
        result.expect.no_more_events()

    text = reply.event().item.text_content or ""
    # It is spoken by a TTS: markdown symbols would be read out or dropped.
    assert not re.search(r"[*#`_\[\]]|^\s*[-\d]+[.)]\s", text, flags=re.M), text
    assert len(re.findall(r"[.!?](\s|$)", text)) <= 3, text


async def test_does_not_claim_to_use_the_cloud(model: llm.LLM) -> None:
    async with AgentSession(llm=model) as session:
        await session.start(LocalAgent())
        result: RunResult[None] = await session.run(user_input="Are you running in the cloud?")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(model, intent="Says it runs locally or offline on the user's computer.")
        )
        result.expect.no_more_events()
