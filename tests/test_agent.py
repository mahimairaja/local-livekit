"""Turn-level tests of the agent in main.py, on the same local LLM it runs with.

The model is tiny (135M), so these check that a turn works end to end and that the reply
is speakable, not what it says: a judge this small would only add noise. Needs the model
once (downloaded on first run); skipped when offline without it.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import pytest

from livekit.agents import AgentSession, RunResult
from livekit.plugins import hf
from main import LLM_MODEL, LLM_REVISION, LocalAgent, build_llm


@pytest.fixture(scope="module")
def model_files() -> None:
    try:
        hf.download_model(LLM_MODEL, revision=LLM_REVISION)
    except Exception as e:
        pytest.skip(f"{LLM_MODEL} not available: {e}")


@pytest.fixture
async def session(model_files: None) -> AsyncIterator[AgentSession]:
    async with AgentSession(llm=build_llm()) as session:
        await session.start(LocalAgent())
        yield session


async def test_a_turn_produces_one_spoken_reply(session: AgentSession) -> None:
    result: RunResult[None] = await session.run(user_input="Why is the sky blue?")

    reply = result.expect.next_event().is_message(role="assistant")
    result.expect.no_more_events()

    text = (reply.event().item.text_content or "").strip()
    assert text
    # A TTS reads this out: markdown would be spoken as symbols.
    assert not re.search(r"[*#`]|^\s*[-\d]+[.)]\s", text, flags=re.M), text


async def test_the_conversation_carries_over_turns(session: AgentSession) -> None:
    await session.run(user_input="Hello!")
    result: RunResult[None] = await session.run(user_input="Thanks, bye.")

    result.expect.next_event().is_message(role="assistant")
    result.expect.no_more_events()
    messages = [(m.role, m.text_content) for m in session.history.items if m.type == "message"]
    # The on_enter greeting runs concurrently with the first turn, so it can land anywhere
    # before the second one. What must hold: both user turns, in order, the last one answered.
    assert [text for role, text in messages if role == "user"] == ["Hello!", "Thanks, bye."]
    assert [role for role, _ in messages[-2:]] == ["user", "assistant"]
