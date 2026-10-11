"""A voice agent with no cloud dependency: every model runs on this machine.

    uv run -m livekit.agents download-files   # once, while online
    uv run python main.py console             # talk to it through your mic and speakers

STT and TTS: sherpa-onnx. LLM: a Hugging Face model run by transformers. Noise
cancellation: RNNoise.
Turn detection: LiveKit's v1-mini audio model, which runs locally on CPU.
"""

import logging
import os
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.agents.voice.io import AudioInput
from livekit.plugins import hf, rnnoise, sherpa

# The smallest models that hold a conversation: this repo shows the pipeline running, not
# answer quality. Swap in e.g. Qwen/Qwen2.5-0.5B-Instruct for better replies.
LLM_MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
# Keep in sync with the LIVEKIT_HF_MODELS pin in README.md (download-files).
LLM_REVISION = os.environ.get("LLM_REVISION", "12fd25f77366fa6b3b4b768ec3050bf629380bac")

logger = logging.getLogger("local-agent")

INSTRUCTIONS = (
    "You are a friendly voice assistant that runs fully offline on the user's computer. "
    "Answer in one or two short sentences. Use plain spoken English: "
    "no emojis, markdown, lists or special characters."
)


class LocalAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the user in one short sentence.")


def build_llm() -> hf.LLM:
    return hf.LLM(model=LLM_MODEL, revision=LLM_REVISION, max_new_tokens=80)


@dataclass
class Models:
    stt: sherpa.STT
    tts: sherpa.TTS
    llm: hf.LLM


def load_models() -> Models:
    llm = build_llm()
    llm.load()
    return Models(stt=sherpa.STT(), tts=sherpa.TTS(voice="af_heart"), llm=llm)


def prewarm(proc: JobProcess) -> None:
    # Loading the models takes seconds; do it once per process, not per call.
    proc.userdata["models"] = load_models()


server = AgentServer(setup_fnc=prewarm)

SILENCE_CHECK_S = 2.0


class MicSilenceGuard(AudioInput):
    """Passes audio through; warns once if the first seconds are exact digital zeros.

    A real microphone always has a noise floor. All-zero samples mean the OS gave the
    process silence instead, which on macOS is what a denied microphone permission does.
    """

    def __init__(self, source: AudioInput) -> None:
        super().__init__(label="MicSilenceGuard", source=source)
        self._checked_s = 0.0
        self._heard_signal = False

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await super().__anext__()
        if self._checked_s < SILENCE_CHECK_S and not self._heard_signal:
            self._heard_signal = any(frame.data)
            self._checked_s += frame.duration
            if self._checked_s >= SILENCE_CHECK_S and not self._heard_signal:
                logger.warning(
                    "microphone input is exact digital silence; on macOS this usually means "
                    "the terminal app has no microphone permission. Enable it in System "
                    "Settings > Privacy & Security > Microphone, then restart the terminal."
                )
        return frame


def build_session(models: Models) -> AgentSession:
    return AgentSession(
        stt=models.stt,
        llm=models.llm,
        tts=models.tts,
        turn_handling=TurnHandlingOptions(
            # Both pinned to local models; the defaults try LiveKit Inference first.
            turn_detection=inference.TurnDetector(version="v1-mini"),
            interruption={"mode": "vad"},
        ),
    )


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = build_session(ctx.proc.userdata["models"])
    await session.start(
        agent=LocalAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(noise_cancellation=rnnoise.RNNoise()),
        ),
    )
    if session.input.audio is not None:
        session.input.audio = MicSilenceGuard(session.input.audio)


if __name__ == "__main__":
    cli.run_app(server)
