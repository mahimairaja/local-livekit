"""A voice agent with no cloud dependency: every model runs on this machine.

    uv run -m livekit.agents download-files   # once, while online
    uv run python main.py console             # talk to it through your mic and speakers

STT and TTS: sherpa-onnx. LLM: a Hugging Face model run by transformers. Noise
cancellation: RNNoise.
Turn detection: LiveKit's v1-mini audio model, which runs locally on CPU.
"""

import os
from dataclasses import dataclass

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
from livekit.plugins import hf, rnnoise, sherpa

# The smallest models that hold a conversation: this repo shows the pipeline running, not
# answer quality. Swap in e.g. Qwen/Qwen2.5-0.5B-Instruct for better replies.
LLM_MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
# Keep in sync with the LIVEKIT_HF_MODELS pin in README.md (download-files).
LLM_REVISION = os.environ.get("LLM_REVISION", "12fd25f77366fa6b3b4b768ec3050bf629380bac")

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


if __name__ == "__main__":
    cli.run_app(server)
