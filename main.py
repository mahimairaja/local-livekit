"""A voice agent with no cloud dependency: every model runs on this machine.

    uv run -m livekit.agents download-files   # once, while online
    uv run python main.py console             # talk to it through your mic and speakers

STT and TTS: sherpa-onnx. LLM: Ollama. Noise cancellation: RNNoise.
Turn detection: LiveKit's v1-mini audio model, which runs locally on CPU.
"""

import os

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
from livekit.plugins import openai, rnnoise, sherpa

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")

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


def prewarm(proc: JobProcess) -> None:
    # Loading the ONNX models takes a moment; do it once per process, not per call.
    proc.userdata["stt"] = sherpa.STT()
    proc.userdata["tts"] = sherpa.TTS(voice="af_heart")


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        stt=ctx.proc.userdata["stt"],
        llm=openai.LLM.with_ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL),
        tts=ctx.proc.userdata["tts"],
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1-mini"),
        ),
    )
    await session.start(
        agent=LocalAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(noise_cancellation=rnnoise.RNNoise()),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
