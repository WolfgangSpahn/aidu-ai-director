# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from aidu.ai.actor.actor import Actor
from aidu.ai.agents.math_student import MathStudent
from aidu.ai.archetype.archetype import archetype_dict
from aidu.ai.core.context import Context, Message
from aidu.ai.director.config import SSE_CONFIG
from aidu.ai.director.director import Director
from aidu.ai.llm.agent import EndAgent, UserInput
from aidu.ai.llm.clients.openai import OpenAIClient

logger = logging.getLogger(__name__)


def build_math_student_director(
    client=None,
    student_port: int = 8001,
    user_port: int = 8004
) -> Director:
    client = client or OpenAIClient(model="gpt-4o-mini")

    student_context = Context()
    student_agents = [
        MathStudent(
            client,
            archetype_dict["balanced_student"],
            archetype_dict["learned_helplessness"],
            0.1,
        ),
        EndAgent(),
    ]
    student_context.create_agent_states(student_agents)

    MathStudent.agent = EndAgent

    math_student_actor = Actor(
        name="math_student_actor",
        agents=student_agents,
        startup=MathStudent,
        description="A simulated math student actor for testing tutoring workflows.",
    )

    user_context = Context()

    class TuiUserInput(UserInput):
        state_key = "TuiUserInput"
        target = EndAgent
        continuations = []

    user_agents = [
        TuiUserInput(),
        EndAgent(),
    ]
    user_context.create_agent_states(user_agents)

    tui_user_actor = Actor(
        name="tui_user_actor",
        agents=user_agents,
        startup=TuiUserInput,
        description="A text user interface actor representing the human tutor.",
    )

    director = Director()
    director.register(actor=math_student_actor, port=student_port)
    director.register(actor=tui_user_actor, port=user_port)
    director.on_input("tui_user_actor").send_to("math_student_actor")
    director.on_input("math_student_actor").send_to("tui_user_actor")

    return director


def main():
    console = Console()

    logging.basicConfig(
        level="INFO",
        format="%(funcName)s() -- %(message)s",
        handlers=[RichHandler(console=console)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info(f"SSE enabled: {SSE_CONFIG.enabled}, host: {SSE_CONFIG.host}, port: {SSE_CONFIG.port}, path: {SSE_CONFIG.path}")

    director = build_math_student_director()

    if SSE_CONFIG.enabled:
        director.start_sse_server(host=SSE_CONFIG.host, port=SSE_CONFIG.port, path=SSE_CONFIG.path)

    input("Press Enter to start the math student simulation director...")

    try:
        director.start()
        director.run(
            start_actor="tui_user_actor",
            message=Message(
                role="start",
                content="Start tutoring the simulated math student.",
            ),
        )
    finally:
        if SSE_CONFIG.enabled:
            director.stop_sse_server()


if __name__ == "__main__":
    main()
