from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from aidu.ai.actor.actor import Actor
from aidu.ai.agents.math_tutor import MathTutor
from aidu.ai.agents.symbolic_solver import SymbolicSolver
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.context import Context, Message
from aidu.ai.director.config import SSE_CONFIG
from aidu.ai.director.director import Director
from aidu.ai.llm.agent import EndAgent, UserInput
from aidu.ai.llm.clients.openai import OpenAIClient

logger = logging.getLogger(__name__)


def build_math_tutor_director(
    client=None,
    tutor_port: int = 8002,
    user_port: int = 8004,
) -> Director:
    client = client or OpenAIClient(model="gpt-4o-mini")

    belief = StudentBelief(
        engagement=0.15,
        confidence=0.40,
        confusion=0.60,
        frustration=0.30,
        curiosity=0.10,
        self_explanation=0.05,
        guessing=0.80,
        help_seeking=0.10,
    )

    tutor_context = Context()
    tutor_agents = [
        MathTutor(
            client,
            prompt_args={
                "tutor_name": "Alice",
                "focus_area": "general math",
                "history": "Student just came in.",
                "student_progress": "We have not started yet.",
                "level": "beginner",
                "student_beliefs": belief.to_tutor_text(),
            },
        ),
        SymbolicSolver(),
        EndAgent(),
    ]
    tutor_context.create_agent_states(tutor_agents)

    math_tutor_actor = Actor(
        name="math_tutor_actor",
        agents=tutor_agents,
        startup=MathTutor,
        description="A math tutor actor for interactive tutoring workflows.",
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
        description="A text user interface actor representing the human student.",
    )

    director = Director()
    director.register(actor=math_tutor_actor, port=tutor_port)
    director.register(actor=tui_user_actor, port=user_port)
    director.on_input("tui_user_actor").send_to("math_tutor_actor")
    director.on_input("math_tutor_actor").send_to("tui_user_actor")

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

    director = build_math_tutor_director()

    if SSE_CONFIG.enabled:
        director.start_sse_server(host=SSE_CONFIG.host, port=SSE_CONFIG.port, path=SSE_CONFIG.path)

    input("Press Enter to start the math tutor director...")

    try:
        director.start()
        director.run(
            start_actor="math_tutor_actor",
            message=Message(
                role="start",
                content="Welcome to our math tutoring session. What would you like to work on today?",
            ),
        )
    finally:
        if SSE_CONFIG.enabled:
            director.stop_sse_server()


if __name__ == "__main__":
    main()
