# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from aidu.ai.actor.actor import Actor
from aidu.ai.director.config import WEB_CONFIG
from aidu.ai.core.context import Context, Message
from aidu.ai.director.director import Director
from aidu.ai.director.serve import Server
from aidu.ai.llm.agent import EchoAgent, EndAgent, UserInput

logger = logging.getLogger(__name__)


def build_tui_echo_director(tui_port: int = 8004, echo_port: int = 8003) -> Director:
    tui_user_context = Context()

    class TuiUserInput(UserInput):
        state_key = "TuiUserInput"
        target = EndAgent
        continuations = []

    tui_user_agents = [TuiUserInput(), EndAgent()]
    tui_user_context.create_agent_states(tui_user_agents)

    tui_user_actor = Actor(
        name="tui_user_actor",
        agents=tui_user_agents,
        startup=TuiUserInput,
        description="A demo text user interface actor for testing purposes.",
    )

    echo_context = Context()
    echo_agents = [EchoAgent(), EndAgent()]
    echo_context.create_agent_states(echo_agents)

    echo_actor = Actor(
        name="echo_actor",
        agents=echo_agents,
        startup=EchoAgent,
        description="A demo echo actor for testing purposes.",
    )

    director = Director()
    director.register(actor=echo_actor, port=echo_port)
    director.register(actor=tui_user_actor, port=tui_port)
    director.on_input("tui_user_actor").send_to("echo_actor")
    director.on_input("echo_actor").send_to("tui_user_actor")

    return director


def main():
    console = Console()

    logging.basicConfig(
        level="INFO",
        format="%(funcName)s() -- %(message)s",
        handlers=[RichHandler(console=console)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    director = build_tui_echo_director()
    server = Server(
        name="dummy_director_server",
        director=director,
        start_actor="tui_user_actor",
        web_dir=WEB_CONFIG.web_dir,
        description="Web frontend for the dummy echo director.",
    )
    server.start(host=WEB_CONFIG.host, port=WEB_CONFIG.port)
    logger.info(f"Web server listening on http://{WEB_CONFIG.host}:{WEB_CONFIG.port}")

    input("Press Enter to start the echo director...")

    director.start()
    director.run(
        start_actor="tui_user_actor",
        message=Message(
            role="start",
            content="Please type something ...",
        ),
    )


if __name__ == "__main__":
    main()
