# src/aidu/ai/director/director.py

from __future__ import annotations

import logging
import time
from collections import deque

import requests

logger = logging.getLogger(__name__)


class RouteBuilder:
    def __init__(self, director, source: str):
        self.director = director
        self.source = source

    def send_to(self, target: str):
        self.director.routes[self.source] = target
        return self.director


class Director:
    def __init__(self):

        self.actors: dict[str, dict] = {}
        self.routes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, actor, port: int):

        self.actors[actor.name] = {
            "actor": actor,
            "port": port,
            "url": f"http://localhost:{port}",
            "thread": None,
        }

    # ------------------------------------------------------------------
    # Routing DSL
    # ------------------------------------------------------------------

    def on_input(self, actor: str):

        return RouteBuilder(self, actor)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self):

        for name, info in self.actors.items():
            logger.debug(f"starting {name} on port {info['port']}")

            thread = info["actor"].start(
                port=info["port"],
            )

            info["thread"] = thread

        # wait until all actors answer REST requests
        for name, info in self.actors.items():
            self._wait_until_ready(
                name=name,
                url=info["url"],
            )

    def _wait_until_ready(self, name: str, url: str, timeout: float = 10.0):

        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                requests.get(
                    f"{url}/docs",
                    timeout=1,
                )

                logger.debug(f"{name} ready")

                return

            except Exception:
                time.sleep(0.25)

        raise RuntimeError(f"{name} failed to start at {url}")

    # ------------------------------------------------------------------
    # REST call
    # ------------------------------------------------------------------

    def call(self, actor: str, message: dict) -> dict:

        url = self.actors[actor]["url"]

        response = requests.post(
            f"{url}/run",
            json=message,
            timeout=300,
        )

        response.raise_for_status()

        return response.json()

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def run(self, actor: str, message: dict, max_step: int = 20):

        mailbox = deque()
        mailbox.append(
            (
                actor,
                message,
            )
        )

        step = 0

        while mailbox:
            step += 1

            if step > max_step:
                logger.warning("[director] maximum steps reached")
                break

            actor_name, message = mailbox.popleft()

            logger.debug(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> call actor: {actor_name} with message: {message} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            response = self.call(actor=actor_name, message=message)

            logger.debug(f"<< Got response: {response}")

            next_actor = self.routes.get(actor_name)

            if next_actor is None:
                logger.warning(f"[director] no route defined for {actor_name}")

                break

            logger.debug(f"[director] route: {actor_name} -> {next_actor}")

            mailbox.append(
                (
                    next_actor,
                    response,
                )
            )


if __name__ == "__main__":
    from aidu.ai.agents.math_tutor import MathTutor
    from aidu.ai.agents.symbolic_solver import SymbolicSolver
    from aidu.ai.llm.clients.openai import OpenAIClient
    from aidu.ai.controller.processor import UserInputProcessor, AgentProcessor, EchoProcessor
    from aidu.ai.actor.actor import Actor

    from rich.logging import RichHandler

    logging.basicConfig(
        level="INFO",
        format="%(message)s - %(funcName)s",
        handlers=[RichHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    client = OpenAIClient(model="gpt-4o-mini")

    # setting up actors

    math_tutor_actor = Actor(
        name="math_tutor",
        processors={
            "math_tutor": AgentProcessor(MathTutor(client,target="exit")),
            "symbolic_solver": AgentProcessor(SymbolicSolver()),
            # "input": UserInputProcessor(target="math_tutor"),
        },
        startup = "math_tutor",
        show_trace=True,
        description="A demo math tutor actor for testing purposes.",
    )

    user_input_actor = Actor(
        name="user_input",
        processors={
            "input": UserInputProcessor(target="echo"),
            "echo": EchoProcessor(target="exit"),
        },
        startup = "input",
        show_trace=True,
        description="A demo user input actor that echoes back input.",
    )

    # setting up director

    director = Director()

    director.register(
        actor=user_input_actor,
        port=8001,
    )

    director.register(
        actor=math_tutor_actor,
        port=8002,
    )

    # setting up routes

    director.on_input("user_input").send_to("math_tutor")

    director.on_input("math_tutor").send_to("user_input")

    director.start()

    director.run(
        actor="user_input",
        message={
            "role": "system",
            "content": "Welcome",
        },
    )
