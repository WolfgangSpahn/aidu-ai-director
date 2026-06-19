# src/aidu/ai/director/director.py

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule

import requests

from aidu.ai.archetype.archetype import archetype_dict

logger = logging.getLogger(__name__)

from aidu.ai.core.context import Message


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
        self._event_subscribers: list[queue.Queue] = []
        self._event_lock = threading.Lock()
        self._sse_server: ThreadingHTTPServer | None = None
        self._sse_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # SSE server
    # ------------------------------------------------------------------

    def start_sse_server(self, host: str = "0.0.0.0", port: int = 8100, path: str = "/events"):

        if self._sse_server is not None:
            raise RuntimeError("SSE server is already running")

        director = self

        class SSEHandler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self):
                if self.path == "/health":
                    body = b'{"status":"ok"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path != path:
                    self.send_response(404)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                client_queue: queue.Queue = queue.Queue(maxsize=200)
                director._add_subscriber(client_queue)

                try:
                    self.wfile.write(b"event: connected\n")
                    self.wfile.write(b'data: {"status":"connected"}\n\n')
                    self.wfile.flush()

                    while True:
                        try:
                            event = client_queue.get(timeout=15)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue

                        event_name = event.get("event", "message")
                        payload = json.dumps(event.get("data", {}), default=str)

                        self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
                        for payload_line in payload.splitlines() or [payload]:
                            self.wfile.write(f"data: {payload_line}\n".encode("utf-8"))
                        self.wfile.write(b"\n")
                        self.wfile.flush()

                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    director._remove_subscriber(client_queue)

            def log_message(self, format: str, *args):
                logger.debug("[director-sse] " + format, *args)

        self._sse_server = ThreadingHTTPServer((host, port), SSEHandler)
        self._sse_thread = threading.Thread(target=self._sse_server.serve_forever, daemon=True)
        self._sse_thread.start()
        logger.info(f"[director] SSE server listening on http://{host}:{port}{path}")

    def stop_sse_server(self):

        if self._sse_server is None:
            return

        self._sse_server.shutdown()
        self._sse_server.server_close()
        self._sse_server = None

        if self._sse_thread is not None:
            self._sse_thread.join(timeout=2)
            self._sse_thread = None

    def _add_subscriber(self, subscriber: queue.Queue):

        with self._event_lock:
            self._event_subscribers.append(subscriber)

    def _remove_subscriber(self, subscriber: queue.Queue):

        with self._event_lock:
            self._event_subscribers = [q for q in self._event_subscribers if q is not subscriber]

    def _publish_event(self, event: str, data: dict[str, Any]):

        payload = {"event": event, "data": data}

        with self._event_lock:
            subscribers = list(self._event_subscribers)

        for subscriber in subscribers:
            try:
                subscriber.put_nowait(payload)
            except queue.Full:
                logger.debug("[director] dropping SSE event for a slow subscriber")

    def _publish_message(self, actor: str, message: Message):
        payload = self._serialize_message(actor=actor, message=message)
        self._publish_event(
            event="message",
            data={**payload, "message": payload},
        )

    @staticmethod
    def _serialize_message(actor: str, message: Message) -> dict[str, Any]:
        payload: dict[str, Any]
        if hasattr(message, "model_dump"):
            payload = message.model_dump()
        elif isinstance(message, dict):
            payload = dict(message)
        else:
            payload = {"content": str(message)}

        # Keep both actor and avatar for compatibility with existing consumers.
        payload["actor"] = actor
        payload["avatar"] = actor
        payload.setdefault("role", "assistant")
        payload.setdefault("content", "")
        return payload

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
            logger.info(f"starting {name} on port {info['port']}")

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

    def run(self, start_actor: str, message: Message, max_step: int = 5, console=None, interactive: bool = False):

        trace = [(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)]
        self._publish_message(start_actor, message)

        mailbox = deque()
        mailbox.append((start_actor, message))

        step = 0
        while mailbox:
            
            step += 1

            if step > max_step:
                logger.debug("[director] maximum steps reached")
                self._publish_event(
                    event="run_stopped",
                    data={
                        "reason": "max_step_reached",
                        "step": step,
                    },
                )
                break

            actor_name, message = mailbox.popleft()
            logger.info(f"[director] step {step}, mailbox len: {len(mailbox)}: calling {actor_name}")


            response = self.call(actor=actor_name, message=message)

            next_actor = self.routes.get(actor_name)
            next_message = Message(role="assistant" if "user" not in actor_name else "user", content=response["content"])

            self._publish_message(next_actor, next_message)

            if next_actor is None:
                logger.debug(f"[director] no route defined for {actor_name}")
                break

            logger.debug(f"[director] route: {actor_name} -> {next_actor}")

            mailbox.append((next_actor, next_message))
            trace.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), next_message))

            
            # for debgugging,
            if console is not None:
                console.print(Rule(title=f"Step {step}"))
                console.print(trace)

            if interactive:
                input("Press Enter to continue...")

        return trace


if __name__ == "__main__":
    from aidu.ai.core.context import Context
    from aidu.ai.core.belief import StudentBelief, StudentKnowledge
    from aidu.ai.agents.math_tutor import MathTutor
    from aidu.ai.agents.math_student import MathStudent
    from aidu.ai.llm.agent import EndAgent, UserInput, EchoAgent
    from aidu.ai.agents.symbolic_solver import SymbolicSolver
    from aidu.ai.llm.clients.openai import OpenAIClient
    from aidu.ai.actor.actor import Actor
    from aidu.ai.actor.frontend_actor import FrontendActor

    console = Console()

    from rich.logging import RichHandler

    logging.basicConfig(
        level="INFO",
        format="%(funcName)s() -- %(message)s",
        handlers=[RichHandler(console=console)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # -----------------------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------------------

    client = OpenAIClient(model="gpt-4o-mini")

    # Initialize belief state, the same for both ftb
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

    knowledge = StudentKnowledge(
        arithmetic=0.20,
        fractions=0.10,
        equations=0.20,
        functions=0.10,
        derivatives=0.00,
        integrals=0.00,
    )

    # ----------------------------------------------------------------------------------------
    # Math student actor
    # ----------------------------------------------------------------------------------------
    
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

    for agent in student_agents:
        student_context.state.data.setdefault(
            agent.__class__.__name__,
            getattr(agent, "default_state", {}).copy(),
        )

    MathStudent.agent = EndAgent

    # console.print("Routes",get_recommendation_data(agents))

    math_student_actor = Actor(
        name="math_student_actor",
        agents=student_agents,
        startup=MathStudent,
        description="A demo math student actor for testing purposes.",
    )

    # ----------------------------------------------------------------------------------------
    # Math student actor
    # ----------------------------------------------------------------------------------------
    tutor_context = Context()

    tutor_agents = [
        MathTutor(
            client,
            prompt_args={
                "tutor_name": "Alice",
                "focus_area": "general math",
                "history": "Student just come in.",
                "student_progress": "We have not started yet.",
                "level": "beginner",
                "student_beliefs": belief.to_tutor_text(),
            },
        ),
        SymbolicSolver(),
        EndAgent(),
    ]

    for agent in tutor_agents:
        tutor_context.state.data.setdefault(
            agent.__class__.__name__,
            getattr(agent, "default_state", {}).copy(),
        )

    math_tutor_actor = Actor(
        name="math_tutor_actor",
        agents=tutor_agents,
        startup=MathTutor,
        # context=tutor_context,
        description="A demo math tutor actor for testing purposes.",
    )

    # ----------------------------------------------------------------------------------------
    # Text User interface actor
    # ----------------------------------------------------------------------------------------

    tui_user_context = Context()

    class TuiUserInput(UserInput):
        state_key = "TuiUserInput"  # store user input in context.state.data["TuiUserInput"]
        target = EndAgent
        continuations = []

    tui_user_agents = [
        TuiUserInput(),
        EndAgent(),
    ]


    tui_user_context.create_agent_states(tui_user_agents)

    tui_user_actor = Actor(
        name="tui_user_actor",
        agents=tui_user_agents,
        startup=TuiUserInput,
        # context=tui_user_context,
        description="A demo user interface actor for testing purposes.",
    )


    # ----------------------------------------------------------------------------------------
    # Graphical User interface actor
    # ----------------------------------------------------------------------------------------



    gui_user_actor = FrontendActor(
        director_url="http://localhost:8100",
        name="gui_user_actor",
        description="A demo graphical user interface actor for testing purposes.",
    )

    # ----------------------------------------------------------------------------------------
    # Echo actor for testing
    # ----------------------------------------------------------------------------------------

    echo_context = Context()

    echo_agents = [
        EchoAgent(),
        EndAgent(),
    ]

    echo_context.create_agent_states(echo_agents)

    echo_actor = Actor(
        name="echo_actor",
        agents=echo_agents,
        startup=EchoAgent,
        # context=echo_context,
        description="A demo echo actor for testing purposes.",
    )

    # ----------------------------------------------------------------------------------------
    # setting up director 1
    # ----------------------------------------------------------------------------------------

    # director1 = Director()

    # director1.register(
    #     actor=math_student_actor,
    #     port=8001,
    # )

    # director1.register(
    #     actor=math_tutor_actor,
    #     port=8002,
    # )

    # # setting up routes

    # director1.on_input("math_student_actor").send_to("math_tutor_actor")

    # director1.on_input("math_tutor_actor").send_to("math_student_actor")

    sse_enabled = os.getenv("AIDU_DIRECTOR_SSE", "1") == "1"
    sse_host = os.getenv("AIDU_DIRECTOR_SSE_HOST", "127.0.0.1")
    sse_port = int(os.getenv("AIDU_DIRECTOR_SSE_PORT", "8100"))
    sse_path = os.getenv("AIDU_DIRECTOR_SSE_PATH", "/events")

    logger.info(f"SSE enabled: {sse_enabled}, host: {sse_host}, port: {sse_port}, path: {sse_path}")

    # if sse_enabled:
    #     director1.start_sse_server(host=sse_host, port=sse_port, path=sse_path)

    # # wait until user presses enter to start the director
    # input("Press Enter to start the director...")

    # try:
    #     director1.start()

    #     director1.run(
    #         start_actor="math_student_actor",
    #         message=Message(
    #             role="start",
    #             content="Welcome Bob to our math tutoring session! Let's start to find out where you are in your math journey. Can you tell me a bit about your experience with math and what topics you feel confident in, as well as areas where you might need some help?",
    #         ),
    #         # console=console,
    #     )
    # finally:
    #     if sse_enabled:
    #         director1.stop_sse_server()

    # -----------------------------------------------------------------------------------------
    # setting up director 2 with echo agent for testing
    # -----------------------------------------------------------------------------------------

    director2 = Director()

    director2.register(
        actor=echo_actor,
        port=8003,
    )

    director2.register(
        actor=tui_user_actor,
        port=8004,
    )

    director2.on_input("tui_user_actor").send_to("echo_actor")

    director2.on_input("echo_actor").send_to("tui_user_actor")

    if sse_enabled:
        director2.start_sse_server(host=sse_host, port=sse_port, path=sse_path)

    input("Press Enter to start the echo director...")

    try:
        director2.start()

        director2.run(
            start_actor="tui_user_actor",
            message=Message(
                role="start",
                content="Please type something ...",
            ),
            # console=console,
        )
    finally:
        if sse_enabled:
            director2.stop_sse_server()
