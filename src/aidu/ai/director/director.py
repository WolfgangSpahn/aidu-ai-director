# src/aidu/ai/director/director.py

from __future__ import annotations

import logging
import time
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
from collections import deque
import textwrap

import requests

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

    def run(self, start_actor: str, message: dict, max_step: int = 20, console=None):

        trace = [(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)]

        mailbox = deque()
        mailbox.append((start_actor, message))

        step = 0

        while mailbox:
            step += 1

            if step > max_step:
                logger.warning("[director] maximum steps reached")
                break

            actor_name, message = mailbox.popleft()

            logger.warning(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> call actor: {actor_name} with message: {message} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            response = self.call(actor=actor_name, message=message)

            logger.warning(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Got response: {actor_name} {response['content']} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            next_actor = self.routes.get(actor_name)
            next_message = Message(role=actor_name, content=response["content"])

            if next_actor is None:
                logger.warning(f"[director] no route defined for {actor_name}")
                break

            logger.debug(f"[director] route: {actor_name} -> {next_actor}")

            mailbox.append((next_actor, next_message))
            trace.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), next_message))

            # for debgugging,
            console.print(Rule(title=f"Step {step}"))
            console.print(trace)
            input("Press Enter to continue...")


if __name__ == "__main__":
    from aidu.ai.core.context import Context
    from aidu.ai.core.belief import StudentBelief, StudentKnowledge
    from aidu.ai.agents.math_tutor import MathTutor
    from aidu.ai.agents.math_student import MathStudent
    from aidu.ai.llm.agent import EndAgent
    from aidu.ai.agents.symbolic_solver import SymbolicSolver
    from aidu.ai.llm.clients.openai import OpenAIClient
    from aidu.ai.actor.actor import Actor

    console = Console()

    from rich.logging import RichHandler

    logging.basicConfig(
        level="INFO",
        format="%(message)s - %(funcName)s",
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

    student_profile = textwrap.dedent("""\
            * Weak arithmetic skills.
            * Weak equation-solving skills.
            * Low engagement.
            * Low curiosity.
            * Low initiative.
            """)    

    student_knowledge = textwrap.dedent("""\
            * The student knows that x represents an unknown value.
            """)

    student_missing_knowledge = textwrap.dedent("""\
            * How to solve quadratic equations.
            * How to factor expressions.
            * Standard equation-solving procedures.
            * Algebraic techniques that have not already been introduced by the tutor.
            """)
    
    student_behaviors = textwrap.dedent("""\
            * Express confusion.
            * Guess an answer.
            * Ask for clarification.
            * Ask for help.
            * Respond briefly.
            * State that they do not know.
            """)
    
    forbidden_behaviors = textwrap.dedent("""\
            * Do not independently discover solution methods.
            * Do not propose multi-step solution strategies.
            * Do not isolate variables unless the tutor has already suggested doing so.
            * Do not introduce factoring, square roots, quadratic formulas, or similar techniques unless the tutor has already introduced them.
            * Do not behave like a good student.
            * Do not behave like a tutor.
            * Do not explain reasoning unless explicitly asked.
            """)
    
    student_conversation_style = textwrap.dedent("""\
            * The student usually responds with a single short sentence.
            * The student rarely volunteers information.
            * The student rarely asks follow-up questions.
            * The student answers the tutor's most recent question and then stops.
            * Typical responses are between 1 and 10 words.
            """)
    important = textwrap.dedent("""\
            * Generate a realistic student response, not an ideal student response.
            * The student must behave consistently with the provided profile and knowledge state.
            * The student should not suddenly demonstrate knowledge that is listed as unavailable.
            * The student should not introduce solution methods that have not already been introduced by the tutor.
            * The student should not make progress that is inconsistent with the student's knowledge state.
            * Prefer realistic student behavior over mathematically correct behavior.
            * If uncertain, produce the simpler and less sophisticated response.
            """)
                                

    examples = textwrap.dedent("""\
        Examples:
                                                 
            Tutor: How would you start?
            Student: I don't know.

            Tutor: What does x² mean?
            Student: Not sure.

            Tutor: Could x be 2?
            Student: Maybe.

            Tutor: Why do you think that?
            Student: Just guessing.

            Tutor: What should we do next?
            Student: No idea.            """)
    student_context = Context()
    student_agents = [
        MathStudent(
            client,
            prompt_args={
                "student_name": "Bob",
                "focus_area": "general math",
                "history": "Student just came in.",
                "student_profile": "Nothing yet.",
                "level": "beginner",
                "student_beliefs": belief.to_student_prompt(),
                "student_knowledge": knowledge.to_student_prompt(),
            },
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
        context=student_context,
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
                "history": "Student had been asked to solve the equation x**2 - 4 = 0.",
                "student_progress": "So far student guessed 3 without any reasoning, you asked to try again.",
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
        context=tutor_context,
        description="A demo math tutor actor for testing purposes.",
    )

    # ----------------------------------------------------------------------------------------
    # setting up director
    # ----------------------------------------------------------------------------------------

    director = Director()

    director.register(
        actor=math_student_actor,
        port=8001,
    )

    director.register(
        actor=math_tutor_actor,
        port=8002,
    )

    # setting up routes

    director.on_input("math_student_actor").send_to("math_tutor_actor")

    director.on_input("math_tutor_actor").send_to("math_student_actor")

    director.start()

    director.run(
        start_actor="math_student_actor",
        message={
            "role": "math_tutor_actor",
            "content": "Welcome Bob to our math tutoring session! Let's work together to solve the equation x**2 - 4 = 0. Any thoughts on how to approach this problem?",
        },
        console=console,
    )
