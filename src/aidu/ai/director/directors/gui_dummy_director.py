from __future__ import annotations

from typing import Any

from aidu.ai.actor.actor import Actor
from aidu.ai.core.context import Message
from aidu.ai.director.director import Director
from aidu.ai.llm.agent import EchoAgent, EndAgent

GUI_USER_ACTOR = "gui_user_actor"
GUI_ECHO_ACTOR = "echo_actor"


class GuiUserActor(Actor):
    def __init__(self):
        super().__init__(
            name=GUI_USER_ACTOR,
            agents=[EndAgent()],
            startup=EndAgent,
            description="The GUI user participating in a director workflow.",
            avatar="Buddy",
        )


def build_gui_dummy_director(echo_port: int = 8003) -> Director:
    echo_actor = Actor(
        name=GUI_ECHO_ACTOR,
        agents=[EchoAgent()],
        startup=EchoAgent,
        description="A GUI demo echo actor for testing purposes.",
        avatar="Robo",
    )

    director = Director()
    director.register(actor=GuiUserActor())
    director.register(actor=echo_actor, port=echo_port)
    director.on_input(GUI_USER_ACTOR).send_to(GUI_ECHO_ACTOR)
    director.on_input(GUI_ECHO_ACTOR).send_to(GUI_USER_ACTOR)

    return director


def build_session_start_message(session_id: str, context: dict[str, Any], actor: Actor) -> Message:
    username = context.get("username") or "student"
    subject = context.get("subject_label") or context.get("subject") or "learning"
    domain = context.get("domain_label") or context.get("domain") or "practice"

    return Message(
        role="assistant",
        content=f"Welcome {username} to our {subject} {domain} session.",
        actor=actor.name,
        avatar=getattr(actor, "avatar", actor.name),
        session_id=session_id,
        session_context=context,
        type="session_start",
    )
