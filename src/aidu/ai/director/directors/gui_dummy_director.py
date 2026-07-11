# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
from __future__ import annotations

from typing import Any

from aidu.ai.actor.actor import Actor
from aidu.ai.core.context import Message
from aidu.ai.director.director import Director
from aidu.ai.llm.agent import EchoAgent, EndAgent


class GuiUserActor(Actor):
    def __init__(self):
        super().__init__(
            name="gui_user_actor",
            agents=[EndAgent()],
            startup=EndAgent,
            description="The GUI user participating in a director workflow.",
            avatar="Buddy",
        )


def build_gui_dummy_director(echo_port: int = 8003) -> Director:
    gui_user_actor = GuiUserActor()
    echo_actor = Actor(
        name="echo_actor",
        agents=[EchoAgent()],
        startup=EchoAgent,
        description="A GUI demo echo actor for testing purposes.",
        avatar="Robo",
    )

    director = Director()
    director.register(actor=gui_user_actor)
    director.register(actor=echo_actor, port=echo_port)
    director.on_input(gui_user_actor.name).send_to(echo_actor.name)
    director.on_input(echo_actor.name).send_to(gui_user_actor.name)

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
