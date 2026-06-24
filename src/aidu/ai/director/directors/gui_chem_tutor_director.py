from __future__ import annotations

import json
import re
from typing import Any

from aidu.ai.actor.actor import Actor, RunRequest
from aidu.ai.agents.chem_tutor import ChemTutor
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.context import Context, Message
from aidu.ai.director.director import Director
from aidu.ai.llm.agent import DebugAgent, EndAgent
from aidu.ai.llm.clients.openai import OpenAIClient

GUI_USER_ACTOR = "gui_user_actor"
GUI_CHEM_TUTOR_ACTOR = "chem_tutor_actor"


class GuiUserActor(Actor):
    def __init__(self):
        super().__init__(
            name=GUI_USER_ACTOR,
            agents=[EndAgent()],
            startup=EndAgent,
            description="The GUI user participating in a director workflow.",
            avatar="Buddy",
        )


class GuiChemTutor(ChemTutor):
    target = EndAgent
    continuations = []


def _student_belief() -> StudentBelief:
    belief = StudentBelief()
    belief.engagement = 0.8
    belief.confusion = 0.6
    return belief


def _prompt_args(session_context: dict[str, Any] | None = None) -> dict[str, Any]:
    session_context = session_context or {}
    subject = session_context.get("subject_label") or session_context.get("subject") or "Chemistry"
    domain = session_context.get("domain_label") or session_context.get("domain") or "Atomic Structure"
    belief = _student_belief()

    return {
        "tutor_name": "Marie",
        "focus_area": f"{subject}: {domain}",
        "level": "beginner",
        "history": " - Student just entered the GUI tutoring session.",
        "student_progress": " - We have not started yet.",
        "student_belief": " - " + belief.to_tutor_text(),
        **GuiChemTutor.default_state,
    }


def _applet_state_from_content(content: str) -> dict[str, Any]:
    if not content.startswith("Applet input:"):
        return {}

    _, _, payload = content.partition("\n")
    payload = payload.strip()
    if not payload:
        return {}

    compact_match = re.fullmatch(r"p:(?P<protons>\d+),n:(?P<neutrons>\d+),e:(?P<electrons>\d+)", payload)
    if compact_match:
        return {key: int(value) for key, value in compact_match.groupdict().items()}

    try:
        applet_state = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    state: dict[str, Any] = {}
    if "protonCount" in applet_state:
        state["protons"] = applet_state["protonCount"]
    if "neutronCount" in applet_state:
        state["neutrons"] = applet_state["neutronCount"]
    if "innerElectronCount" in applet_state or "outerElectronCount" in applet_state:
        state["electrons"] = int(applet_state.get("innerElectronCount") or 0) + int(applet_state.get("outerElectronCount") or 0)
    return state


class GuiChemTutorActor(Actor):
    def __init__(self, client=None, session_context: dict[str, Any] | None = None):
        client = client or OpenAIClient(model="gpt-5-mini")
        agents = [
            GuiChemTutor(client, prompt_args=_prompt_args(session_context)),
            DebugAgent(),
            EndAgent(),
        ]

        super().__init__(
            name=GUI_CHEM_TUTOR_ACTOR,
            agents=agents,
            startup=GuiChemTutor,
            description="A GUI chemistry tutor actor.",
            avatar="Robo",
        )

    def build_context_from_request(self, req: RunRequest) -> Context:
        context = Context()
        context.state.data["StudentBelief"] = _student_belief()
        context.create_agent_states(self.agents)

        tutor_state = context.state.data.setdefault(GuiChemTutor.__name__, {})
        tutor_state.update(_prompt_args(req.session_context))
        tutor_state.update(_applet_state_from_content(req.content))

        return context


def build_gui_chem_tutor_director(client=None, chem_tutor_port: int = 8003) -> Director:
    chem_tutor_actor = GuiChemTutorActor(client=client)

    director = Director()
    director.register(actor=GuiUserActor())
    director.register(actor=chem_tutor_actor, port=chem_tutor_port)
    director.on_input(GUI_USER_ACTOR).send_to(GUI_CHEM_TUTOR_ACTOR)
    director.on_input(GUI_CHEM_TUTOR_ACTOR).send_to(GUI_USER_ACTOR)

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
