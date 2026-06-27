from __future__ import annotations

import json
import logging
from typing import Any

from aidu.ai.actor.actor import Actor, RunRequest
from aidu.ai.agents.chem_applet_tutor import (
    ChemAppletTutor,
    RespondToAppletInputAgent,
    build_chem_applet_prompt_args,
)
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.context import Context, Message
from aidu.ai.director.director import Director
from aidu.ai.llm.agent import DebugAgent, EndAgent, WorkflowAgent
from aidu.ai.llm.clients.openai import OpenAIClient

logger = logging.getLogger(__name__)

GUI_USER_ACTOR = "gui_user_actor"
GUI_CHEM_TUTOR_ACTOR = "chem_tutor_actor"
MAX_HISTORY_TURNS = 10


class GuiUserActor(Actor):
    def __init__(self):
        super().__init__(
            name=GUI_USER_ACTOR,
            agents=[EndAgent()],
            startup=EndAgent,
            description="The GUI user participating in a director workflow.",
            avatar="Buddy",
        )


class GuiChemAppletTutor(ChemAppletTutor):
    target = EndAgent
    continuations = []


class GuiAppletInputTutor(WorkflowAgent):
    target = None
    continuations = [RespondToAppletInputAgent, GuiChemAppletTutor]

    def run(
        self,
        artifact: TextArtifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        is_applet_input = artifact.content.startswith("Applet input:")
        target = RespondToAppletInputAgent if is_applet_input else GuiChemAppletTutor
        mode = "applet_input" if is_applet_input else "typed_input"
        recommendation = self.register_recommendation(
            mode,
            target=target,
            continuations=[],
            utility=1.0,
            rationale=(
                "Applet input should be handled by the deterministic applet-response agent."
                if is_applet_input
                else "Typed dialog input should be handled by the LLM tutor."
            ),
        )
        logger.warning(
            "GuiAppletInputTutor.route mode=%s target=%s content_prefix=%r",
            mode,
            target.__name__,
            artifact.content[:160],
        )
        return self.result(artifacts=[], recommendations=[recommendation]), context


def _student_belief() -> StudentBelief:
    belief = StudentBelief()
    belief.engagement = 0.8
    belief.confusion = 0.6
    return belief


def _domain_from_context(session_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": session_context.get("domain"),
        "label": session_context.get("domain_label"),
        "description": session_context.get("domain_description", ""),
        "targets": session_context.get("domain_targets", []),
    }


def _applet_from_context(session_context: dict[str, Any]) -> dict[str, Any]:
    applet = session_context.get("applet")
    if isinstance(applet, dict) and applet.get("id"):
        return applet

    return {
        "id": session_context.get("applet_id") or session_context.get("applet_name"),
        "name": session_context.get("applet_name"),
        "description": session_context.get("applet_description", ""),
    }


def _compact_applet_input(content: str) -> str:
    if not content.startswith("Applet input:"):
        return content

    _, _, payload = content.partition("\n")
    payload = payload.strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return "Applet input: " + payload[:500]

    if not isinstance(parsed, dict):
        return f"Applet input: {parsed!r}"

    info_store = parsed.get("infoStore")
    if isinstance(info_store, dict):
        details = ", ".join(
            f"{key}={value}"
            for key, value in info_store.items()
            if value is not None
        )
        return f"Applet input: {parsed.get('applet', 'applet')} with {details}"

    return f"Applet input: {parsed.get('applet', 'applet')}"


def _clean_dialog_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = message.get("role")
    if role not in {"user", "assistant"}:
        return None

    content = str(message.get("content") or "").strip()
    if not content:
        return None

    actor = message.get("actor") or message.get("avatar") or role
    compact_content = _compact_applet_input(content)
    return {
        "role": role,
        "content": f"[{actor}] {compact_content}",
    }


def _dialog_history(messages: list[dict[str, Any]]) -> str:
    cleaned = [
        cleaned_message
        for message in messages[-MAX_HISTORY_TURNS:]
        if (cleaned_message := _clean_dialog_message(message))
    ]
    if not cleaned:
        return " - No previous dialog turns are available."

    return "\n".join(
        f" - {message['role']}: {message['content']}"
        for message in cleaned
    )


def _prompt_args(
    session_context: dict[str, Any] | None = None,
    applet_state: dict[str, Any] | str | None = None,
    history: str | None = None,
) -> dict[str, Any]:
    session_context = session_context or {}
    belief = _student_belief()

    args = build_chem_applet_prompt_args(
        tutor_name="Marie",
        level="beginner",
        history=history or " - Student just entered the GUI tutoring session.",
        student_progress=" - We have not started yet.",
        student_belief=" - " + belief.to_tutor_text(),
        domain=_domain_from_context(session_context),
        applet=_applet_from_context(session_context),
        applet_state=applet_state,
    )
    logger.debug(
        "GUI tutor prompt args domain=%s:%s applet=%s:%s state=%s remote_control=%s",
        args.get("domain_id"),
        args.get("domain_label"),
        args.get("applet_id"),
        args.get("applet_name"),
        str(args.get("applet_state", ""))[:240],
        str(args.get("applet_remote_control", ""))[:160],
    )
    return args


def _applet_state_from_content(content: str) -> dict[str, Any]:
    if not content.startswith("Applet input:"):
        logger.warning("GUI tutor applet state absent content_prefix=%r", content[:160])
        return {}

    _, _, payload = content.partition("\n")
    payload = payload.strip()
    if not payload:
        logger.warning("GUI tutor applet state empty payload")
        return {}

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("GUI tutor applet state raw payload=%r", payload[:240])
        return {"raw": payload}

    if isinstance(parsed, dict):
        logger.warning(
            "GUI tutor applet state parsed keys=%s applet=%s",
            sorted(parsed.keys()),
            parsed.get("applet"),
        )
        return parsed

    logger.warning("GUI tutor applet state parsed non-dict type=%s", type(parsed).__name__)
    return {"value": parsed}


class GuiChemTutorActor(Actor):
    def __init__(self, client=None, session_context: dict[str, Any] | None = None):
        client = client or OpenAIClient(model="gpt-5-mini")
        agents = [
            GuiAppletInputTutor(),                                             # Frontend input router.
            RespondToAppletInputAgent(),                                       # Deterministic response for applet-only turns.
            GuiChemAppletTutor(client, prompt_args=_prompt_args(session_context)),  # LLM tutor for typed dialog turns.
            DebugAgent(),                                                      # Debug/logging agent.
            EndAgent(),                                                        # Workflow terminator.
        ]
        logger.debug(
            "Creating GuiChemTutorActor startup=%s agents=%s",
            GuiAppletInputTutor.__name__,
            [agent.__class__.__name__ for agent in agents],
        )

        super().__init__(
            name=GUI_CHEM_TUTOR_ACTOR,
            agents=agents,
            startup=GuiAppletInputTutor,
            description="A GUI chemistry tutor actor.",
            avatar="Robo",
        )

    def build_context_from_request(self, req: RunRequest) -> Context:
        prior_messages = req.messages[:-1] if req.messages else []
        history = _dialog_history(prior_messages)
        logger.warning(
            "GUI tutor build_context tutor_class=%s request_domain=%s:%s request_applet=%s:%s history_turns=%s content_prefix=%r",
            GuiChemAppletTutor.__name__,
            req.session_context.get("domain"),
            req.session_context.get("domain_label"),
            req.session_context.get("applet_id"),
            req.session_context.get("applet_name"),
            len(prior_messages),
            req.content[:240],
        )
        context = Context()
        context.state.data["StudentBelief"] = _student_belief()
        context.create_agent_states(self.agents)

        applet_state = _applet_state_from_content(req.content)
        tutor_state = context.state.data.setdefault(GuiChemAppletTutor.__name__, {})
        tutor_state.update(_prompt_args(req.session_context, applet_state=applet_state, history=history))
        if prior_messages:
            context.trace.messages = [{"role": "system", "content": ""}]
            context.trace.messages.extend(
                cleaned_message
                for message in prior_messages[-MAX_HISTORY_TURNS:]
                if (cleaned_message := _clean_dialog_message(message))
            )
        logger.warning(
            "GUI tutor context ready state_class=%s domain=%s applet=%s trace_messages=%s state_keys=%s",
            GuiChemAppletTutor.__name__,
            tutor_state.get("domain_id"),
            tutor_state.get("applet_id"),
            len(context.trace.messages),
            sorted(tutor_state.keys()),
        )

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
