"""GUI chemistry tutor actor.

``GuiChemTutorActor`` owns the tutor-side workflow used by the frontend
chemistry experience. It receives GUI user turns, builds prompt/context state
from the backend session payload, and starts at ``GuiInputRouter``.

The actor contains two response paths:
- applet artifacts go to ``AppletRuleResponder`` for deterministic
  applet-aware handling;
- typed text artifacts go to ``GuiChemLlmTutor`` for language-model tutoring.

The Director decides that this actor should receive GUI user turns. Once a turn
arrives here, the actor decides which internal agent should handle it.
"""

from __future__ import annotations

import logging
from typing import Any

from aidu.ai.actor.actor import Actor, RunRequest
from aidu.ai.agents.chem_applet_tutor import (
    AppletRuleResponder,
    ChemLlmTutor,
    build_chem_applet_prompt_args,
)
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.artifacts import AppletArtifact, Artifact
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.context import Context
from aidu.ai.llm.agent import BeginAgent, DebugAgent, EndAgent, WorkflowAgent
from aidu.ai.llm.clients.openai import OpenAIClient

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10
CHEM_APPLET_INFO_TEXT_KEYS = (
    "elementName",
    "elementSymbol",
    "selectedName",
    "name",
    "moleculeName",
    "selected",
    "atomicNumber",
    "valenceElectrons",
)


class GuiChemLlmTutor(ChemLlmTutor):
    target = EndAgent
    continuations = []


class GuiInputRouter(WorkflowAgent):
    """Route each GUI turn to the right tutor path.

    The GUI sends two kinds of messages into the chemistry tutor workflow:
    direct applet events and normal typed conversation. Applet events arrive as
    ``AppletArtifact`` objects and describe something that changed in the
    interactive chemistry applet, such as a slider value or selected molecule.
    These artifacts are sent to ``AppletRuleResponder`` so the applet can
    receive a predictable, deterministic response.

    Text artifacts are treated as typed student input and are sent to
    ``GuiChemLlmTutor``, the language-model tutor that can answer questions,
    explain chemistry ideas, and continue the conversation.

    In short: this class is the front door for GUI tutor messages. It does not
    answer the student itself; it decides which specialist should answer next.
    """

    target = None
    continuations = [AppletRuleResponder, GuiChemLlmTutor]

    def run(
        self,
        artifact: Artifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        is_applet_input = isinstance(artifact, AppletArtifact)
        target = AppletRuleResponder if is_applet_input else GuiChemLlmTutor
        mode = "applet_input" if is_applet_input else "typed_input"
        recommendation = self.register_recommendation(
            mode,
            target=target,
            continuations=[],
            utility=1.0,
            rationale=(
                "Applet input should be handled by the deterministic applet rule responder."
                if is_applet_input
                else "Typed dialog input should be handled by the LLM tutor."
            ),
        )
        logger.warning(
            "GuiInputRouter.route mode=%s target=%s artifact_type=%s content=%r",
            mode,
            target.__name__,
            artifact.type,
            artifact.content if is_applet_input else str(artifact.content)[:160],
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


def _applet_prompt_metadata_from_context(session_context: dict[str, Any]) -> dict[str, Any]:
    """Return the active applet metadata used to fill tutor prompt placeholders.

    This is not the applet's live infoStore/state. It describes which applet
    belongs to the current curriculum context so the tutor can talk about the
    correct interactive tool and use the right remote-control contract.
    """
    applet = session_context.get("applet")
    if isinstance(applet, dict) and applet.get("id"):
        return applet

    return {
        "id": session_context.get("applet_id") or session_context.get("applet_name"),
        "name": session_context.get("applet_name"),
        "description": session_context.get("applet_description", ""),
    }


def _clean_dialog_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    if role not in {"user", "assistant"}:
        return None

    applet_info = AppletInfo.from_message(message)
    if applet_info:
        content = applet_info.to_text(CHEM_APPLET_INFO_TEXT_KEYS)
    else:
        content = str(message.get("content") or "").strip()

    if not content:
        return None

    cleaned: dict[str, Any] = {
        "role": role,
        "content": content,
    }
    if message.get("kind") == "applet" and isinstance(message.get("applet_input"), dict):
        cleaned["kind"] = "applet"
        cleaned["applet_input"] = message["applet_input"]

    return cleaned


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
        applet=_applet_prompt_metadata_from_context(session_context),
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


def _latest_applet_info_store_from_request(req: RunRequest) -> dict[str, Any]:
    """Return the latest structured applet payload from the GUI.

    The GUI sends live applet changes as ``RunRequest.message["applet_input"]``.
    The actor already turns that payload into an ``AppletArtifact`` for routing;
    this helper keeps the same structured payload available so the LLM tutor can
    see the latest applet values when it writes its next response.
    """
    applet_input = req.message.get("applet_input")
    if isinstance(applet_input, dict):
        applet_info = AppletInfo.from_payload(applet_input)
        logger.warning(
            "GUI tutor applet state parsed keys=%s applet=%s",
            sorted(applet_info.to_state().keys()),
            applet_info.applet,
        )
        return applet_info.to_state()

    return {}


class GuiChemTutorActor(Actor):
    """Actor that runs the GUI chemistry tutoring workflow."""

    def __init__(self, client=None, session_context: dict[str, Any] | None = None):
        client = client or OpenAIClient(model="gpt-5-mini")
        agents = [
            BeginAgent(target=GuiInputRouter, interactive=True),
            GuiInputRouter(),
            AppletRuleResponder(),
            GuiChemLlmTutor(client, prompt_args=_prompt_args(session_context)),
            DebugAgent(),
            EndAgent(),
        ]
        logger.debug(
            "Creating GuiChemTutorActor startup=%s agents=%s",
            GuiInputRouter.__name__,
            [agent.__class__.__name__ for agent in agents],
        )

        super().__init__(
            name="chem_tutor_actor",
            agents=agents,
            startup=BeginAgent,
            description="A GUI chemistry tutor actor.",
            avatar="Robo",
        )

    def build_context_from_request(self, req: RunRequest) -> Context:
        session_context = req.info.session_context
        forwarded_messages = req.info.messages or []
        prior_messages = forwarded_messages[:-1] if forwarded_messages else []
        history = _dialog_history(prior_messages)
        logger.warning(
            "GUI tutor build_context tutor_class=%s request_domain=%s:%s request_applet=%s:%s history_turns=%s content_prefix=%r",
            GuiChemLlmTutor.__name__,
            session_context.get("domain"),
            session_context.get("domain_label"),
            session_context.get("applet_id"),
            session_context.get("applet_name"),
            len(prior_messages),
            str(req.message.get("content") or "")[:240],
        )
        context = Context()
        context.state.data["StudentBelief"] = _student_belief()
        context.create_agent_states(self.agents)

        applet_state = _latest_applet_info_store_from_request(req)
        tutor_state = context.state.data.setdefault(GuiChemLlmTutor.__name__, {})
        tutor_state.update(_prompt_args(session_context, applet_state=applet_state, history=history))
        if forwarded_messages:
            context.trace.messages = [
                cleaned_message
                for message in forwarded_messages[-MAX_HISTORY_TURNS:]
                if (cleaned_message := _clean_dialog_message(message))
            ]
        logger.warning(
            "GUI tutor context ready state_class=%s domain=%s applet=%s trace_messages=%s state_keys=%s",
            GuiChemLlmTutor.__name__,
            tutor_state.get("domain_id"),
            tutor_state.get("applet_id"),
            len(context.trace.messages),
            sorted(tutor_state.keys()),
        )

        return context
