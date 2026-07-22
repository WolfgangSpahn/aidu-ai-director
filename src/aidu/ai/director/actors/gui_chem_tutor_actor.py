# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
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

import json
import logging
import os
import copy
from typing import Any

from aidu.ai.actor.actor import Actor, RunRequest
from aidu.ai.actor.turn_scope import JoinEndAgent, get_turn_side_tasks
from aidu.ai.agents.chem_assessor import ChemAssessor
from aidu.ai.agents.chem_applet_tutor import (
    AppletRuleResponder,
    ChemLlmTutor,
    build_chem_applet_prompt_args,
)
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.artifacts import AppletArtifact, Artifact
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context
from aidu.ai.llm.agent import BeginAgent, DebugAgent, EndAgent, WorkflowAgent
from aidu.ai.llm.agent_runner import run_agent_text_turn
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

DEBUG_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
PROGRESS_TARGET_ALIASES = {
    "atomic-particles": "neutron-identity",
}
PROGRESS_META_KEYS = {
    "progress_update_count",
    "progress_update_indicator",
}
INITIAL_NEGATIVE_EVIDENCE = 4.0
EVIDENCE_WEIGHTS = {
    "w": 1.0,
    "m": 2.0,
    "s": 4.0,
}


Strength = str
Polarity = str


def _canonical_progress_target_id(target_id: str) -> str:
    return PROGRESS_TARGET_ALIASES.get(target_id, target_id)


class GuiChemLlmTutor(ChemLlmTutor):
    target = JoinEndAgent
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

    def __init__(self, *, client=None, session_context: dict[str, Any] | None = None):
        self.client = client
        self.session_context = session_context or {}

    def run(
        self,
        artifact: Artifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        is_applet_input = isinstance(artifact, AppletArtifact)
        target = AppletRuleResponder if is_applet_input else GuiChemLlmTutor
        mode = "applet_input" if is_applet_input else "typed_input"
        if not is_applet_input and isinstance(artifact.content, str):
            side = get_turn_side_tasks(context)
            assessor_context = copy.deepcopy(context)
            assessor_context.control.data.pop("stream_callback", None)
            current_turn = _current_turn_text(artifact)
            prompt_params = _chem_assessor_prompt_args(
                context=assessor_context,
                session_context=self.session_context,
                current_turn=current_turn,
            )
            logger.debug(
                "ChemAssessor.spawn indicators=%s current_turn=%r",
                prompt_params.get("valid_indicators"),
                current_turn[:160],
            )
            side.spawn(
                "chem_assessor",
                lambda: run_chem_assessor_sync(
                    client=self.client,
                    prompt_params=prompt_params,
                    context=assessor_context,
                ),
                on_result=lambda assessment, join_context: apply_chem_assessment(
                    assessment=assessment,
                    context=join_context,
                ),
            )
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
        logger.debug(
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


def _student_progress_from_context(session_context: dict[str, Any]) -> dict[str, Any]:
    domain = _domain_from_context(session_context)
    targets = domain.get("targets")
    progress_by_target: dict[str, float] = {}
    if isinstance(targets, list):
        for target in targets:
            if not isinstance(target, dict):
                continue
            target_id = _canonical_progress_target_id(str(target.get("id") or "").strip())
            if not target_id or target_id in PROGRESS_META_KEYS:
                continue
            progress_by_target[target_id] = {
                "mastery": 0.0,
                "positive_evidence": 0.0,
                "negative_evidence": INITIAL_NEGATIVE_EVIDENCE,
            }

    return progress_by_target


def _student_goal_from_context(session_context: dict[str, Any]) -> dict[str, Any]:
    applet_id = str(session_context.get("applet_id") or "")
    if applet_id != "applet-build-an-atom":
        return {}
    return {
        "goal_to_targets": {
            "start_with_protons": ["proton-identity"],
            "build_nucleus": ["neutron-identity"],
            "identify_element": ["proton-identity"],
            "neutrality": ["electron-ions"],
            "charge_balance": ["electron-ions"],
            "electron_shells": ["electron-arrangement"],
            "mass_number": ["atomic-number-mass-isotopes"],
            "isotope": ["isotope-notation", "neutron-isotopes"],
            "stability": ["neutron-isotopes"],
            "reflection": ["neutron-identity"],
            "observe": ["neutron-identity"],
        }
    }


def _normalize_student_progress(progress: Any) -> dict[str, dict[str, float]]:
    """Validate the evidence-backed target progress emitted by the actor."""
    if not isinstance(progress, dict):
        return {}

    normalized: dict[str, dict[str, float]] = {}
    for key, value in progress.items():
        target_id = _canonical_progress_target_id(str(key))
        if target_id in PROGRESS_META_KEYS or not isinstance(value, dict):
            continue
        mastery = value.get("mastery")
        positive = value.get("positive_evidence")
        negative = value.get("negative_evidence")
        if not all(isinstance(item, (int, float)) for item in (mastery, positive, negative)):
            continue
        normalized[target_id] = {
            "mastery": max(0.0, min(1.0, float(mastery))),
            "positive_evidence": max(0.0, float(positive)),
            "negative_evidence": max(0.0, float(negative)),
        }
    return normalized


def _latest_backend_progress_state(messages: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        state = message.get("backend_progress_state")
        normalized = _normalize_student_progress(state)
        if normalized:
            return normalized
    return {}


def _valid_indicators_from_context(context: Context, session_context: dict[str, Any]) -> list[str]:
    progress = _normalize_student_progress(context.state.data.get("StudentProgress"))
    if progress:
        return list(progress.keys())
    return list(_student_progress_from_context(session_context).keys())


def _current_turn_text(artifact: Artifact) -> str:
    return f"Student: {artifact.content}"


def _last_tutor_turn_from_context(context: Context) -> str:
    """Return the tutor utterance that the current student answer responds to.

    Virtual students commonly send an applet event between the tutor's prompt
    and their spoken answer. Selecting the last two raw trace messages therefore
    loses the tutor prompt and leaves terse answers such as ``6`` impossible to
    assess. Walk backward to the latest real assistant dialog instead.
    """
    for message in reversed(context.trace.messages[-MAX_HISTORY_TURNS:]):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        cleaned = _clean_dialog_message(message)
        if cleaned and str(cleaned.get("content") or "").strip():
            return f"Tutor: {cleaned['content']}"
    return "No previous tutor turn."


def _current_applet_state_from_context(context: Context) -> Any:
    state = context.state.data.get(GuiChemLlmTutor.__name__, {})
    if not state:
        state = context.state.data.get(ChemLlmTutor.__name__, {})
    return state.get("applet_state") or {}


def _chem_assessor_prompt_args(
    *,
    context: Context,
    session_context: dict[str, Any],
    current_turn: str,
) -> dict[str, Any]:
    return {
        "valid_indicators": _valid_indicators_from_context(context, session_context),
        "last_turn": _last_tutor_turn_from_context(context),
        "current_turn": current_turn,
        "current_applet_state": _current_applet_state_from_context(context),
    }


def run_chem_assessor_sync(
    *,
    client,
    prompt_params: dict[str, Any],
    context: Context,
) -> dict[str, Any]:
    logger.debug("ChemAssessor.start")
    ChemAssessor.target = EndAgent
    assessor = ChemAssessor(client=client or OpenAIClient(model="gpt-5-mini"))
    result, _ = run_agent_text_turn(
        starting_agent=assessor,
        user_text="Assess the current chemistry learning evidence.",
        context=context,
        agents=[assessor, EndAgent()],
        prompt_params=prompt_params,
        ask_config=AskConfig(
            json_mode=True,
            max_tokens=512,
            vendor_config={"reasoning": {"effort": "minimal"}, "verbosity": "low"},
        ),
    )
    content = result.content()
    try:
        assessment = json.loads(content)
        logger.debug("ChemAssessor.done assessment=%s", assessment)
        return assessment
    except json.JSONDecodeError:
        logger.warning("ChemAssessor returned non-JSON content: %r", content)
        return {"e": [], "review": True, "raw": content}


def apply_chem_assessment(*, assessment: dict[str, Any], context: Context) -> None:
    progress = context.state.data.get("StudentProgress")
    if not isinstance(progress, dict):
        logger.debug("ChemAssessor.apply skipped reason=no_student_progress")
        return

    evidence = assessment.get("e")
    if not isinstance(evidence, list):
        logger.debug("ChemAssessor.apply skipped reason=no_evidence assessment=%s", assessment)
        return

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        indicator = str(item.get("i") or "").strip()
        if indicator not in progress:
            skipped.append({"indicator": indicator, "reason": "missing_progress_key"})
            continue
        target_state = progress.get(indicator)
        if not isinstance(target_state, dict):
            skipped.append({"indicator": indicator, "reason": "invalid_evidence_state"})
            continue
        polarity = str(item.get("p") or "?")
        if polarity not in {"+", "-"}:
            skipped.append({"indicator": indicator, "reason": "zero_delta"})
            continue
        weight = EVIDENCE_WEIGHTS.get(str(item.get("s") or "w"), 1.0)
        positive = max(0.0, float(target_state.get("positive_evidence", 0.0) or 0.0))
        negative = max(0.0, float(target_state.get("negative_evidence", INITIAL_NEGATIVE_EVIDENCE) or 0.0))
        prior = positive / (positive + negative) if positive + negative else 0.0
        if polarity == "+":
            positive += weight
        else:
            negative += weight
        posterior = positive / (positive + negative) if positive + negative else 0.0
        target_state.update({
            "mastery": posterior,
            "positive_evidence": positive,
            "negative_evidence": negative,
        })
        applied.append({
            "indicator": indicator,
            "prior": prior,
            "weight": weight,
            "polarity": polarity,
            "posterior": posterior,
        })

    if applied:
        context.control.data["chem_assessor_evidence"] = assessment
        logger.debug("ChemAssessor progress applied: %s", applied)
    else:
        logger.debug("ChemAssessor.apply no_progress_change skipped=%s assessment=%s", skipped, assessment)


def _student_progress_tutor_text(student_progress: dict[str, Any]) -> str:
    if not student_progress:
        return " - We have not started yet."

    return (
        f" - Learning targets loaded: {len(student_progress)}. "
        "Each domain target progress starts at 0.0."
    )


def _debug_enabled() -> bool:
    return os.getenv("AIDU_DEBUG", "").strip().lower() in DEBUG_TRUE_VALUES


def _domain_from_context(session_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject": session_context.get("subject"),
        "subject_label": session_context.get("subject_label"),
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
    student_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_context = session_context or {}
    belief = _student_belief()
    student_progress = student_progress or _student_progress_from_context(session_context)

    args = build_chem_applet_prompt_args(
        tutor_name="Marie",
        level="beginner",
        history=history or " - Student just entered the GUI tutoring session.",
        student_progress=_student_progress_tutor_text(student_progress),
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

    The GUI sends live applet changes as ``RunRequest.info.applet_input``.
    The actor already turns that payload into an ``AppletArtifact`` for routing;
    this helper keeps the same structured payload available so the LLM tutor can
    see the latest applet values when it writes its next response.
    """
    applet_input = req.info.applet_input
    if isinstance(applet_input, dict):
        applet_info = AppletInfo.from_payload(applet_input)
        logger.debug(
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
        session_context = session_context or {}
        agents = [
            BeginAgent(target=GuiInputRouter, interactive=_debug_enabled()),
            GuiInputRouter(client=client, session_context=session_context),
            AppletRuleResponder(),
            GuiChemLlmTutor(client, prompt_args=_prompt_args(session_context)),
            DebugAgent(),
            JoinEndAgent(),
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
        logger.debug(
            "GUI tutor build_context tutor_class=%s request_domain=%s:%s request_applet=%s:%s history_turns=%s content_prefix=%r",
            GuiChemLlmTutor.__name__,
            session_context.get("domain"),
            session_context.get("domain_label"),
            session_context.get("applet_id"),
            session_context.get("applet_name"),
            len(prior_messages),
            str(req.message.content or "")[:240],
        )
        context = Context()
        self.configure_context_from_request(context, req)
        context.state.data["StudentBelief"] = _student_belief()
        context.state.data["StudentProgress"] = (
            _latest_backend_progress_state(forwarded_messages)
            or _student_progress_from_context(session_context)
        )
        context.state.data["StudentGoal"] = _student_goal_from_context(session_context)
        context.create_agent_states(self.agents)

        applet_state = _latest_applet_info_store_from_request(req)
        tutor_state = context.state.data.setdefault(GuiChemLlmTutor.__name__, {})
        tutor_state.update(
            _prompt_args(
                session_context,
                applet_state=applet_state,
                history=history,
                student_progress=context.state.data["StudentProgress"],
            )
        )
        if forwarded_messages:
            context.trace.messages = [
                cleaned_message
                for message in forwarded_messages[-MAX_HISTORY_TURNS:]
                if (cleaned_message := _clean_dialog_message(message))
            ]
        logger.debug(
            "GUI tutor context ready state_class=%s domain=%s applet=%s trace_messages=%s state_keys=%s",
            GuiChemLlmTutor.__name__,
            tutor_state.get("domain_id"),
            tutor_state.get("applet_id"),
            len(context.trace.messages),
            sorted(tutor_state.keys()),
        )

        return context
