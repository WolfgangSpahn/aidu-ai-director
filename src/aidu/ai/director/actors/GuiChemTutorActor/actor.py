"""Assemble and run the GUI chemistry tutor actor workflow."""

import logging
import os
from typing import Any

from aidu.ai.actor.actor import Actor
from aidu.ai.actor.turn_scope import JoinEndAgent
from aidu.ai.actor.types import RunRequest
from aidu.ai.agents.ai_supervisor import AiSupervisor
from aidu.ai.agents.ai_label_intervention import AiLabelIntervention
from aidu.ai.agents.chem_applet_tutor import (
    ChemLlmTutor,
    build_chem_applet_prompt_args,
)
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.knowledge_progress import StudentKnowledgeProgress
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessor
from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.context import Context
from aidu.ai.core.session import SessionContext
from aidu.ai.llm.agent import BeginAgent, DebugAgent, EndAgent
from aidu.ai.llm.clients.google import GoogleClient

from .accessor_router import AssessorRouter
from .gui_input_router import GuiInputRouter
from .helpers import (
    MAX_HISTORY_TURNS,
    _latest_applet_info_store_from_request,
)

logger = logging.getLogger(__name__)
TUTOR_MODEL = os.getenv("AIDU_TUTOR_MODEL", "gemini-3.6-flash")
ASSESSOR_MODEL = os.getenv("AIDU_ASSESSOR_MODEL", "gemini-3.5-flash-lite")
KNOWLEDGE_ASSESSOR_MODEL = os.getenv(
    "AIDU_KNOWLEDGE_ASSESSOR_MODEL",
    "gemini-3.5-flash-lite",
)
INTERVENTION_LABEL_MODEL = os.getenv(
    "AIDU_INTERVENTION_LABEL_MODEL",
    "gemini-3.5-flash-lite",
)


def _debug_enabled() -> bool:
    """Enable interactive server debugging only for canonical value ``1``."""
    return os.getenv("AIDU_DEBUG") == "1"


class GuiChemLlmTutor(ChemLlmTutor):
    """Generate the tutor reply after entry-time assessments have started."""

    target = JoinEndAgent
    terminal_target = JoinEndAgent
    continuations = []

    @classmethod
    def build_prompt_args(
        cls,
        *,
        tutor_name: str,
        session_context: SessionContext,
        applet_state: dict[str, Any] | str | None = None,
        history: str = " - Student just entered the GUI tutoring session.",
        student_knowledge_progress: StudentKnowledgeProgress | None = None,
        student_belief: StudentBelief | None = None,
    ) -> dict[str, Any]:
        """Build this agent's prompt state from the validated actor context."""
        knowledge_progress = (
            student_knowledge_progress
            or session_context.initial_student_knowledge_progress()
        )
        belief = student_belief or StudentBelief()
        progress_percent = knowledge_progress.mean_mastery_percent()
        return build_chem_applet_prompt_args(
            tutor_name=tutor_name,
            level="beginner",
            history=history,
            student_knowledge_progress=knowledge_progress.to_tutor_text(),
            student_belief=" - " + belief.to_tutor_text(),
            current_progress_percent=progress_percent,
            domain=session_context.domain_prompt_metadata(),
            applet=session_context.applet_prompt_metadata(),
            applet_state=applet_state,
        )


GuiInputRouter.continuations = [GuiChemLlmTutor]


class GuiChemTutorActor(Actor):
    """Run GUI routing, tutoring, assessment, and turn-state collection."""

    def __init__(
        self,
        client=None,
        session_context: SessionContext | None = None,
        assessor_client=None,
        knowledge_assessor_client=None,
        intervention_label_client=None,
        tutor_name: str = "Marie",
    ):
        self.tutor_name = tutor_name
        tutor_client = client or GoogleClient(
            model=TUTOR_MODEL,
            config={"max_tokens": 1024, "thinking_level": "medium"},
        )
        explicit_assessor_client = assessor_client
        assessor_client = assessor_client or (
            client
            if client is not None
            else GoogleClient(
                model=ASSESSOR_MODEL,
                config={"max_tokens": 1024, "thinking_level": "low"},
            )
        )
        knowledge_assessor_client = knowledge_assessor_client or (
            assessor_client
            if client is not None or explicit_assessor_client is not None
            else GoogleClient(
                model=KNOWLEDGE_ASSESSOR_MODEL,
                config={"max_tokens": 1024, "thinking_level": "low"},
            )
        )
        intervention_label_client = intervention_label_client or (
            assessor_client
            if client is not None or explicit_assessor_client is not None
            else GoogleClient(
                model=INTERVENTION_LABEL_MODEL,
                config={"max_tokens": 256, "thinking_level": "low"},
            )
        )
        session_context = session_context or SessionContext(on_air=True)
        assessors = (
            LearningTargetAssessor(client=knowledge_assessor_client, target=EndAgent),
            StudentBeliefAssessor(client=assessor_client, target=EndAgent),
            AiSupervisor(client=assessor_client, target=EndAgent),
            AiLabelIntervention(client=intervention_label_client, target=EndAgent),
        )
        agents = [
            BeginAgent(target=AssessorRouter, interactive=_debug_enabled()),
            AssessorRouter(assessors=assessors),
            GuiInputRouter(),
            GuiChemLlmTutor(
                tutor_client,
                prompt_args=GuiChemLlmTutor.build_prompt_args(
                    tutor_name=self.tutor_name,
                    session_context=session_context,
                ),
            ),
            DebugAgent(),
            JoinEndAgent(),
            EndAgent(),
        ]
        logger.debug(
            "Creating GuiChemTutorActor startup=%s agents=%s",
            AssessorRouter.__name__,
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
        """Build the workflow context from the current GUI run request."""
        session_context = req.info.session_context
        forwarded_messages = req.info.messages
        history = forwarded_messages.before_last().dialog_history(MAX_HISTORY_TURNS)
        context = Context()
        self.configure_context_from_request(context, req)
        context.state.data["SessionContext"] = session_context
        context.state.data["TurnIndex"] = len(forwarded_messages)
        # Keep learner-authored text separate from machine-generated applet
        # payloads. Assessors may use the latter only to verify the former.
        context.state.data["CurrentStudentMessage"] = str(req.message.content or "").strip()
        context.state.data["OutcomeStudentTurnIndex"] = max(0, len(forwarded_messages) - 1)
        context.state.data["LastTutorTurnIndex"] = next(
            (
                index
                for index in range(len(forwarded_messages.root) - 1, -1, -1)
                if forwarded_messages.root[index].get("role") == "assistant"
            ),
            None,
        )
        context.state.data["StudentBelief"] = forwarded_messages.latest_belief()
        context.state.data["IsInitialTutorTurn"] = (
            context.state.data["LastTutorTurnIndex"] == next(
                (index for index, turn in enumerate(forwarded_messages.root) if turn.get("role") == "assistant"),
                None,
            )
        ) if context.state.data["LastTutorTurnIndex"] is not None else None
        context.state.data["SupervisorState"] = forwarded_messages.latest_supervisor()
        # The newest persisted value is useful as context, but must not be
        # emitted again if this turn's supervisor side task fails.
        context.control.data["emit_supervision_state"] = False
        knowledge_progress = forwarded_messages.latest_knowledge_progress()
        context.state.data["StudentKnowledgeProgress"] = (
            knowledge_progress
            if knowledge_progress.root
            else session_context.initial_student_knowledge_progress()
        )
        context.create_agent_states(self.agents)

        applet_state = _latest_applet_info_store_from_request(req)
        context.state.data["AppletState"] = applet_state
        tutor_state = context.state.data.setdefault(GuiChemLlmTutor.__name__, {})
        tutor_state.update(
            GuiChemLlmTutor.build_prompt_args(
                tutor_name=self.tutor_name,
                session_context=session_context,
                applet_state=applet_state,
                history=history,
                student_knowledge_progress=context.state.data["StudentKnowledgeProgress"],
                student_belief=context.state.data["StudentBelief"],
            )
        )
        if forwarded_messages:
            context.trace.messages = forwarded_messages.cleaned_dialog(MAX_HISTORY_TURNS)
        return context


__all__ = ["GuiChemLlmTutor", "GuiChemTutorActor"]
