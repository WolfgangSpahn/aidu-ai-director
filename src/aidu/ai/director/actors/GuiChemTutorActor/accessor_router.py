"""Start and coordinate the three entry-time assessment side tasks."""

import json
import logging
from typing import Any

from aidu.ai.actor.turn_scope import get_turn_side_tasks
from aidu.ai.agents.ai_supervisor import AiSupervisor
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessor
from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import Artifact, TextArtifact
from aidu.ai.core.artifacts import AppletArtifact
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context
from aidu.ai.llm.agent import Agent, EndAgent, WorkflowAgent

from .gui_input_router import GuiInputRouter
from .helpers import (
    apply_target_assessment,
    update_context_with_belief_assessment,
    update_context_with_supervision_assessment,
)

logger = logging.getLogger(__name__)


def learner_evidence_text(artifact: Artifact, context: Context) -> str:
    """Return learner-authored text, never serialized applet telemetry."""

    if isinstance(artifact, AppletArtifact):
        return str(context.state.data.get("CurrentStudentMessage") or "").strip()
    return artifact.to_text()


def _run_assessment(
    *,
    agent: Agent,
    context: Context,
    prompt_params: dict[str, Any],
    instruction: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Run one injected assessor directly and decode its structured result."""
    result, _ = agent.run(
        artifact=TextArtifact(
            producer="AssessorRouter",
            step=context.step,
            content=instruction,
        ),
        context=context,
        agents=[agent, EndAgent()],
        ask_params=prompt_params,
        ask_config=AskConfig(
            json_mode=True,
            max_tokens=max_tokens,
            vendor_config={
                "reasoning": {"effort": "minimal"},
                "verbosity": "low",
            },
        ),
    )
    content = result.content()
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{agent.id} returned non-JSON assessment content."
        ) from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{agent.id} assessment must be a JSON object.")
    return decoded


class AssessorRouter(WorkflowAgent):
    """Launch all assessors immediately, then continue GUI routing."""

    target = GuiInputRouter
    continuations = []

    def __init__(
        self,
        assessors: tuple[
            LearningTargetAssessor,
            StudentBeliefAssessor,
            AiSupervisor,
        ],
    ):
        self.learning_target_assessor, self.student_belief_assessor, self.ai_supervisor = assessors

    def run(self, artifact: Artifact, context: Context, agents=None) -> tuple[AgentResult, Context]:
        current_turn = learner_evidence_text(artifact, context)
        side = get_turn_side_tasks(context)

        target_context = context.for_assessor()
        target_params = self.learning_target_assessor.build_prompt_args(
            context=target_context,
            current_message=current_turn,
        )
        side.spawn(
            "learning_target_assessor",
            lambda: _run_assessment(
                agent=self.learning_target_assessor,
                prompt_params=target_params,
                context=target_context,
                instruction="Assess the current chemistry learning evidence.",
                max_tokens=512,
            ),
            on_result=lambda result, joined: apply_target_assessment(
                assessment=result,
                context=joined,
                current_message=current_turn,
            ),
        )

        belief_context = context.for_assessor()
        belief_params = self.student_belief_assessor.build_prompt_args(
            context=belief_context,
            current_message=current_turn,
        )
        side.spawn(
            "student_belief_assessor",
            lambda: _run_assessment(
                agent=self.student_belief_assessor,
                prompt_params=belief_params,
                context=belief_context,
                instruction="Assess the student's current belief state.",
                max_tokens=512,
            ),
            on_result=lambda result, joined: update_context_with_belief_assessment(
                assessment=result,
                context=joined,
            ),
        )

        supervisor_context = context.for_assessor()
        supervisor_params = self.ai_supervisor.build_prompt_args(
            context=supervisor_context,
            current_student_message=current_turn,
        )
        side.spawn(
            "ai_supervisor",
            lambda: _run_assessment(
                agent=self.ai_supervisor,
                prompt_params=supervisor_params,
                context=supervisor_context,
                instruction="Assess the preceding AI tutor response.",
                max_tokens=1024,
            ),
            on_result=lambda result, joined: update_context_with_supervision_assessment(
                assessment=result,
                context=joined,
                assessed_tutor_turn_index=context.state.data.get("LastTutorTurnIndex"),
                outcome_student_turn_index=max(0, context.state.data["TurnIndex"] - 1),
            ),
        )

        recommendation = self.register_recommendation(
            "assess_and_route",
            target=GuiInputRouter,
            continuations=[],
            utility=1.0,
            rationale="Assessors are running; continue the GUI-originated route.",
        )
        return self.result(artifacts=[], recommendations=[recommendation]), context
