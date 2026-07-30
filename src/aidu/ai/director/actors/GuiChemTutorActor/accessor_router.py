"""Start and coordinate the three entry-time assessment side tasks."""

import json
import logging
from typing import Any

from aidu.ai.actor.turn_scope import get_turn_side_tasks
from aidu.ai.agents.ai_supervisor import AiSupervisor
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessor
from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import AppletArtifact, Artifact, TextArtifact
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


def _run_assessment(
    *,
    agent: Agent,
    context: Context,
    prompt_params: dict[str, Any],
    instruction: str,
    max_tokens: int,
    invalid_result: dict[str, Any],
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
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("%s returned non-JSON content: %r", agent.id, content)
        return {**invalid_result, "review": True, "raw": content}


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
        current_turn = artifact.to_text()
        is_applet_input = isinstance(artifact, AppletArtifact)
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
                invalid_result={"e": []},
            ),
            on_result=lambda result, joined: apply_target_assessment(
                assessment=result,
                context=joined,
                evidence_scale=0.1 if is_applet_input else 1.0,
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
                invalid_result={"belief": {}},
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
                invalid_result={},
            ),
            on_result=lambda result, joined: update_context_with_supervision_assessment(
                assessment=result,
                context=joined,
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
