"""Route GUI-originated applet and typed input to the chemistry tutor."""

import logging

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import AppletArtifact, Artifact
from aidu.ai.core.context import Context
from aidu.ai.llm.agent import WorkflowAgent

logger = logging.getLogger(__name__)


class GuiInputRouter(WorkflowAgent):
    """Choose the tutor path for a GUI-originated input artifact."""

    target = None
    continuations = []

    def run(
        self,
        artifact: Artifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        from .actor import GuiChemLlmTutor

        is_applet_input = isinstance(artifact, AppletArtifact)
        mode = "applet_input" if is_applet_input else "typed_input"
        recommendation = self.register_recommendation(
            mode,
            target=GuiChemLlmTutor,
            continuations=[],
            utility=1.0,
            rationale="Applet and typed input are handled by the LLM tutor.",
        )
        logger.debug(
            "GuiInputRouter.route mode=%s artifact_type=%s content=%r",
            mode,
            artifact.type,
            artifact.content if is_applet_input else str(artifact.content)[:160],
        )
        return self.result(artifacts=[], recommendations=[recommendation]), context
