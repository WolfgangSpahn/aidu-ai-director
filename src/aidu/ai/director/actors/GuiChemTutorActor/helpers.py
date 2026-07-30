# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.

"""Pure context, prompt, parsing, and assessment-result helpers for the actor."""

from __future__ import annotations

import logging
from typing import Any

from aidu.ai.actor.types import RunRequest
from aidu.ai.core.knowledge_progress import StudentKnowledgeProgress
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.belief import StudentBelief, StudentBeliefAssessment
from aidu.ai.core.supervisor import SupervisorState
from aidu.ai.core.context import Context

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10
EVIDENCE_WEIGHTS = {
    "w": 1.0,
    "m": 2.0,
    "s": 4.0,
}


def update_context_with_supervision_assessment(assessment: dict[str, Any], context: Context) -> None:
    """Update the joined turn context from the AI supervisor contract."""
    context.state.data["SupervisorState"] = SupervisorState.model_validate(assessment)
    context.control.data["ai_supervision_assessment"] = assessment


def update_context_with_belief_assessment(assessment: dict[str, Any], context: Context) -> None:
    """
    Update the joined turn context from the assessor's belief contract.
    """
    belief_assessment = StudentBeliefAssessment.model_validate(assessment)
    context.state.data["StudentBelief"] = StudentBelief(**belief_assessment.belief.model_dump())
    context.control.data["student_belief_assessment"] = assessment


def apply_target_assessment(assessment: dict[str, Any], context: Context, evidence_scale: float = 1.0) -> None:
    progress: StudentKnowledgeProgress = context.state.data["StudentKnowledgeProgress"]

    evidence = assessment.get("e")
    if not isinstance(evidence, list):
        logger.debug("LearningTargetAssessor.apply skipped reason=no_evidence assessment=%s", assessment)
        return

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        indicator = str(item.get("i") or "").strip()
        if indicator not in progress.root:
            skipped.append({"indicator": indicator, "reason": "missing_progress_key"})
            continue
        target_state = progress.root[indicator]
        polarity = str(item.get("p") or "?")
        if polarity not in {"+", "-"}:
            skipped.append({"indicator": indicator, "reason": "zero_delta"})
            continue
        weight = EVIDENCE_WEIGHTS.get(str(item.get("s") or "w"), 1.0) * evidence_scale
        positive = max(0.0, target_state.positive_evidence)
        negative = max(0.0, target_state.negative_evidence)
        prior = positive / (positive + negative) if positive + negative else 0.0
        if polarity == "+":
            positive += weight
        else:
            negative += weight
        posterior = positive / (positive + negative) if positive + negative else 0.0
        target_state.mastery = posterior
        target_state.positive_evidence = positive
        target_state.negative_evidence = negative
        applied.append(
            {
                "indicator": indicator,
                "prior": prior,
                "weight": weight,
                "polarity": polarity,
                "posterior": posterior,
            }
        )

    if applied:
        context.control.data["learning_target_assessor_evidence"] = assessment
        logger.debug("LearningTargetAssessor progress applied: %s", applied)
    else:
        logger.debug("LearningTargetAssessor.apply no_progress_change skipped=%s assessment=%s", skipped, assessment)


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

    for message in reversed(req.info.messages):
        if not isinstance(message, dict):
            continue
        applet_info = AppletInfo.from_message(message)
        if applet_info:
            return applet_info.to_state()

    return {}
