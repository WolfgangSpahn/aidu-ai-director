# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.

"""Pure context, prompt, parsing, and assessment-result helpers for the actor."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from aidu.ai.actor.types import RunRequest
from aidu.ai.core.knowledge_progress import StudentKnowledgeProgress
from aidu.ai.core.knowledge_progress import EvidenceKnowledgeProgress
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessment
from aidu.support.scoring import (
    MAX_TARGET_WEIGHT_PER_TURN,
    TurnAssessment,
    apply_turn_assessment,
)
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.belief import StudentBelief, StudentBeliefAssessment
from aidu.ai.core.supervisor import SupervisorState
from aidu.ai.core.context import Context

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10
MAX_BELIEF_DELTA_PER_TURN = 0.15

_HINDSIGHT_REASON_PATTERNS = (
    re.compile(r"\b(?:did not|didn't|failed to)\s+(?:\w+\s+){0,2}(?:address|acknowledge|answer|react|respond)\b", re.I),
    re.compile(r"\bdoes not\s+(?:\w+\s+){0,2}(?:address|acknowledge|answer|react|respond)\b", re.I),
    re.compile(r"\bdoesn't\s+(?:\w+\s+){0,2}(?:address|acknowledge|answer|react|respond)\b", re.I),
)


def _remove_supervision_hindsight(assessment: dict[str, Any]) -> dict[str, Any]:
    """Prevent a later learner outcome being described as prior tutor context."""
    cleaned = dict(assessment)
    for dimension, value in assessment.items():
        if not isinstance(value, dict):
            continue
        reason = str(value.get("reason") or "")
        if not any(pattern.search(reason) for pattern in _HINDSIGHT_REASON_PATTERNS):
            continue
        cleaned[dimension] = {
            **value,
            "reason": (
                "The learner outcome suggests that the preceding tutor support "
                "may not have been sufficiently clear or actionable."
            ),
        }
    return cleaned


def update_context_with_supervision_assessment(
    assessment: dict[str, Any],
    context: Context,
    *,
    assessed_tutor_turn_index: int | None = None,
    outcome_student_turn_index: int | None = None,
    outcome_evidence_available: bool = True,
) -> None:
    """Update the joined turn context from the AI supervisor contract."""
    aligned = {
        **_remove_supervision_hindsight(assessment),
        "assessed_tutor_turn_index": assessed_tutor_turn_index,
        "outcome_student_turn_index": outcome_student_turn_index,
        "outcome_evidence_available": outcome_evidence_available,
    }
    context.state.data["SupervisorState"] = SupervisorState.model_validate(aligned)
    context.control.data["ai_supervision_assessment"] = aligned
    context.control.data["emit_supervision_state"] = True


def update_context_with_belief_assessment(assessment: dict[str, Any], context: Context) -> None:
    """
    Update the joined turn context from the assessor's belief contract.
    """
    belief_assessment = StudentBeliefAssessment.model_validate(assessment)
    prior: StudentBelief = context.state.data["StudentBelief"]
    proposed = belief_assessment.belief.model_dump()
    smoothed = {
        key: max(0.0, min(1.0, max(
            old - MAX_BELIEF_DELTA_PER_TURN,
            min(old + MAX_BELIEF_DELTA_PER_TURN, proposed[key]),
        )))
        for key, old in prior.model_dump().items()
    }
    context.state.data["StudentBelief"] = StudentBelief(**smoothed)
    context.control.data["student_belief_assessment"] = {
        **assessment,
        "belief": smoothed,
    }


def _normalized_evidence_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def apply_target_assessment(
    assessment: dict[str, Any],
    context: Context,
    *,
    current_message: str | None = None,
) -> None:
    """Apply validated assessor evidence through the canonical update engine."""

    progress: StudentKnowledgeProgress = context.state.data["StudentKnowledgeProgress"]
    turn_index = context.state.data["TurnIndex"]
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError("Context TurnIndex must be a non-negative integer.")
    validated = LearningTargetAssessment.model_validate(assessment)
    applied: list[dict[str, Any]] = []
    target_weights: dict[str, float] = {}
    related_counts: dict[str, int] = {}
    for item in validated.evidence:
        indicator = item.target
        if indicator not in progress.root:
            raise ValueError(
                f"Assessor returned target {indicator!r} outside the active domain."
            )
        target_state = progress.root[indicator]
        normalized_quote = _normalized_evidence_text(item.quote)
        normalized_message = _normalized_evidence_text(current_message or "")
        if item.confidence < 0.35 or not normalized_quote:
            logger.info("Ignoring weak target evidence target=%s confidence=%s", indicator, item.confidence)
            continue
        if current_message is not None and normalized_quote not in normalized_message:
            logger.warning("Ignoring non-verbatim target evidence target=%s quote=%r", indicator, item.quote)
            continue
        fingerprint = hashlib.sha256(
            f"{turn_index}\0{indicator}\0{normalized_quote}".encode("utf-8")
        ).hexdigest()
        if fingerprint in target_state.evidence_fingerprints:
            logger.info("Ignoring duplicate target evidence target=%s", indicator)
            continue
        prior = target_state.mastery
        remaining = MAX_TARGET_WEIGHT_PER_TURN - target_weights.get(indicator, 0.0)
        updated, weight = apply_turn_assessment(
            target_state.as_evidence_state(),
            TurnAssessment(
                target=indicator,
                direction=item.direction,
                strength=item.strength,
                confidence=item.confidence,
                evidence_type=item.evidence_type,
                support_level=item.support_level,
            ),
            turn_index=turn_index,
            previous_related_assessments=related_counts.get(indicator, 0),
            remaining_target_weight=remaining,
        )
        progress.root[indicator] = EvidenceKnowledgeProgress.from_evidence_state(
            updated.__class__(
                **{
                    **updated.__dict__,
                    "evidence_fingerprints": (
                        (*updated.evidence_fingerprints, fingerprint)[-100:]
                        if weight > 0
                        else updated.evidence_fingerprints
                    ),
                }
            )
        )
        target_weights[indicator] = target_weights.get(indicator, 0.0) + weight
        related_counts[indicator] = related_counts.get(indicator, 0) + 1
        applied.append(
            {
                "indicator": indicator,
                "prior": prior,
                "weight": weight,
                "direction": item.direction,
                "posterior": updated.mastery,
            }
        )

    context.control.data["learning_target_assessor_evidence"] = assessment
    logger.debug("LearningTargetAssessor progress applied: %s", applied)


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
        applet_info = AppletInfo.from_message(message)
        if applet_info:
            return applet_info.to_state()

    return {}
