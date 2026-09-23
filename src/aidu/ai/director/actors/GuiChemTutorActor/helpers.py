# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.

"""Validate assessment evidence and apply it to the joined tutor context.

The assessor models identify observable evidence. This module owns the
deterministic transition from that evidence to persistent learner and
supervision state. Keeping the transition here prevents an LLM from directly
choosing final knowledge or belief values.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any
from pydantic import ValidationError

from aidu.ai.actor.types import RunRequest
from aidu.ai.core.knowledge_progress import StudentKnowledgeProgress
from aidu.ai.core.knowledge_progress import EvidenceKnowledgeProgress
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessment
from aidu.support.scoring import (
    BeliefEvidenceSignal,
    MAX_TARGET_WEIGHT_PER_TURN,
    TurnAssessment,
    apply_turn_assessment,
    project_belief_state,
)
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.belief import StudentBelief, StudentBeliefAssessment
from aidu.ai.core.supervisor import InterventionLabel, SupervisorState
from aidu.ai.core.context import Context

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10
_HINDSIGHT_REASON_PATTERNS = (
    re.compile(r"\b(?:did not|didn't|failed to)\s+(?:\w+\s+){0,2}(?:address|acknowledge|answer|react|respond)\b", re.I),
    re.compile(r"\bdoes not\s+(?:\w+\s+){0,2}(?:address|acknowledge|answer|react|respond)\b", re.I),
    re.compile(r"\bdoesn't\s+(?:\w+\s+){0,2}(?:address|acknowledge|answer|react|respond)\b", re.I),
)


def _remove_supervision_hindsight(assessment: dict[str, Any]) -> dict[str, Any]:
    """Remove reasons that treat a later learner response as prior context.

    Args:
        assessment: Raw supervisor dimensions and their explanations.

    Returns:
        A shallow copy whose hindsight-contaminated explanations are replaced
        with wording that correctly treats the learner response as an outcome.
    """
    # Work on a copy because the raw assessment is retained for diagnostics by
    # callers and must not be changed in place.
    cleaned = dict(assessment)

    # Inspect only dimension-shaped entries; metadata does not contain reasons.
    for dimension, value in assessment.items():
        if not isinstance(value, dict):
            continue
        reason = str(value.get("reason") or "")
        if not any(pattern.search(reason) for pattern in _HINDSIGHT_REASON_PATTERNS):
            continue
        # Preserve the score and other fields while replacing only the invalid
        # temporal interpretation in its explanation.
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
    """Validate and store supervision for the tutor turn being assessed.

    Args:
        assessment: Raw JSON object returned by ``AiSupervisor``.
        context: Joined turn context that receives the validated state.
        assessed_tutor_turn_index: History index of the evaluated tutor turn.
        outcome_student_turn_index: Index of its following learner response.
        outcome_evidence_available: Whether such a learner response exists.

    Raises:
        pydantic.ValidationError: If the completed supervisor contract is
            invalid.
    """
    # Align the assessor output with explicit turn provenance before validating
    # it as the canonical supervisor state.
    current_supervision = context.state.data.get("SupervisorState")
    aligned = {
        **_remove_supervision_hindsight(assessment),
        "assessed_tutor_turn_index": assessed_tutor_turn_index,
        "outcome_student_turn_index": outcome_student_turn_index,
        "outcome_evidence_available": outcome_evidence_available,
        "intervention": getattr(current_supervision, "intervention", None),
    }
    # Store typed state for downstream tutor decisions and retain the aligned
    # assessment in control data for response serialization.
    context.state.data["SupervisorState"] = SupervisorState.model_validate(aligned)
    context.control.data["ai_supervision_assessment"] = aligned
    context.control.data["emit_supervision_state"] = True


def update_context_with_intervention_label(
    assessment: dict[str, Any],
    context: Context,
) -> None:
    """Validate and attach a tutor-intervention label to supervision state."""
    label = InterventionLabel.model_validate(assessment)
    current = context.state.data.get("SupervisorState") or SupervisorState.prior()
    context.state.data["SupervisorState"] = current.model_copy(
        update={"intervention": label},
    )
    context.control.data["ai_intervention_label"] = label.model_dump(mode="json")
    context.control.data["emit_supervision_state"] = True


def update_context_with_belief_assessment(
    assessment: dict[str, Any],
    context: Context,
    *,
    current_message: str | None = None,
) -> None:
    """Derive the next belief state from validated learner speech acts.

    Args:
        assessment: Speech-act evidence returned by ``StudentBeliefAssessor``.
        context: Joined turn context containing the prior ``StudentBelief``.
        current_message: Learner-authored text used to verify evidence quotes.

    Raises:
        KeyError: If the context does not contain a prior belief state.
        pydantic.ValidationError: If the assessment contract is invalid.

    The assessor never supplies final belief values. Each accepted speech act
    maps to deterministic dimension effects, and the accumulated change for a
    dimension is capped per learner turn.
    """
    # Validate the untrusted assessor payload before reading any evidence.
    rejected_evidence: list[dict[str, Any]] = []
    try:
        belief_assessment = StudentBeliefAssessment.model_validate(assessment)
    except ValidationError as exc:
        # Unsupported model labels have no scoring interpretation. Preserve
        # them for review rather than aborting the entire dialog replay.
        errors = exc.errors()
        if not all(
            error["type"] == "literal_error"
            and len(error["loc"]) == 3
            and error["loc"][0] == "evidence"
            and isinstance(error["loc"][1], int)
            and error["loc"][2] == "speech_act"
            for error in errors
        ):
            raise
        rejected_indices = {error["loc"][1] for error in errors}
        rejected_evidence = [
            {**item, "reason": "Unsupported speech act"}
            for index, item in enumerate(assessment["evidence"])
            if index in rejected_indices
        ]
        belief_assessment = StudentBeliefAssessment.model_validate({
            **assessment,
            "evidence": [
                item for index, item in enumerate(assessment["evidence"])
                if index not in rejected_indices
            ],
            "review": True,
        })
        logger.warning("Ignoring unsupported belief evidence: %s", rejected_evidence)

    # Begin from the prior state; dimensions without accepted evidence remain
    # unchanged rather than being re-estimated by the model.
    prior: StudentBelief = context.state.data["StudentBelief"]
    normalized_message = _normalized_evidence_text(current_message or "")
    applied_evidence: list[dict[str, Any]] = []
    signals: list[BeliefEvidenceSignal] = []
    # Accept only sufficiently reliable, learner-authored evidence. Exact quote
    # verification prevents history or tutor text from becoming learner state.
    for item in belief_assessment.evidence:
        normalized_quote = _normalized_evidence_text(item.quote)
        if item.confidence < 0.35 or not normalized_quote:
            continue
        if current_message is not None and normalized_quote not in normalized_message:
            logger.warning(
                "Ignoring non-verbatim belief evidence speech_act=%s quote=%r",
                item.speech_act,
                item.quote,
            )
            continue
        # Build the ordered evidence vector inputs without deriving belief fields
        # inside the assessor integration layer.
        signals.append(BeliefEvidenceSignal(
            speech_act=item.speech_act,
            strength=item.strength,
            confidence=item.confidence,
        ))
        applied_evidence.append(item.model_dump(mode="json"))

    # Project the speech-act evidence vector through the canonical coefficient
    # matrix: next belief = prior belief + M · evidence.
    values, belief_delta, evidence_vector = project_belief_state(
        prior.model_dump(),
        signals,
    )

    # Publish typed derived state and the evidence trail separately so consumers
    # can distinguish observations from conclusions.
    context.state.data["StudentBelief"] = StudentBelief(**values)
    context.control.data["student_belief_assessment"] = {
        "evidence": applied_evidence,
        "review": belief_assessment.review,
        "evidence_vector": evidence_vector,
        "belief_delta": belief_delta,
        "derived_belief": values,
        "rejected_evidence": rejected_evidence,
    }


def _normalized_evidence_text(value: str) -> str:
    """Normalize text for robust, case-insensitive quote containment checks."""
    return " ".join(re.findall(r"\w+", value.casefold()))


def apply_target_assessment(
    assessment: dict[str, Any],
    context: Context,
    *,
    current_message: str | None = None,
) -> None:
    """Apply learning-target evidence to ``StudentKnowledgeProgress``.

    Args:
        assessment: Evidence returned by ``LearningTargetAssessor``.
        context: Joined turn context containing knowledge progress and index.
        current_message: Learner-authored text used to verify evidence quotes.

    Raises:
        KeyError: If required knowledge state or turn metadata is missing.
        ValueError: If the turn index or an assessed target is invalid.
        pydantic.ValidationError: If the assessment contract is invalid.

    Each accepted evidence item is passed to the canonical scoring engine. The
    resulting target state replaces that target's entry in the progress object,
    which is already stored in the joined context.
    """

    # Resolve the mutable knowledge state and validate the turn identity used
    # for scoring and evidence deduplication.
    progress: StudentKnowledgeProgress = context.state.data["StudentKnowledgeProgress"]
    turn_index = context.state.data["TurnIndex"]
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError("Context TurnIndex must be a non-negative integer.")

    # Validate the entire assessor response before applying any item from it.
    validated = LearningTargetAssessment.model_validate(assessment)
    applied: list[dict[str, Any]] = []
    target_weights: dict[str, float] = {}
    related_counts: dict[str, int] = {}

    # Process selected evidence only. Targets omitted by the assessor receive no
    # update on this turn, matching the evidence-first knowledge model.
    for item in validated.evidence:
        indicator = item.target
        if indicator not in progress.root:
            raise ValueError(
                f"Assessor returned target {indicator!r} outside the active domain."
            )

        # Reject weak or non-verbatim evidence before it can affect mastery.
        target_state = progress.root[indicator]
        normalized_quote = _normalized_evidence_text(item.quote)
        normalized_message = _normalized_evidence_text(current_message or "")
        if item.confidence < 0.35 or not normalized_quote:
            logger.info("Ignoring weak target evidence target=%s confidence=%s", indicator, item.confidence)
            continue
        if current_message is not None and normalized_quote not in normalized_message:
            logger.warning("Ignoring non-verbatim target evidence target=%s quote=%r", indicator, item.quote)
            continue

        # Bind the fingerprint to turn, target, and quote so repeated callbacks
        # cannot count the same evidence twice.
        fingerprint = hashlib.sha256(
            f"{turn_index}\0{indicator}\0{normalized_quote}".encode("utf-8")
        ).hexdigest()
        if fingerprint in target_state.evidence_fingerprints:
            logger.info("Ignoring duplicate target evidence target=%s", indicator)
            continue

        # Respect the per-target turn budget, then delegate all mastery math to
        # the shared canonical scoring engine.
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
                response_mode=item.response_mode,
                support_level=item.support_level,
            ),
            turn_index=turn_index,
            previous_related_assessments=related_counts.get(indicator, 0),
            remaining_target_weight=remaining,
        )

        # Replace the target entry in the context-owned progress object and keep
        # a bounded history of fingerprints for future duplicate detection.
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

        # Track this turn's accumulated weight and related evidence count for
        # subsequent evidence items concerning the same target.
        target_weights[indicator] = target_weights.get(indicator, 0.0) + weight
        related_counts[indicator] = related_counts.get(indicator, 0) + 1
        applied.append(
            {
                **item.model_dump(mode="json"),
                "fingerprint": fingerprint,
                "indicator": indicator,
                "prior": prior,
                "weight": weight,
                "direction": item.direction,
                "posterior": updated.mastery,
            }
        )

    # Retain the original assessment for serialization and audit independently
    # of the derived knowledge state stored under StudentKnowledgeProgress.
    context.control.data["learning_target_assessor_evidence"] = assessment
    context.control.data["learning_target_applied_evidence"] = {"evidence": applied, "review": validated.review}
    logger.debug("LearningTargetAssessor progress applied: %s", applied)


def _latest_applet_info_store_from_request(req: RunRequest) -> dict[str, Any]:
    """Return the latest structured applet payload from the GUI.

    The GUI sends live applet changes as ``RunRequest.info.applet_input``.
    The actor already turns that payload into an ``AppletArtifact`` for routing;
    this helper keeps the same structured payload available so the LLM tutor can
    see the latest applet values when it writes its next response.

    Args:
        req: Current GUI run request with live input and message history.

    Returns:
        The normalized applet state, or an empty dictionary when no structured
        applet information is available.
    """
    # Prefer current-turn GUI telemetry because it is newer than archived
    # applet snapshots attached to earlier messages.
    applet_input = req.info.applet_input
    if isinstance(applet_input, dict):
        applet_info = AppletInfo.from_payload(applet_input)
        logger.debug(
            "GUI tutor applet state parsed keys=%s applet=%s",
            sorted(applet_info.to_state().keys()),
            applet_info.applet,
        )
        return applet_info.to_state()

    # Fall back to the newest structured applet snapshot in message history.
    for message in reversed(req.info.messages):
        applet_info = AppletInfo.from_message(message)
        if applet_info:
            return applet_info.to_state()

    # An absent applet is valid for text-only activities.
    return {}
