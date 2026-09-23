"""Replay assessment agents over an existing dialog without generating tutor turns."""

from __future__ import annotations

from typing import Any

from aidu.ai.agents.ai_supervisor import AiSupervisor
from aidu.ai.agents.ai_label_intervention import AiLabelIntervention
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessor
from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.context import Context, Messages
from aidu.ai.core.session import SessionContext

from .accessor_router import AssessorRouter
from .helpers import (
    apply_target_assessment,
    update_context_with_belief_assessment,
    update_context_with_intervention_label,
    update_context_with_supervision_assessment,
)


def _configured_assessors(actor):
    """Return assessors from the live nested router or a legacy flat actor."""
    router = next(
        (agent for agent in actor.agents if isinstance(agent, AssessorRouter)),
        None,
    )
    if router is not None:
        return (
            router.learning_target_assessor,
            router.student_belief_assessor,
            router.ai_supervisor,
        )

    target = next(
        (agent for agent in actor.agents if isinstance(agent, LearningTargetAssessor)),
        None,
    )
    belief = next(
        (agent for agent in actor.agents if isinstance(agent, StudentBeliefAssessor)),
        None,
    )
    supervisor = next(
        (agent for agent in actor.agents if isinstance(agent, AiSupervisor)),
        None,
    )
    if target is None or belief is None or supervisor is None:
        raise ValueError(
            "Chemistry tutor actor does not expose the three configured assessors."
        )
    return target, belief, supervisor


def _configured_intervention_labeler(actor):
    """Return the optional intervention labeler from a nested or flat actor."""
    if actor is None:
        return None
    router = next(
        (agent for agent in actor.agents if isinstance(agent, AssessorRouter)),
        None,
    )
    if router is not None:
        return router.ai_label_intervention
    return next(
        (agent for agent in actor.agents if isinstance(agent, AiLabelIntervention)),
        None,
    )


def assess_unmatched_final_tutor(
    actor,
    turns: list[dict[str, Any]],
    domain_targets: list[dict[str, Any]],
    *,
    progress=None,
    belief: StudentBelief | None = None,
) -> dict[str, Any] | None:
    """Assess a trailing tutor turn without pretending a learner outcome exists."""
    persisted_turns = Messages.model_validate(turns)
    if not persisted_turns or persisted_turns[-1].role != "assistant":
        return None
    tutor_index = len(turns) - 1
    _, _, supervisor_agent = _configured_assessors(actor)
    label_agent = _configured_intervention_labeler(actor)
    session_context = SessionContext(on_air=True, domain_targets=domain_targets)
    if progress is None:
        progress = session_context.initial_student_knowledge_progress()
        for turn in reversed(persisted_turns):
            state = turn.backend_knowledge_progress_state
            if state is not None:
                progress = state
                break
    if belief is None:
        belief = StudentBelief()
        for turn in reversed(persisted_turns):
            state = turn.backend_belief_state
            if state is not None:
                belief = state
                break
    context = Context()
    context.state.data.update({
        "SessionContext": session_context,
        "TurnIndex": tutor_index,
        "LastTutorTurnIndex": tutor_index,
        "IsInitialTutorTurn": tutor_index == next(
            (index for index, turn in enumerate(persisted_turns) if turn.role == "assistant"), None,
        ),
        "StudentKnowledgeProgress": progress,
        "StudentBelief": belief,
        "AppletState": {},
    })
    context.trace.messages = persisted_turns.cleaned_dialog(limit=len(persisted_turns))
    assessor_context = context.for_assessor()
    result = AssessorRouter._run_assessment(
        agent=supervisor_agent,
        context=assessor_context,
        prompt_params=supervisor_agent.build_prompt_args(
            context=assessor_context,
        ),
        instruction="Assess the unmatched final AI tutor response using only context available at that turn.",
        max_tokens=1024,
    )
    update_context_with_supervision_assessment(
        result,
        context,
        assessed_tutor_turn_index=tutor_index,
        outcome_student_turn_index=None,
        outcome_evidence_available=False,
    )
    if label_agent is not None:
        label_result = AssessorRouter._run_assessment(
            agent=label_agent,
            context=assessor_context.model_copy(deep=True),
            prompt_params=label_agent.build_prompt_args(
                context=assessor_context,
                current_student_message="No subsequent learner turn is available.",
            ),
            instruction="Label the intervention in the unmatched final AI tutor response.",
            max_tokens=256,
        )
        update_context_with_intervention_label(label_result, context)
    return {
        "turn_index": tutor_index,
        "assessed_tutor_turn_index": tutor_index,
        "role": "assistant",
        "actor": persisted_turns[-1].actor,
        "supervision_state": context.state.data["SupervisorState"].model_dump(mode="json"),
    }


def reassess_dialog(actor, turns: list[dict[str, Any]], domain_targets: list[dict[str, Any]]) -> dict[str, Any]:
    """Return fresh per-student-turn states using the actor's configured assessors."""
    persisted_turns = Messages.model_validate(turns)
    target_agent, belief_agent, supervisor_agent = _configured_assessors(actor)
    label_agent = _configured_intervention_labeler(actor)
    session_context = SessionContext(on_air=True, domain_targets=domain_targets)
    progress = session_context.initial_student_knowledge_progress()
    belief = StudentBelief()
    # Opening snapshots carry the test-derived knowledge and entry belief.
    opening_turns = []
    for turn in persisted_turns:
        if turn.role == "user":
            break
        opening_turns.append(turn)
    for turn in opening_turns:
        if turn.backend_knowledge_progress_state is not None:
            progress = turn.backend_knowledge_progress_state.model_copy(deep=True)
        if turn.backend_belief_state is not None:
            belief = turn.backend_belief_state.model_copy(deep=True)
    knowledge_states: list[dict[str, Any]] = [{
        "turn_index": -1, "assessed_student_turn_index": None,
        "state_kind": "prior", "knowledge_state": progress.model_dump(mode="json"),
    }]
    belief_states: list[dict[str, Any]] = [{
        "turn_index": -1, "assessed_student_turn_index": None,
        "state_kind": "prior", "belief_state": belief.model_dump(mode="json"),
    }]
    supervision_states: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(persisted_turns):
        if turn.role != "user":
            continue
        current_message = str(turn.content or "").strip()
        if not current_message:
            continue
        prior_turns = Messages(root=persisted_turns.root[:turn_index]).cleaned_dialog(
            limit=turn_index
        )
        context = Context()
        context.state.data.update({
            "SessionContext": session_context,
            "TurnIndex": turn_index,
            "StudentKnowledgeProgress": progress,
            "StudentBelief": belief,
            "AppletState": turn.applet_input or {},
        })
        context.trace.messages = prior_turns
        assessor_context = context.model_copy(deep=True)

        target_result = AssessorRouter._run_assessment(
            agent=target_agent,
            context=assessor_context.model_copy(deep=True),
            prompt_params=target_agent.build_prompt_args(context=assessor_context, current_message=current_message),
            instruction="Reassess the archived learner message for learning evidence.",
            max_tokens=512,
        )
        apply_target_assessment(target_result, context, current_message=current_message)
        progress = context.state.data["StudentKnowledgeProgress"]

        belief_result = AssessorRouter._run_assessment(
            agent=belief_agent,
            context=assessor_context.model_copy(deep=True),
            prompt_params=belief_agent.build_prompt_args(context=assessor_context, current_message=current_message),
            instruction="Classify observable speech acts in the archived learner message.",
            max_tokens=512,
        )
        update_context_with_belief_assessment(
            belief_result,
            context,
            current_message=current_message,
        )
        belief = context.state.data["StudentBelief"]

        tutor_index = next(
            (index for index in range(turn_index - 1, -1, -1) if persisted_turns[index].role == "assistant"),
            None,
        )
        if tutor_index is not None:
            assessor_context.state.data["IsInitialTutorTurn"] = tutor_index == next(
                (index for index, candidate in enumerate(persisted_turns) if candidate.role == "assistant"), None,
            )
            assessor_context.state.data["LastTutorTurnIndex"] = tutor_index
            assessor_context.state.data["OutcomeStudentTurnIndex"] = turn_index
            supervisor_result = AssessorRouter._run_assessment(
                agent=supervisor_agent,
                context=assessor_context.model_copy(deep=True),
                prompt_params=supervisor_agent.build_prompt_args(
                    context=assessor_context,
                ),
                instruction="Reassess the archived preceding AI tutor response.",
                max_tokens=1024,
            )
            update_context_with_supervision_assessment(
                supervisor_result,
                context,
                assessed_tutor_turn_index=tutor_index,
                outcome_student_turn_index=turn_index,
            )
            if label_agent is not None:
                label_result = AssessorRouter._run_assessment(
                    agent=label_agent,
                    context=assessor_context.model_copy(deep=True),
                    prompt_params=label_agent.build_prompt_args(
                        context=assessor_context,
                        current_student_message=current_message,
                    ),
                    instruction="Label the intervention in the archived preceding AI tutor response.",
                    max_tokens=256,
                )
                update_context_with_intervention_label(label_result, context)
            supervision_states.append({
                "turn_index": turn_index,
                "assessed_tutor_turn_index": tutor_index,
                "role": "user",
                "actor": turn.actor,
                "supervision_state": context.state.data["SupervisorState"].model_dump(mode="json"),
            })

        knowledge_states.append({
            "turn_index": turn_index,
            "assessed_student_turn_index": turn_index,
            "role": "user",
            "actor": turn.actor,
            "knowledge_state": progress.model_dump(mode="json"),
            "assessment_evidence": context.control.data.get("learning_target_applied_evidence"),
        })
        belief_states.append({
            "turn_index": turn_index,
            "assessed_student_turn_index": turn_index,
            "role": "user",
            "actor": turn.actor,
            "belief_state": belief.model_dump(mode="json"),
            "assessment_evidence": context.control.data.get("student_belief_assessment"),
        })

    terminal_supervision = assess_unmatched_final_tutor(
        actor,
        turns,
        domain_targets,
        progress=progress,
        belief=belief,
    )
    if terminal_supervision is not None:
        supervision_states.append(terminal_supervision)

    return {
        "knowledge_states": knowledge_states,
        "belief_states": belief_states,
        "supervision_states": supervision_states,
    }
