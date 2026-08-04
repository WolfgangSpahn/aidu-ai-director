"""Replay assessment agents over an existing dialog without generating tutor turns."""

from __future__ import annotations

from typing import Any

from aidu.ai.agents.ai_supervisor import AiSupervisor
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessor
from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.context import Context, Message
from aidu.ai.core.session import SessionContext

from .accessor_router import AssessorRouter, _run_assessment
from .helpers import (
    apply_target_assessment,
    update_context_with_belief_assessment,
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


def assess_unmatched_final_tutor(
    actor,
    turns: list[dict[str, Any]],
    domain_targets: list[dict[str, Any]],
    *,
    progress=None,
    belief: StudentBelief | None = None,
) -> dict[str, Any] | None:
    """Assess a trailing tutor turn without pretending a learner outcome exists."""
    if not turns or turns[-1].get("role") != "assistant":
        return None
    tutor_index = len(turns) - 1
    _, _, supervisor_agent = _configured_assessors(actor)
    session_context = SessionContext(on_air=True, domain_targets=domain_targets)
    if progress is None:
        progress = session_context.initial_student_knowledge_progress()
        for turn in reversed(turns):
            state = turn.get("backend_knowledge_progress_state")
            if isinstance(state, dict):
                progress = progress.model_validate(state)
                break
    if belief is None:
        belief = StudentBelief()
        for turn in reversed(turns):
            state = turn.get("backend_belief_state")
            if isinstance(state, dict):
                belief = StudentBelief.model_validate(state)
                break
    context = Context()
    context.state.data.update({
        "SessionContext": session_context,
        "TurnIndex": tutor_index,
        "LastTutorTurnIndex": tutor_index,
        "StudentKnowledgeProgress": progress,
        "StudentBelief": belief,
        "AppletState": {},
    })
    context.trace.messages = [
        cleaned
        for turn in turns
        if (cleaned := Message.clean_dialog_record(turn)) is not None
    ]
    assessor_context = context.for_assessor()
    result = _run_assessment(
        agent=supervisor_agent,
        context=assessor_context,
        prompt_params=supervisor_agent.build_prompt_args(
            context=assessor_context,
            current_student_message="No subsequent learner turn is available.",
            outcome_evidence_available=False,
        ),
        instruction="Assess the unmatched final AI tutor response without learner outcome evidence.",
        max_tokens=1024,
    )
    update_context_with_supervision_assessment(
        result,
        context,
        assessed_tutor_turn_index=tutor_index,
        outcome_student_turn_index=None,
        outcome_evidence_available=False,
    )
    return {
        "turn_index": tutor_index,
        "assessed_tutor_turn_index": tutor_index,
        "role": "assistant",
        "actor": turns[-1].get("actor"),
        "supervision_state": context.state.data["SupervisorState"].model_dump(mode="json"),
    }


def reassess_dialog(actor, turns: list[dict[str, Any]], domain_targets: list[dict[str, Any]]) -> dict[str, Any]:
    """Return fresh per-student-turn states using the actor's configured assessors."""
    target_agent, belief_agent, supervisor_agent = _configured_assessors(actor)
    session_context = SessionContext(on_air=True, domain_targets=domain_targets)
    progress = session_context.initial_student_knowledge_progress()
    belief = StudentBelief()
    knowledge_states: list[dict[str, Any]] = []
    belief_states: list[dict[str, Any]] = []
    supervision_states: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(turns):
        if turn.get("role") != "user":
            continue
        current_message = str(turn.get("content") or "").strip()
        if not current_message:
            continue
        prior_turns = [
            cleaned
            for item in turns[:turn_index]
            if (cleaned := Message.clean_dialog_record(item)) is not None
        ]
        context = Context()
        context.state.data.update({
            "SessionContext": session_context,
            "TurnIndex": turn_index,
            "StudentKnowledgeProgress": progress,
            "StudentBelief": belief,
            "AppletState": turn.get("applet_input") or {},
        })
        context.trace.messages = prior_turns
        assessor_context = context.model_copy(deep=True)

        target_result = _run_assessment(
            agent=target_agent,
            context=assessor_context.model_copy(deep=True),
            prompt_params=target_agent.build_prompt_args(context=assessor_context, current_message=current_message),
            instruction="Reassess the archived learner message for learning evidence.",
            max_tokens=512,
        )
        apply_target_assessment(target_result, context, current_message=current_message)
        progress = context.state.data["StudentKnowledgeProgress"]

        belief_result = _run_assessment(
            agent=belief_agent,
            context=assessor_context.model_copy(deep=True),
            prompt_params=belief_agent.build_prompt_args(context=assessor_context, current_message=current_message),
            instruction="Reassess the archived learner message's belief state.",
            max_tokens=512,
        )
        update_context_with_belief_assessment(belief_result, context)
        belief = context.state.data["StudentBelief"]

        tutor_index = next(
            (index for index in range(turn_index - 1, -1, -1) if turns[index].get("role") == "assistant"),
            None,
        )
        if tutor_index is not None:
            assessor_context.state.data["LastTutorTurnIndex"] = tutor_index
            assessor_context.state.data["OutcomeStudentTurnIndex"] = turn_index
            supervisor_result = _run_assessment(
                agent=supervisor_agent,
                context=assessor_context.model_copy(deep=True),
                prompt_params=supervisor_agent.build_prompt_args(
                    context=assessor_context,
                    current_student_message=current_message,
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
            supervision_states.append({
                "turn_index": turn_index,
                "assessed_tutor_turn_index": tutor_index,
                "role": "user",
                "actor": turn.get("actor"),
                "supervision_state": context.state.data["SupervisorState"].model_dump(mode="json"),
            })

        knowledge_states.append({
            "turn_index": turn_index,
            "assessed_student_turn_index": turn_index,
            "role": "user",
            "actor": turn.get("actor"),
            "knowledge_state": progress.model_dump(mode="json"),
        })
        belief_states.append({
            "turn_index": turn_index,
            "assessed_student_turn_index": turn_index,
            "role": "user",
            "actor": turn.get("actor"),
            "belief_state": belief.model_dump(mode="json"),
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
