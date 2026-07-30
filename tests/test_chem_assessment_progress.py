import pytest

from aidu.ai.core.context import Context
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.session import SessionContext
from aidu.ai.core.knowledge_progress import StudentKnowledgeProgress
from aidu.ai.agents.ai_supervisor import AiSupervisor
from aidu.ai.agents.assessment_context import last_tutor_message
from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessor
from aidu.ai.director.actors.GuiChemTutorActor.helpers import (
    apply_target_assessment,
    update_context_with_belief_assessment,
    update_context_with_supervision_assessment,
)

TARGET = "proton-identity"


def test_belief_assessor_prompt_uses_prior_canonical_belief():
    context = Context()
    context.state.data["StudentBelief"] = StudentBelief(confusion=0.8)
    context.state.data["AppletState"] = {}

    params = StudentBeliefAssessor.build_prompt_args(
        context=context,
        current_message="I do not understand what to do next.",
    )

    assert '"confusion":0.8' in params["prior_belief"]
    assert params["current_message"] == "I do not understand what to do next."


def test_belief_assessment_updates_context_from_contract():
    context = Context()
    original = StudentBelief()
    context.state.data["StudentBelief"] = original
    assessed = {field_name: 0.25 for field_name in StudentBelief.model_fields}

    update_context_with_belief_assessment(
        assessment={"belief": assessed, "review": False},
        context=context,
    )

    assert context.state.data["StudentBelief"] == StudentBelief(**assessed)

    with pytest.raises(ValueError):
        update_context_with_belief_assessment(
            assessment={"belief": {"confidence": 0.9}},
            context=context,
        )


def test_supervisor_prompt_uses_applet_state_before_generated_reply():
    context = context_with_target()
    context.state.data["SessionContext"] = SessionContext(
        on_air=True,
        domain_targets=[{"id": TARGET, "text": "Identify elements from proton count."}],
    )
    context.state.data["StudentBelief"] = StudentBelief(confusion=0.7)
    context.state.data["AppletState"] = {
        "applet": "build-an-atom",
        "infoStore": {"protonCount": 1},
    }
    context.trace.messages = [
        {
            "role": "user",
            "kind": "applet",
            "applet_input": {
                "applet": "build-an-atom",
                "infoStore": {"protonCount": 0},
            },
        },
        {"role": "assistant", "content": "Which element has six protons?"},
        {
            "role": "user",
            "kind": "applet",
            "applet_input": {
                "applet": "build-an-atom",
                "infoStore": {"protonCount": 1},
            },
        },
    ]

    params = AiSupervisor.build_prompt_args(
        context=context,
        current_student_message="I think it is carbon.",
    )

    assert params["current_student_message"] == "I think it is carbon."
    assert params["last_tutor_message"] == "Tutor: Which element has six protons?"
    assert '"confusion":0.7' in params["student_belief"]
    assert TARGET in params["teacher_targets"]
    assert '"protonCount": 0' in params["applet_state_at_tutor_turn"]
    assert '"protonCount": 1' not in params["applet_state_at_tutor_turn"]


def test_supervision_assessment_updates_context_from_contract():
    context = Context()
    valid = {
        dimension: {
            "fit": 0.8,
            "reason": "It matches the supplied context.",
        }
        for dimension in (
            "factual_fit",
            "goal_alignment",
            "knowledge_alignment",
            "belief_alignment",
            "scaffolding_fit",
        )
    }

    update_context_with_supervision_assessment(
        assessment=valid,
        context=context,
    )

    assert context.state.data["SupervisorState"].factual_fit.fit == 0.8

    invalid = {**valid, "factual_fit": {**valid["factual_fit"], "fit": 1.2}}
    with pytest.raises(ValueError):
        update_context_with_supervision_assessment(
            assessment=invalid,
            context=context,
        )


def context_with_target() -> Context:
    context = Context()
    session_context = SessionContext(
        on_air=True,
        domain_targets=[
            {"id": TARGET, "text": "Identify elements from proton count."}
        ],
    )
    context.state.data["StudentKnowledgeProgress"] = (
        session_context.initial_student_knowledge_progress()
    )
    return context


def assess(context: Context, polarity: str = "+", strength: str = "m") -> float:
    apply_target_assessment(
        assessment={"e": [{"i": TARGET, "p": polarity, "s": strength, "q": "evidence"}]},
        context=context,
    )
    return context.state.data["StudentKnowledgeProgress"].root[TARGET].mastery


def test_medium_positive_evidence_moves_fast_then_diminishes():
    context = context_with_target()

    trajectory = [assess(context) for _ in range(8)]

    assert trajectory == pytest.approx(
        [
            1 / 3,
            1 / 2,
            3 / 5,
            2 / 3,
            5 / 7,
            3 / 4,
            7 / 9,
            4 / 5,
        ]
    )
    gains = [trajectory[0], *[trajectory[index] - trajectory[index - 1] for index in range(1, len(trajectory))]]
    assert gains == sorted(gains, reverse=True)


def test_strength_controls_evidence_weight():
    weak = context_with_target()
    medium = context_with_target()
    strong = context_with_target()

    assert assess(weak, strength="w") == pytest.approx(0.2)
    assert assess(medium, strength="m") == pytest.approx(1 / 3)
    assert assess(strong, strength="s") == pytest.approx(0.5)


def test_applet_only_evidence_is_small_compared_with_explained_evidence():
    applet_only = context_with_target()
    explained = context_with_target()

    apply_target_assessment(
        assessment={"e": [{"i": TARGET, "p": "+", "s": "s", "q": "applet state"}]},
        context=applet_only,
        evidence_scale=0.1,
    )
    apply_target_assessment(
        assessment={"e": [{"i": TARGET, "p": "+", "s": "s", "q": "learner explanation"}]},
        context=explained,
    )

    assert applet_only.state.data["StudentKnowledgeProgress"].root[TARGET].positive_evidence == pytest.approx(0.4)
    assert explained.state.data["StudentKnowledgeProgress"].root[TARGET].positive_evidence == pytest.approx(4.0)


def test_negative_evidence_reduces_mastery_and_slows_recovery():
    context = context_with_target()
    assert assess(context, strength="s") == pytest.approx(0.5)

    reduced = assess(context, polarity="-", strength="m")
    recovered = assess(context, polarity="+", strength="m")

    assert reduced == pytest.approx(0.4)
    assert recovered == pytest.approx(0.5)
    state = context.state.data["StudentKnowledgeProgress"].root[TARGET]
    assert state.model_dump() == pytest.approx(
        {
            "mastery": 0.5,
            "positive_evidence": 6.0,
            "negative_evidence": 6.0,
        }
    )


def test_unclear_evidence_does_not_change_state():
    context = context_with_target()
    before = context.state.data["StudentKnowledgeProgress"].root[TARGET].model_copy()

    assess(context, polarity="?", strength="s")

    assert context.state.data["StudentKnowledgeProgress"].root[TARGET] == before


def test_actor_end_state_preserves_mastery_and_evidence_counts():
    context = context_with_target()
    assess(context, strength="m")

    emitted = context.state.data["StudentKnowledgeProgress"].clamped()

    assert emitted.root[TARGET].model_dump() == pytest.approx(
        {
            "mastery": 1 / 3,
            "positive_evidence": 2.0,
            "negative_evidence": 4.0,
        }
    )


def test_assessor_keeps_tutor_question_when_applet_event_precedes_student_reply():
    context = Context()
    context.trace.messages = [
        {"role": "assistant", "content": "How many protons identify carbon?"},
        {
            "role": "user",
            "content": "Applet event: applet-build-an-atom",
            "kind": "applet",
        },
        {"role": "user", "content": "6"},
    ]

    assert last_tutor_message(context) == "Tutor: How many protons identify carbon?"


def test_assessor_uses_only_teacher_defined_target_meanings():
    prompt = LearningTargetAssessor.prompt_template

    assert "Interpret each ID only through its teacher-defined text." in prompt
    assert "{learning_targets}" in prompt
    assert "{history}" in prompt
    assert "{last_message}" in prompt
    assert "atomic-number-mass-isotopes" not in prompt


def test_assessor_strength_rubric_credits_concise_relationships():
    prompt = LearningTargetAssessor.prompt_template

    assert "a correct relationship, prediction, comparison, or explanation" in prompt
    assert "a general rule stated in the learner's own words" in prompt
    assert "spelling mistakes, and imperfect grammar do not reduce strength" in prompt


def test_assessor_prompt_receives_arbitrary_target_text_and_history():
    context = Context()
    context.state.data["StudentKnowledgeProgress"] = StudentKnowledgeProgress.model_validate(
        {
            "teacher-defined-target": {
                "mastery": 0.0,
                "positive_evidence": 0.0,
                "negative_evidence": 4.0,
            },
        }
    )
    context.trace.messages = [
        {"role": "assistant", "content": "What pattern do you notice?"},
        {"role": "user", "content": "The second value is twice the first."},
    ]

    context.state.data["SessionContext"] = SessionContext(
        on_air=True,
        domain_targets=[
            {
                "id": "teacher-defined-target",
                "text": "Identify and explain a doubling pattern.",
            },
        ],
    )
    context.state.data["AppletState"] = {}
    prompt = LearningTargetAssessor.build_prompt_args(
        context=context,
        current_message="It doubles each time.",
    )

    assert "Identify and explain a doubling pattern." in prompt["learning_targets"]
    assert "The second value is twice the first." in prompt["history"]
    assert prompt["last_message"] == "Tutor: What pattern do you notice?"
    assert prompt["current_message"] == "It doubles each time."
