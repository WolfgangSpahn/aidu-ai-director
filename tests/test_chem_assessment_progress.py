import pytest
from types import SimpleNamespace

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
from aidu.ai.director.actors.GuiChemTutorActor.reassessment import (
    _configured_assessors,
    assess_unmatched_final_tutor,
)
from aidu.ai.director.actors.GuiChemTutorActor.accessor_router import (
    learner_evidence_text,
    smoke_test,
)
from aidu.ai.core.artifacts import AppletArtifact
from aidu.ai.director.actors.GuiChemTutorActor.accessor_router import AssessorRouter

TARGET = "proton-identity"


def test_smoke_test_runs_all_three_assessors_through_router(monkeypatch):
    calls = []

    def fake_run_assessment(**kwargs):
        agent = kwargs["agent"]
        calls.append(type(agent))
        if isinstance(agent, LearningTargetAssessor):
            return {"evidence": [{
                "target": TARGET,
                "direction": "positive",
                "strength": "moderate",
                "confidence": 0.9,
                "evidence_type": "explanation",
                "response_mode": "deliberate",
                "support_level": "independent",
                "quote": "atomic number is six",
            }], "review": False}
        if isinstance(agent, StudentBeliefAssessor):
            return {"evidence": [{
                "speech_act": "express_understanding",
                "strength": "moderate",
                "confidence": 0.8,
                "quote": "Carbon",
            }], "review": False}
        return {
            dimension: {"fit": 0.8, "reason": "Appropriate for this turn."}
            for dimension in (
                "factual_fit", "goal_alignment", "knowledge_alignment",
                "belief_alignment", "scaffolding_fit",
            )
        }

    monkeypatch.setattr(
        AssessorRouter,
        "_run_assessment",
        staticmethod(fake_run_assessment),
    )

    result = smoke_test(
        tutor_turn="Which element has six protons?",
        student_turn="Carbon, because its atomic number is six.",
        learning_targets=[
            {"id": TARGET, "text": "Identify elements from proton count."}
        ],
        client=SimpleNamespace(),
    )

    assert set(calls) == {LearningTargetAssessor, StudentBeliefAssessor, AiSupervisor}
    assert result["targets"][0]["id"] == TARGET
    assert result["assessments"]["knowledge"]["evidence"][0]["target"] == TARGET
    assert result["assessments"]["belief"]["evidence"][0]["speech_act"] == "express_understanding"
    assert result["initial_state"]["knowledge"][TARGET]["mastery"] == 0.5
    assert result["final_state"]["knowledge"][TARGET]["mastery"] > 0.5
    assert result["initial_state"]["belief"]["confidence"] == 0.5
    assert result["final_state"]["belief"]["confidence"] > 0.5
    assert result["final_state"]["supervision"]["factual_fit"]["fit"] == 0.8


def test_reassessment_resolves_assessors_from_live_nested_router():
    target = object.__new__(LearningTargetAssessor)
    belief = object.__new__(StudentBeliefAssessor)
    supervisor = object.__new__(AiSupervisor)
    router = AssessorRouter((target, belief, supervisor))

    assert _configured_assessors(SimpleNamespace(agents=[router])) == (
        target,
        belief,
        supervisor,
    )


def test_reassessment_reports_missing_assessors_without_stop_iteration():
    with pytest.raises(ValueError, match="does not expose"):
        _configured_assessors(SimpleNamespace(agents=[]))


def test_unmatched_final_tutor_gets_provisional_supervision(monkeypatch):
    assessment = {
        dimension: {"fit": 0.8, "reason": "Appropriate without outcome evidence."}
        for dimension in (
            "factual_fit",
            "goal_alignment",
            "knowledge_alignment",
            "belief_alignment",
            "scaffolding_fit",
        )
    }
    captured = {}

    def fake_run_assessment(**kwargs):
        captured.update(kwargs["prompt_params"])
        return assessment

    monkeypatch.setattr(
        "aidu.ai.director.actors.GuiChemTutorActor.accessor_router.AssessorRouter._run_assessment",
        fake_run_assessment,
    )
    actor = SimpleNamespace(agents=[
        object.__new__(LearningTargetAssessor),
        object.__new__(StudentBeliefAssessor),
        object.__new__(AiSupervisor),
    ])
    turns = [
        {"role": "user", "content": "It gains one electron.", "actor": "Buddy"},
        {"role": "assistant", "content": "What rule do you notice?", "actor": "Tutor"},
    ]

    snapshot = assess_unmatched_final_tutor(
        actor,
        turns,
        [{"id": TARGET, "text": "Identify elements from proton count."}],
    )

    assert snapshot is not None
    assert snapshot["assessed_tutor_turn_index"] == 1
    assert snapshot["supervision_state"]["outcome_student_turn_index"] is None
    assert snapshot["supervision_state"]["outcome_evidence_available"] is False
    assert "outcome_evidence_available" not in captured
    assert "current_student_message" not in captured


def test_terminal_supervision_is_not_created_when_a_learner_turn_follows():
    actor = SimpleNamespace(agents=[object.__new__(AiSupervisor)])

    assert assess_unmatched_final_tutor(
        actor,
        [
            {"role": "assistant", "content": "What do you notice?"},
            {"role": "user", "content": "The charge changed."},
        ],
        [],
    ) is None


def test_belief_assessor_prompt_does_not_expose_final_belief_dimensions():
    context = Context()
    context.state.data["StudentBelief"] = StudentBelief(confusion=0.8)
    context.state.data["AppletState"] = {}

    params = StudentBeliefAssessor.build_prompt_args(
        context=context,
        current_message="I do not understand what to do next.",
    )

    assert "prior_belief" not in params
    assert params["current_message"] == "I do not understand what to do next."


def test_belief_assessment_updates_context_from_contract():
    context = Context()
    original = StudentBelief()
    context.state.data["StudentBelief"] = original
    update_context_with_belief_assessment(
        assessment={
            "evidence": [{
                "speech_act": "express_understanding",
                "strength": "strong",
                "confidence": 1.0,
                "quote": "I am certain",
            }],
            "review": False,
        },
        context=context,
        current_message="I am certain this is carbon.",
    )

    updated = context.state.data["StudentBelief"]
    assert updated.confidence == pytest.approx(original.confidence + 0.15)
    assert updated.confusion == pytest.approx(original.confusion - 0.15)
    assert context.control.data["student_belief_assessment"]["derived_belief"] == updated.model_dump()

    with pytest.raises(ValueError):
        update_context_with_belief_assessment(
            assessment={"evidence": [{"speech_act": "express_understanding"}]},
            context=context,
        )


@pytest.mark.parametrize("include_valid", [False, True])
def test_unsupported_belief_act_is_stored_for_review_without_aborting(include_valid):
    context = Context()
    original = StudentBelief()
    context.state.data["StudentBelief"] = original
    unsupported = {"speech_act": "express_surprise", "strength": "strong",
                   "confidence": 0.9, "quote": "Wow"}
    evidence = [unsupported]
    if include_valid:
        evidence.append({"speech_act": "explain", "strength": "moderate",
                         "confidence": 1.0, "quote": "because protons define the element"})
    raw = {"evidence": evidence, "review": False}
    update_context_with_belief_assessment(
        raw, context, current_message="Wow, because protons define the element",
    )
    stored = context.control.data["student_belief_assessment"]
    assert stored["review"] is True
    assert stored["rejected_evidence"] == [{**unsupported, "reason": "Unsupported speech act"}]
    assert len(stored["evidence"]) == int(include_valid)
    assert context.state.data["StudentBelief"].self_explanation == pytest.approx(
        original.self_explanation + (0.1 if include_valid else 0)
    )
    assert context.state.data["StudentBelief"].confidence == original.confidence
    assert raw["review"] is False
    assert raw["evidence"][0] == unsupported


def test_unknown_belief_act_does_not_hide_other_validation_errors():
    context = Context()
    context.state.data["StudentBelief"] = StudentBelief()
    with pytest.raises(ValueError):
        update_context_with_belief_assessment(
            {"evidence": [{"speech_act": "express_surprise", "strength": "strong",
                           "confidence": 2.0, "quote": "Wow"}], "review": False},
            context, current_message="Wow",
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
    )

    assert "I think it is carbon." not in params["history"]
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
    assert context.control.data["emit_supervision_state"] is True

    invalid = {**valid, "factual_fit": {**valid["factual_fit"], "fit": 1.2}}
    with pytest.raises(ValueError):
        update_context_with_supervision_assessment(
            assessment=invalid,
            context=context,
        )


def context_with_target() -> Context:
    context = Context()
    context.state.data["TurnIndex"] = 1
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


def assess(
    context: Context,
    direction: str = "positive",
    strength: str = "moderate",
    support_level: str = "independent",
    quote: str = "evidence",
) -> float:
    apply_target_assessment(
        assessment={
            "evidence": [
                {
                    "target": TARGET,
                    "direction": direction,
                    "strength": strength,
                    "confidence": 1.0,
                    "evidence_type": "explanation",
                    "response_mode": "deliberate",
                    "support_level": support_level,
                    "quote": quote,
                }
            ],
            "review": False,
        },
        context=context,
    )
    return context.state.data["StudentKnowledgeProgress"].root[TARGET].mastery


def test_repeated_identical_claim_is_counted_only_once():
    context = context_with_target()

    trajectory = [assess(context) for _ in range(3)]

    assert trajectory == [0.6875, 0.6875, 0.6875]
    state = context.state.data["StudentKnowledgeProgress"].root[TARGET]
    assert state.positive_evidence == pytest.approx(0.6)
    assert state.turn_assessment_count == 1


def test_same_claim_on_a_later_turn_is_new_evidence():
    context = context_with_target()
    first = assess(context)
    context.state.data["TurnIndex"] += 1
    second = assess(context)

    assert second > first
    assert context.state.data["StudentKnowledgeProgress"].root[TARGET].turn_assessment_count == 2


def test_supervision_hindsight_reason_is_reframed_as_outcome():
    context = Context()
    assessment = {
        dimension: {"fit": 0.5, "reason": "The tutor doesn't explicitly address the learner's later question."}
        for dimension in (
            "factual_fit", "goal_alignment", "knowledge_alignment",
            "belief_alignment", "scaffolding_fit",
        )
    }
    update_context_with_supervision_assessment(assessment, context)

    state = context.control.data["ai_supervision_assessment"]
    assert all("learner outcome suggests" in state[dimension]["reason"] for dimension in assessment)


def test_strength_controls_evidence_weight():
    weak = context_with_target()
    medium = context_with_target()
    strong = context_with_target()

    assess(weak, strength="weak")
    assess(medium, strength="moderate")
    assess(strong, strength="strong")

    assert weak.state.data["StudentKnowledgeProgress"].root[TARGET].positive_evidence == pytest.approx(0.25)
    assert medium.state.data["StudentKnowledgeProgress"].root[TARGET].positive_evidence == pytest.approx(0.6)
    assert strong.state.data["StudentKnowledgeProgress"].root[TARGET].positive_evidence == pytest.approx(1.0)


def test_negative_evidence_reduces_mastery_and_slows_recovery():
    context = context_with_target()
    assert assess(context, strength="strong", quote="first claim") == 0.75

    reduced = assess(context, direction="negative", quote="second claim")
    recovered = assess(context, quote="third claim")

    assert reduced == pytest.approx(1.5 / 2.6)
    assert recovered == pytest.approx(2.1 / 3.2)
    state = context.state.data["StudentKnowledgeProgress"].root[TARGET]
    assert state.positive_evidence == pytest.approx(1.6)
    assert state.negative_evidence == pytest.approx(0.6)
    assert state.source_count == 3
    assert state.turn_assessment_count == 3


def test_invalid_direction_is_rejected():
    context = context_with_target()
    with pytest.raises(ValueError):
        assess(context, direction="unclear")


def test_actor_end_state_preserves_mastery_and_evidence_counts():
    context = context_with_target()
    assess(context)

    emitted = context.state.data["StudentKnowledgeProgress"].clamped()

    state = emitted.root[TARGET]
    assert state.mastery == 0.6875
    assert state.positive_evidence == pytest.approx(0.6)
    assert state.negative_evidence == 0.0
    assert state.turn_assessment_count == 1


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


def test_assessor_never_uses_applet_telemetry_as_learner_authored_text():
    context = Context()
    context.state.data["CurrentStudentMessage"] = "I have added something, but still see '?'."
    artifact = AppletArtifact(
        producer="user",
        step=0,
        content={"protons": 0, "neutrons": 1},
    )

    assert learner_evidence_text(artifact, context) == (
        "I have added something, but still see '?'."
    )


def test_assessor_uses_only_teacher_defined_target_meanings():
    prompt = LearningTargetAssessor.prompt_template

    assert "Interpret each ID only through its teacher-defined text." in prompt
    assert "{learning_targets}" in prompt
    assert "{history}" in prompt
    assert "{tutor_question}" in prompt
    assert "atomic-number-mass-isotopes" not in prompt


def test_assessor_strength_rubric_credits_concise_relationships():
    prompt = LearningTargetAssessor.prompt_template

    assert "a correct relationship, prediction, comparison, or explanation" in prompt
    assert "a general rule stated in the learner's own words" in prompt
    assert "spelling mistakes, and imperfect grammar do not reduce strength" in prompt


def test_assessor_prompt_rejects_applet_observation_as_inferred_understanding():
    prompt = LearningTargetAssessor.prompt_template
    prompt_flat = " ".join(prompt.split())

    assert "Prefer no evidence over speculative evidence." in prompt_flat
    assert "does not by itself demonstrate the underlying concept" in prompt_flat
    assert "must never add knowledge, reasoning, or particle identification" in prompt_flat
    assert "at most weak evidence" in prompt_flat
    assert "Merely observing the result" in prompt_flat
    assert "Machine-generated status text" in prompt_flat
    assert "A mismatch must never receive positive evidence" in prompt_flat
    assert 'A vague claim such as "I added something"' in prompt_flat
    assert "First identify exactly what TUTOR_QUESTION_OR_INSTRUCTION asked" in prompt_flat
    assert "do not assess an unrelated fact merely because it appears in the applet state" in prompt_flat


def test_assessor_prompt_routes_uncertainty_only_to_directly_tested_target():
    prompt = LearningTargetAssessor.prompt_template
    prompt_flat = " ".join(prompt.split())

    assert 'An uncertainty response such as "I don\'t know"' in prompt_flat
    assert "only for the specific target directly tested by TUTOR_QUESTION_OR_INSTRUCTION" in prompt_flat
    assert 'support_level "answer_revealed"' in prompt_flat


def test_assessor_prompt_does_not_treat_interface_failure_as_conceptual_evidence():
    prompt_flat = " ".join(LearningTargetAssessor.prompt_template.split())

    assert "the failure itself directly demonstrates knowledge" in prompt_flat
    assert "placement, visibility, or motor difficulty alone is not evidence" in prompt_flat
    assert '"I cannot add an electron" does not contradict understanding' in prompt_flat
    assert "Never invent an explanation for why the attempt failed" in prompt_flat


def test_assessor_prompt_receives_arbitrary_target_text_and_history():
    context = Context()
    context.state.data["StudentKnowledgeProgress"] = StudentKnowledgeProgress.model_validate(
        {
                "teacher-defined-target": {
                    "mastery": 0.5,
                    "positive_evidence": 0.0,
                    "negative_evidence": 0.0,
                    "entry_prior": 0.5,
                    "entry_weight": 0.0,
                    "source_count": 0,
                    "turn_assessment_count": 0,
                    "last_updated_turn": None,
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
    assert prompt["tutor_question"] == "Tutor: What pattern do you notice?"
    assert prompt["current_message"] == "It doubles each time."


def test_assessment_logs_turn_numbers_and_raw_invalid_result(caplog):
    import logging

    context = Context()
    context.state.data.update({"LastTutorTurnIndex": 14, "OutcomeStudentTurnIndex": 15})
    agent = SimpleNamespace(id="supervisor", run=lambda **kwargs: (
        SimpleNamespace(content=lambda: "invalid raw result"), context,
    ))
    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="non-JSON"):
        AssessorRouter._run_assessment(agent=agent, context=context, prompt_params={},
                                      instruction="Assess tutor", max_tokens=1024)
    label = context.control.data["assessment_log_label"]
    assert "tutor_turn=#15 learner_turn=#16" in label
    assert f"Assessment started {label}" in caplog.text
    assert f"Assessment raw result {label}\ninvalid raw result" in caplog.text


def test_reassessment_preserves_test_prior_and_default_entry_belief(monkeypatch):
    from aidu.ai.director.actors.GuiChemTutorActor import reassessment
    from aidu.ai.core.supervisor import SUPERVISOR_DIMENSIONS

    agent = SimpleNamespace(build_prompt_args=lambda **kwargs: {})
    monkeypatch.setattr(reassessment, "_configured_assessors", lambda actor: (agent, agent, agent))
    def assess(**kwargs):
        if "tutor response" in kwargs["instruction"]:
            return {key: {"fit": 0.5, "reason": "Test"} for key in SUPERVISOR_DIMENSIONS}
        return {"evidence": [], "review": False}
    monkeypatch.setattr(AssessorRouter, "_run_assessment", staticmethod(assess))
    result = reassessment.reassess_dialog(None, [
        {"role": "assistant", "content": "Welcome", "backend_knowledge_state_kind": "prior",
         "backend_knowledge_progress_state": {TARGET: {"mastery": 0.8, "entry_prior": 0.8, "positive_evidence": 0, "negative_evidence": 0, "entry_weight": 1, "source_count": 0, "turn_assessment_count": 0, "last_updated_turn": None, "evidence_fingerprints": []}}},
        {"role": "user", "content": "Hello"},
    ], [{"id": TARGET, "text": "Identify protons"}])
    assert result["knowledge_states"][0]["state_kind"] == "prior"
    assert result["knowledge_states"][0]["knowledge_state"][TARGET]["mastery"] == 0.8
    assert result["knowledge_states"][1]["knowledge_state"][TARGET]["mastery"] == 0.8
    assert result["belief_states"][0]["belief_state"] == StudentBelief().model_dump()
