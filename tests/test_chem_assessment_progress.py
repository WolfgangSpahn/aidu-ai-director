import pytest

from aidu.ai.core.context import Context
from aidu.ai.actor.actor import _normalize_student_progress
from aidu.ai.director.actors.gui_chem_tutor_actor import (
    _last_tutor_turn_from_context,
    _student_progress_from_context,
    apply_chem_assessment,
)


TARGET = "proton-identity"


def context_with_target() -> Context:
    context = Context()
    context.state.data["StudentProgress"] = _student_progress_from_context({
        "domain_targets": [{"id": TARGET, "text": "Identify elements from proton count."}],
    })
    return context


def assess(context: Context, polarity: str = "+", strength: str = "m") -> float:
    apply_chem_assessment(
        assessment={"e": [{"i": TARGET, "p": polarity, "s": strength, "q": "evidence"}]},
        context=context,
    )
    return context.state.data["StudentProgress"][TARGET]["mastery"]


def test_medium_positive_evidence_moves_fast_then_diminishes():
    context = context_with_target()

    trajectory = [assess(context) for _ in range(8)]

    assert trajectory == pytest.approx([
        1 / 3,
        1 / 2,
        3 / 5,
        2 / 3,
        5 / 7,
        3 / 4,
        7 / 9,
        4 / 5,
    ])
    gains = [trajectory[0], *[
        trajectory[index] - trajectory[index - 1]
        for index in range(1, len(trajectory))
    ]]
    assert gains == sorted(gains, reverse=True)


def test_strength_controls_evidence_weight():
    weak = context_with_target()
    medium = context_with_target()
    strong = context_with_target()

    assert assess(weak, strength="w") == pytest.approx(0.2)
    assert assess(medium, strength="m") == pytest.approx(1 / 3)
    assert assess(strong, strength="s") == pytest.approx(0.5)


def test_negative_evidence_reduces_mastery_and_slows_recovery():
    context = context_with_target()
    assert assess(context, strength="s") == pytest.approx(0.5)

    reduced = assess(context, polarity="-", strength="m")
    recovered = assess(context, polarity="+", strength="m")

    assert reduced == pytest.approx(0.4)
    assert recovered == pytest.approx(0.5)
    state = context.state.data["StudentProgress"][TARGET]
    assert state == pytest.approx({
        "mastery": 0.5,
        "positive_evidence": 6.0,
        "negative_evidence": 6.0,
    })


def test_unclear_evidence_does_not_change_state():
    context = context_with_target()
    before = dict(context.state.data["StudentProgress"][TARGET])

    assess(context, polarity="?", strength="s")

    assert context.state.data["StudentProgress"][TARGET] == before


def test_actor_end_state_preserves_mastery_and_evidence_counts():
    context = context_with_target()
    assess(context, strength="m")

    emitted = _normalize_student_progress(context.state.data["StudentProgress"])

    assert emitted[TARGET] == pytest.approx({
        "mastery": 1 / 3,
        "positive_evidence": 2.0,
        "negative_evidence": 4.0,
    })


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

    assert _last_tutor_turn_from_context(context) == (
        "Tutor: How many protons identify carbon?"
    )
