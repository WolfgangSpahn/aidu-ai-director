from aidu.ai.agents.chem_student import ChemStudent
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.context import Context
from aidu.ai.director.actors.chem_student_actor import (
    ChemStudentActor,
    RandomDenier,
    StudentBeginAgent,
)


class NoCallClient:
    pass


def test_chem_student_actor_contains_student_and_end_agent():
    actor = ChemStudentActor(client=NoCallClient())

    assert actor.name == "chem_student_actor"
    assert actor.startup is StudentBeginAgent
    assert isinstance(actor.student, ChemStudent)
    assert any(isinstance(agent, RandomDenier) for agent in actor.agents)


def test_denier_route_returns_non_answer_without_calling_llm():
    actor = ChemStudentActor(
        client=NoCallClient(),
        random_value=lambda: 0.1,
        denial_choice=lambda responses: responses[0],
    )
    context = Context()
    context.create_agent_states(actor.agents)

    result, _ = actor.run_student_turn(
        TextArtifact(producer="tutor", step=0, content="Why?"), context
    )

    assert [artifact.content for artifact in result.artifacts] == ["I don't know."]


def test_begin_routes_to_llm_for_upper_half():
    actor = ChemStudentActor(client=NoCallClient(), random_value=lambda: 0.9)
    begin = next(agent for agent in actor.agents if isinstance(agent, StudentBeginAgent))
    context = Context()
    context.create_agent_states(actor.agents)

    result, _ = begin.run(
        TextArtifact(producer="tutor", step=0, content="Why?"),
        context,
        actor.agents,
    )

    assert result.recommendations[0].target is ChemStudent


def test_persistent_student_context_keeps_routing_after_ten_steps():
    actor = ChemStudentActor(
        client=NoCallClient(),
        denial_probability=1.0,
        denial_choice=lambda responses: responses[0],
    )
    context = Context()
    context.create_agent_states(actor.agents)

    for turn in range(12):
        result, context = actor.run_student_turn(
            TextArtifact(producer="tutor", step=context.step, content=f"Question {turn}"),
            context,
        )
        assert [artifact.content for artifact in result.artifacts] == ["I don't know."]

    assert context.step > 10


def test_default_routing_contains_one_denier_in_every_two_turns():
    actor = ChemStudentActor(client=NoCallClient(), random_value=lambda: 0.9)
    begin = next(agent for agent in actor.agents if isinstance(agent, StudentBeginAgent))
    context = Context()
    context.create_agent_states(actor.agents)
    targets = []

    for turn in range(4):
        result, context = begin.run(
            TextArtifact(producer="tutor", step=turn, content="Question"),
            context,
            actor.agents,
        )
        targets.append(result.recommendations[0].target)

    assert targets == [ChemStudent, RandomDenier, ChemStudent, RandomDenier]
