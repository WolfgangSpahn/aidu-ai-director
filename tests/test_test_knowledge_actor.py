from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.context import Context
from aidu.ai.director.actors.test_knowledge_actor import TestKnowledgeAnalyzer


def test_test_knowledge_actor_aggregates_scored_questions_by_target():
    context = Context()
    _, context = TestKnowledgeAnalyzer().run(TextArtifact(
        producer="test",
        step=0,
        content='{"questions": ['
        '{"id":"q1","correct":true,"targets":["atomic-structure"]},'
        '{"id":"q2","correct":false,"targets":["atomic-structure"]},'
        '{"id":"q3","correct":true,"targets":["electron-arrangement"]}'
        ']}'
    ), context)

    assert context.state.data["StudentProgress"] == {
        "atomic-structure": {
            "mastery": 0.5,
            "positive_evidence": 1.0,
            "negative_evidence": 1.0,
        },
        "electron-arrangement": {
            "mastery": 1.0,
            "positive_evidence": 1.0,
            "negative_evidence": 0.0,
        },
    }


def test_test_knowledge_actor_falls_back_to_question_id_without_targets():
    context = Context()
    _, context = TestKnowledgeAnalyzer().run(TextArtifact(
        producer="test",
        step=0,
        content='{"questions":[{"id":"q1","correct":true}]}',
    ), context)

    assert context.state.data["StudentProgress"]["q1"]["mastery"] == 1.0


def test_test_knowledge_actor_includes_all_authoritative_targets():
    context = Context()
    _, context = TestKnowledgeAnalyzer().run(TextArtifact(
        producer="test",
        step=0,
        content='{"targets":["one","two","three"],"questions":['
        '{"id":"q1","correct":true,"targets":["one","not-configured"]}]}'
    ), context)

    state = context.state.data["StudentProgress"]
    assert list(state) == ["one", "two", "three"]
    assert state["one"]["mastery"] == 1.0
    assert state["two"]["mastery"] == 0.0
    assert state["three"]["mastery"] == 0.0
