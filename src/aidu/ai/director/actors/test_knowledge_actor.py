# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
"""Director actor that converts a scored test into a knowledge state."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from aidu.ai.actor.actor import Actor
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.context import Context
from aidu.ai.llm.agent import EndAgent, WorkflowAgent


def analyze_test_questions(
    questions: list[dict[str, Any]],
    target_ids: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate objective question results without controller/thread state."""
    evidence: dict[str, dict[str, float]] = defaultdict(
        lambda: {"positive_evidence": 0.0, "negative_evidence": 0.0}
    )
    authoritative_ids = [str(value).strip() for value in target_ids or [] if str(value).strip()]
    authoritative = set(authoritative_ids)
    for target_id in authoritative_ids:
        evidence[target_id]

    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = str(question.get("id") or "").strip()
        targets = question.get("targets")
        question_target_ids = [str(value).strip() for value in targets] if isinstance(targets, list) else []
        question_target_ids = [value for value in question_target_ids if value] or ([question_id] if question_id else [])
        if authoritative:
            question_target_ids = [value for value in question_target_ids if value in authoritative]
        field = "positive_evidence" if question.get("correct") is True else "negative_evidence"
        for target_id in question_target_ids:
            evidence[target_id][field] += 1.0

    knowledge_state: dict[str, dict[str, float]] = {}
    for target_id, counts in evidence.items():
        positive = counts["positive_evidence"]
        negative = counts["negative_evidence"]
        total = positive + negative
        knowledge_state[target_id] = {
            "mastery": positive / total if total else 0.0,
            "positive_evidence": positive,
            "negative_evidence": negative,
        }
    return knowledge_state


class TestKnowledgeAnalyzer(WorkflowAgent):
    target = EndAgent
    continuations = []

    def run(self, artifact, context: Context, agents=None) -> tuple[AgentResult, Context]:
        payload = json.loads(str(artifact.content or "{}"))
        knowledge_state = analyze_test_questions(payload.get("questions", []), payload.get("targets"))
        context.state.data["StudentProgress"] = knowledge_state
        result = TextArtifact(
            producer=self.id,
            step=context.step,
            content=json.dumps({"knowledge_state": knowledge_state}),
        )
        return self.result(artifacts=[result], recommendations=[
            self.register_recommendation(
                "test_assessed",
                target=EndAgent,
                continuations=[],
                utility=1.0,
                rationale="The scored test has been converted into target evidence.",
            )
        ]), context


class TestKnowledgeActor(Actor):
    def __init__(self):
        super().__init__(
            name="test_knowledge_actor",
            agents=[TestKnowledgeAnalyzer(), EndAgent()],
            startup=TestKnowledgeAnalyzer,
            description="Analyze scored poll/test responses into a knowledge state.",
            avatar="Assessment",
        )

    def execute_run(self, req, stream_callback=None) -> dict[str, Any]:
        """Execute the deterministic assessment safely in an HTTP worker thread."""
        payload = json.loads(str(req.message.content or "{}"))
        knowledge_state = analyze_test_questions(payload.get("questions", []), payload.get("targets"))
        return {
            "role": TestKnowledgeAnalyzer.__name__,
            "content": json.dumps({"knowledge_state": knowledge_state}),
            "backend_progress_state": knowledge_state,
        }
