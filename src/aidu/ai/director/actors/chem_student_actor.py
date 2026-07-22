# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
"""Actor service for the archetype-driven chemistry student."""

from __future__ import annotations

from collections import deque
import random
from collections.abc import Callable, Sequence

from aidu.ai.actor.actor import Actor
from aidu.ai.agents.chem_student import ChemStudent
from aidu.ai.archetype.archetype import Archetype, archetype_dict
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import Artifact, EndArtifact, TextArtifact
from aidu.ai.core.context import Context
from aidu.ai.llm.agent import BeginAgent, EndAgent, WorkflowAgent
from aidu.ai.llm.clients.openai import OpenAIClient


class RandomDenier(WorkflowAgent):
    """Return a short non-answer without spending an LLM request."""

    RESPONSES = (
        "I don't know.",
        "No idea.",
        "I'm not sure.",
        "I don't get it.",
        "I have no clue.",
        "Not sure at all.",
        "I can't remember.",
        "I really don't know.",
        "Maybe? I don't know.",
        "I wouldn't know.",
        "I'm lost.",
        "I can't tell.",
        "No clue, sorry.",
        "I don't have an answer.",
        "Can we do something else?",
        "Do I have to answer this?",
    )

    def __init__(self, choice: Callable[[Sequence[str]], str] = random.choice):
        self._choice = choice

    def run(self, artifact, context, agents=None):
        context.step += 1
        response = TextArtifact(
            producer=self.id,
            step=context.step,
            content=self._choice(self.RESPONSES),
        )
        recommendation = self.register_recommendation(
            "deny",
            target=EndAgent,
            rationale="The simulated student gave a short non-answer.",
        )
        return self.result([response], [recommendation]), context


class StudentBeginAgent(BeginAgent):
    """Choose an LLM answer or a cheap non-answer for each student turn."""

    def __init__(
        self,
        denial_probability: float = 0.5,
        random_value: Callable[[], float] = random.random,
    ):
        if not 0.0 <= denial_probability <= 1.0:
            raise ValueError("denial_probability must be between 0 and 1")
        super().__init__(target=ChemStudent)
        self.denial_probability = denial_probability
        self._random_value = random_value
        self._next_balanced_target: type[WorkflowAgent] | None = None

    def run(self, artifact, context, agents=None):
        if self.denial_probability == 0.5:
            if self._next_balanced_target is None:
                self.target_agent = (
                    RandomDenier if self._random_value() < 0.5 else ChemStudent
                )
                self._next_balanced_target = (
                    ChemStudent if self.target_agent is RandomDenier else RandomDenier
                )
            else:
                self.target_agent = self._next_balanced_target
                self._next_balanced_target = None
        else:
            self.target_agent = (
                RandomDenier
                if self._random_value() < self.denial_probability
                else ChemStudent
            )
        return super().run(artifact, context, agents)


class ChemStudentActor(Actor):
    """Represent a simulated student, including GUI applet infoStore output."""

    def __init__(
        self,
        client=None,
        primary_anchor: Archetype | None = None,
        secondary_anchor: Archetype | None = None,
        primary_weight: float = 0.7,
        denial_probability: float = 0.5,
        random_value: Callable[[], float] = random.random,
        denial_choice: Callable[[Sequence[str]], str] = random.choice,
    ):
        student = ChemStudent(
            client or OpenAIClient(model="gpt-5-mini"),
            primary_anchor or archetype_dict["balanced_student"],
            secondary_anchor or archetype_dict["curious_novice"],
            primary_weight,
        )
        begin = StudentBeginAgent(denial_probability, random_value)
        denier = RandomDenier(denial_choice)
        super().__init__(
            name="chem_student_actor",
            agents=[begin, student, denier, EndAgent()],
            startup=StudentBeginAgent,
            description="An archetype-driven chemistry student that simulates GUI applet interaction.",
            avatar="Student",
        )

    @property
    def student(self) -> ChemStudent:
        return next(agent for agent in self.agents if isinstance(agent, ChemStudent))

    def run_student_turn(
        self, artifact: Artifact, context: Context
    ) -> tuple[AgentResult, Context]:
        """Run the complete actor route used by the headless student engine."""
        existing_artifact_ids = set(context.artifacts)
        # Controller ``max_step`` is an absolute context step, while a virtual
        # student deliberately keeps its context across turns. Give every turn
        # its own routing budget instead of stopping forever once the persistent
        # context has passed the controller's default value of ten.
        turn_max_step = context.step + 10
        context = self.controller.run(
            start=self.startup,
            artifact=artifact,
            mailbox=deque(),
            context=context,
            max_step=turn_max_step,
        )
        artifacts = [
            produced
            for produced in context.artifacts.values()
            if produced.id not in existing_artifact_ids
            and produced.id != artifact.id
            and not isinstance(produced, EndArtifact)
        ]
        return AgentResult(artifacts=artifacts, recommendations=[]), context
