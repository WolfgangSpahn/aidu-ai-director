# dummy actors for testing

from aidu.ai.director.actor import Actor
from aidu.ai.director.actor import ActorResult


class HumanStudentActor(Actor):
    id = "human_student"

    def perform(self, task, state):

        print(f"Student: {task}")

        return ActorResult()


class TutorActor(Actor):
    id = "ai_tutor"

    def perform(self, task, state):

        print(f"Tutor: {task}")

        attempts = state.data.get("attempts", 0) + 1

        return ActorResult(
            updates={
                "attempts": attempts,
                "solved": attempts >= 3,
            }
        )
