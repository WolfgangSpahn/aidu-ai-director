# examples/tutoring.py

from aidu.ai.director.workflow import Workflow
from aidu.ai.director.workflow import Action


class TutoringWorkflow(Workflow):
    def run(self, director):

        yield Action(
            actor="ai_tutor",
            task="Present the task",
        )

        while not director.state.data.get("solved", False):
            yield Action(
                actor="human_student",
                task="Answer the task",
            )

            yield Action(
                actor="ai_tutor",
                task="Evaluate the answer",
            )

        yield Action(
            actor="ai_tutor",
            task="Summarize the solution",
        )


def smoke_test():
    from aidu.ai.director.director import Director

    director = Director()

    director.register(TutorActor())
    director.register(HumanStudentActor())

    director.run(TutoringWorkflow())
