"""GUI user actor for frontend-driven Director workflows.

The GUI user actor represents the frontend boundary inside a Director graph.
Backend chat/session events enter the Director under this actor name, and tutor
responses are routed back to this actor before the backend writes them into the
session store.

This actor does not run model logic. Its only agent is ``EndAgent`` because it
marks the frontend side of the workflow: input starts here, and responses end
here.
"""

from __future__ import annotations

from aidu.ai.actor.actor import Actor
from aidu.ai.llm.agent import EndAgent


class GuiUserActor(Actor):
    """Frontend-facing user boundary in a GUI Director workflow."""

    def __init__(self):
        super().__init__(
            name="gui_user_actor",
            agents=[EndAgent()],
            startup=EndAgent,
            description="The GUI user participating in a director workflow.",
            avatar="Buddy",
        )
