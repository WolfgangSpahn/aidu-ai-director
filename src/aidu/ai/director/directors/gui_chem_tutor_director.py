# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.

"""Entry point for understanding the GUI chemistry tutor's actor graph.

Turn flow: frontend -> backend -> GuiUserActor -> GuiChemTutorActor
-> GuiUserActor -> backend -> frontend.

The Director routes messages; the tutor actor owns assessments and replies.
Read next: ``actors/gui_user_actor.py``, ``actors/GuiChemTutorActor/actor.py``,
then ``director.py`` (paths under ``src/aidu/ai/director``).

The sibling backend connects this graph to chat and applets through
``src/aidu/backend/services/director_sessions.py``. Its ``app.py`` assembles
the same graph separately, so routing changes here must also be reviewed there.
"""

from __future__ import annotations

from aidu.ai.director.actors.GuiChemTutorActor import GuiChemTutorActor
from aidu.ai.director.actors.gui_user_actor import GuiUserActor
from aidu.ai.director.director import Director


def build_gui_chem_tutor_director(client=None, chem_tutor_port: int = 8003) -> Director:
    """Build the graph; the caller starts and runs it.

    ``client`` overrides the tutor's default LLM clients.
    ``chem_tutor_port`` is the actor service port.
    """
    # Frontend boundary and tutoring workflow.
    gui_user_actor = GuiUserActor()
    chem_tutor_actor = GuiChemTutorActor(client=client)

    director = Director()
    # Only the tutor runs as a service when Director.start() is called.
    director.register(actor=gui_user_actor)
    director.register(actor=chem_tutor_actor, port=chem_tutor_port)
    # Student input goes to the tutor; its reply returns to the GUI boundary.
    director.on_input(gui_user_actor.name).send_to(chem_tutor_actor.name)
    director.on_input(chem_tutor_actor.name).send_to(gui_user_actor.name)
    return director
