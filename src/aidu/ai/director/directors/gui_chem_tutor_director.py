# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""Director composition for the GUI chemistry tutor.

The actor implementations live in ``aidu.ai.director.actors``. This module is
only responsible for assembling the Director graph: which actors participate and
how messages route between them.
"""

from __future__ import annotations

from aidu.ai.director.actors.gui_chem_tutor_actor import GuiChemTutorActor
from aidu.ai.director.actors.gui_user_actor import GuiUserActor
from aidu.ai.director.director import Director


def build_gui_chem_tutor_director(client=None, chem_tutor_port: int = 8003) -> Director:
    """Build the two-actor GUI chemistry tutor Director graph."""
    gui_user_actor = GuiUserActor()
    chem_tutor_actor = GuiChemTutorActor(client=client)

    director = Director()
    director.register(actor=gui_user_actor)
    director.register(actor=chem_tutor_actor, port=chem_tutor_port)
    director.on_input(gui_user_actor.name).send_to(chem_tutor_actor.name)
    director.on_input(chem_tutor_actor.name).send_to(gui_user_actor.name)
    return director
