from aidu.ai.actor.actor import RunRequest
from aidu.ai.director.actors.gui_chem_tutor_actor import GuiChemLlmTutor, GuiChemTutorActor
from fastapi.testclient import TestClient


class FakeClient:
    pass


def test_gui_chem_tutor_context_trace_contains_forwarded_dialog_only():
    actor = GuiChemTutorActor(client=FakeClient())
    request = RunRequest(
        message={
            "role": "user",
            "content": "Applet event: applet-periodic-table",
            "kind": "applet",
        },
        info={
            "applet_input": {
                "applet": "applet-periodic-table",
                "infoStore": {
                    "elementSymbol": "H",
                    "elementName": "Hydrogen",
                    "atomicNumber": 1,
                },
            },
            "messages": [
                {
                    "role": "assistant",
                    "content": "Welcome Anonymous to our Chemistry Periodic Table session.",
                },
                {
                    "role": "user",
                    "content": "Applet event: applet-periodic-table",
                    "kind": "applet",
                    "applet_input": {
                        "applet": "applet-periodic-table",
                        "infoStore": {
                            "elementSymbol": "H",
                            "elementName": "Hydrogen",
                            "atomicNumber": 1,
                        },
                    },
                },
            ],
        },
    )

    context = actor.build_context_from_request(request)

    assert context.trace.messages == [
        {
            "role": "assistant",
            "content": "Welcome Anonymous to our Chemistry Periodic Table session.",
        },
        {
            "role": "user",
            "content": "Applet event: applet-periodic-table with elementName=Hydrogen, elementSymbol=H, atomicNumber=1",
            "kind": "applet",
            "applet_input": {
                "applet": "applet-periodic-table",
                "infoStore": {
                    "elementSymbol": "H",
                    "elementName": "Hydrogen",
                    "atomicNumber": 1,
                },
            },
        },
    ]


def test_gui_chem_tutor_prompt_contains_active_atomic_structure_context():
    actor = GuiChemTutorActor(client=FakeClient())
    request = RunRequest(
        message={
            "role": "user",
            "content": "What is an atom?",
        },
        info={
            "session_context": {
                "subject": "chemistry",
                "subject_label": "Chemistry",
                "domain": "atomic-structure",
                "domain_label": "Atomic Structure",
                "domain_description": "Students learn how atoms are built from subatomic particles.",
                "domain_targets": [
                    {
                        "id": "atomic-particles",
                        "text": "describe protons, neutrons, and electrons.",
                    },
                ],
                "applet_id": "applet-build-an-atom",
                "applet_name": "Build an Atom",
                "applet_description": "Build atoms from protons, neutrons, and electrons.",
            },
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hi — I'm Marie, your chemistry tutor for atomic structure.",
                },
                {
                    "role": "user",
                    "content": "What is an atom?",
                },
            ],
        },
    )

    context = actor.build_context_from_request(request)
    tutor_state = context.state.data[GuiChemLlmTutor.__name__]
    tutor = next(agent for agent in actor.agents if isinstance(agent, GuiChemLlmTutor))
    system_prompt = tutor.build_system_prompt(tutor_state)[0]["content"]

    assert tutor_state["context_summary"] == "Chemistry / Atomic Structure"
    assert "Active tutoring context: Chemistry / Atomic Structure" in system_prompt
    assert "- subject: Chemistry (chemistry)" in system_prompt
    assert "- title: Atomic Structure" in system_prompt
    assert "- id: applet-build-an-atom" in system_prompt


def test_gui_chem_tutor_begin_agent_interactive_follows_debug_env(monkeypatch):
    monkeypatch.delenv("AIDU_DEBUG", raising=False)
    actor = GuiChemTutorActor(client=FakeClient())
    assert actor.agents[0].interactive is False

    monkeypatch.setenv("AIDU_DEBUG", "True")
    debug_actor = GuiChemTutorActor(client=FakeClient())
    assert debug_actor.agents[0].interactive is True


def test_gui_chem_tutor_applet_input_returns_visible_dialog_response():
    actor = GuiChemTutorActor(client=FakeClient())
    actor.agents[0].interactive = False
    client = TestClient(actor.app)

    response = client.post(
        "/run",
        json={
            "message": {
                "role": "user",
                "content": "Applet event: applet-periodic-table",
                "kind": "applet",
            },
            "info": {
                "applet_input": {
                    "applet": "applet-periodic-table",
                    "infoStore": {
                        "elementSymbol": "H",
                        "elementName": "Hydrogen",
                        "atomicNumber": 1,
                        "valenceElectrons": 1,
                        "responseExpectation": "both",
                    },
                },
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Welcome Anonymous to our Chemistry Periodic Table session.",
                    },
                    {
                        "role": "user",
                        "content": "Applet event: applet-periodic-table",
                        "kind": "applet",
                        "applet_input": {
                            "applet": "applet-periodic-table",
                            "infoStore": {
                                "elementSymbol": "H",
                                "elementName": "Hydrogen",
                                "atomicNumber": 1,
                                "valenceElectrons": 1,
                                "responseExpectation": "both",
                            },
                        },
                    },
                ],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == "You have clicked this. What was your intent"
    assert "applet" not in response.json()
    assert "applet_command" not in response.json()
