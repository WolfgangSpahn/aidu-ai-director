from aidu.ai.actor.actor import RunRequest
from aidu.ai.director.actors.gui_chem_tutor_actor import GuiChemTutorActor
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
            "applet_input": {
                "applet": "applet-periodic-table",
                "infoStore": {
                    "elementSymbol": "H",
                    "elementName": "Hydrogen",
                    "atomicNumber": 1,
                },
            },
        },
        info={
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
                "applet_input": {
                    "applet": "applet-periodic-table",
                    "infoStore": {
                        "elementSymbol": "H",
                        "elementName": "Hydrogen",
                        "atomicNumber": 1,
                        "valenceElectrons": 1,
                    },
                },
            },
            "info": {
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
                            },
                        },
                    },
                ],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == "I see you clicked this. What was your intent"
    assert "applet" not in response.json()
    assert "applet_command" not in response.json()
