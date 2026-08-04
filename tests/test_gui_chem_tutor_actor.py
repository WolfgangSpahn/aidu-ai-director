from aidu.ai.actor.types import RunRequest
from aidu.ai.actor.turn_scope import JoinEndAgent
from aidu.ai.agents.chem_applet_tutor import AppletRuleResponder
from aidu.ai.core.artifacts import AppletArtifact
from aidu.ai.core.context import Context
from aidu.ai.director.actors.GuiChemTutorActor import (
    AssessorRouter,
    GuiChemLlmTutor,
    GuiChemTutorActor,
    GuiInputRouter,
)
from fastapi.testclient import TestClient


class FakeClient:
    pass


def test_applet_input_routes_to_tutor_without_symbolic_responder():
    actor = GuiChemTutorActor(client=FakeClient())

    assert GuiInputRouter.continuations == [GuiChemLlmTutor]
    assert not any(isinstance(agent, AppletRuleResponder) for agent in actor.agents)
    assert (
        AppletArtifact(
            producer="user",
            step=0,
            content={
                "applet": "applet-build-an-atom",
                "infoStore": {"protonCount": 1},
            },
        ).to_text()
        == '{"applet": "applet-build-an-atom", "infoStore": {"protonCount": 1}}'
    )


def test_gui_chem_tutor_uses_separate_tutor_and_assessor_clients():
    tutor_client = FakeClient()
    assessor_client = FakeClient()

    actor = GuiChemTutorActor(
        client=tutor_client,
        assessor_client=assessor_client,
    )

    router = next(agent for agent in actor.agents if isinstance(agent, AssessorRouter))
    tutor = next(agent for agent in actor.agents if isinstance(agent, GuiChemLlmTutor))
    assert all(
        assessor.client is assessor_client
        for assessor in (
            router.learning_target_assessor,
            router.student_belief_assessor,
            router.ai_supervisor,
        )
    )
    assert tutor.client is tutor_client


def test_gui_chem_tutor_can_use_google_client_only_for_knowledge_assessment():
    tutor_client = FakeClient()
    assessor_client = FakeClient()
    knowledge_assessor_client = FakeClient()

    actor = GuiChemTutorActor(
        client=tutor_client,
        assessor_client=assessor_client,
        knowledge_assessor_client=knowledge_assessor_client,
    )

    router = next(agent for agent in actor.agents if isinstance(agent, AssessorRouter))
    assert router.learning_target_assessor.client is knowledge_assessor_client
    assert router.student_belief_assessor.client is assessor_client
    assert router.ai_supervisor.client is assessor_client


def test_gui_tutor_function_calls_route_through_assessor_join():
    tutor = GuiChemLlmTutor(FakeClient())

    result, _ = tutor.fc_close_dialog(
        Context(),
        disposition="pause",
        final_message="See you later.",
    )

    assert tutor.terminal_target is JoinEndAgent
    assert result.recommendations[0].target is JoinEndAgent


def test_gui_chem_tutor_context_trace_contains_forwarded_dialog_only():
    actor = GuiChemTutorActor(client=FakeClient())
    request = RunRequest(
        message={
            "role": "user",
            "content": "Applet event: applet-periodic-table",
            "kind": "applet",
        },
        info={
            "session_context": {"on_air": False},
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

    assert context.control.data["emit_supervision_state"] is False
    assert context.trace.messages.root == [
        {
            "role": "assistant",
            "content": "Welcome Anonymous to our Chemistry Periodic Table session.",
        },
        {
            "role": "user",
            "content": "Applet event: applet-periodic-table with elementSymbol=H, elementName=Hydrogen, atomicNumber=1",
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


def test_gui_chem_tutor_uses_latest_backend_belief_state():
    actor = GuiChemTutorActor(client=FakeClient())
    request = RunRequest(
        message={"role": "user", "content": "I am not sure."},
        info={
            "session_context": {"on_air": False},
            "messages": [
                {
                    "role": "assistant",
                    "content": "Try changing the neutron count.",
                    "backend_belief_state": {
                        "engagement": 0.7,
                        "confidence": 0.2,
                        "confusion": 0.9,
                        "frustration": 0.1,
                        "curiosity": 0.8,
                        "self_explanation": 0.3,
                        "guessing": 0.7,
                        "help_seeking": 0.6,
                    },
                },
                {"role": "user", "content": "I am not sure."},
            ],
        },
    )

    context = actor.build_context_from_request(request)
    belief = context.state.data["StudentBelief"]
    tutor_state = context.state.data[GuiChemLlmTutor.__name__]

    assert belief.confidence == 0.2
    assert belief.confusion == 0.9
    assert belief.curiosity == 0.8
    assert "confidence appears limited" in tutor_state["student_belief"]
    assert "confusion is likely" in tutor_state["student_belief"]
    assert "strong curiosity" in tutor_state["student_belief"]


def test_gui_chem_tutor_keeps_only_ten_recent_messages():
    actor = GuiChemTutorActor(client=FakeClient())
    messages = [{"role": "user", "content": f"turn {index}"} for index in range(13)]
    request = RunRequest(
        message={"role": "user", "content": "turn 12"},
        info={"session_context": {"on_air": False}, "messages": messages},
    )

    context = actor.build_context_from_request(request)

    assert [message["content"] for message in context.trace.messages] == [
        "turn 3",
        "turn 4",
        "turn 5",
        "turn 6",
        "turn 7",
        "turn 8",
        "turn 9",
        "turn 10",
        "turn 11",
        "turn 12",
    ]


def test_gui_chem_tutor_history_keeps_applet_state_and_student_text():
    actor = GuiChemTutorActor(client=FakeClient())
    applet_input = {
        "applet": "applet-build-an-atom",
        "infoStore": {
            "protonCount": 1,
            "neutronCount": 0,
            "innerElectronCount": 1,
            "outerElectronCount": 0,
        },
    }
    request = RunRequest(
        message={"role": "user", "content": "I have now a neutral atom"},
        info={
            "session_context": {"on_air": False},
            "messages": [
                {
                    "role": "user",
                    "content": "I have now a neutral atom",
                    "kind": "applet",
                    "applet_input": applet_input,
                },
            ],
        },
    )

    context = actor.build_context_from_request(request)
    tutor_state = context.state.data[GuiChemLlmTutor.__name__]

    assert context.trace.messages[-1]["content"].endswith("Student said: I have now a neutral atom")
    assert '"innerElectronCount": 1' in tutor_state["applet_state"]


def test_gui_chem_tutor_prompt_contains_active_atomic_structure_context():
    actor = GuiChemTutorActor(client=FakeClient())
    request = RunRequest(
        message={
            "role": "user",
            "content": "What is an atom?",
        },
        info={
            "session_context": {
                "on_air": False,
                "subject": "chemistry",
                "subject_label": "Chemistry",
                "domain": "atomic-structure",
                "domain_label": "Atomic Structure",
                "domain_description": "Students learn how atoms are built from subatomic particles.",
                "domain_targets": [
                    {
                        "id": "neutron-identity",
                        "text": "identify the neutron count.",
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
    assert context.state.data["SessionContext"].domain_targets == [
        {
            "id": "neutron-identity",
            "text": "identify the neutron count.",
        },
    ]
    assert "Active tutoring context: Chemistry / Atomic Structure" in system_prompt
    assert "- subject: Chemistry (chemistry)" in system_prompt
    assert "- title: Atomic Structure" in system_prompt
    assert "- id: applet-build-an-atom" in system_prompt


def test_gui_chem_tutor_begin_agent_interactive_follows_debug_env(monkeypatch):
    monkeypatch.delenv("AIDU_DEBUG", raising=False)
    actor = GuiChemTutorActor(client=FakeClient())
    assert actor.agents[0].interactive is False

    monkeypatch.setenv("AIDU_DEBUG", "1")
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
