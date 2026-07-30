from aidu.ai.director.director import Director
from aidu.ai.actor.types import RunRequest
from aidu.ai.core.context import Message
from aidu.ai.core.session import SessionInfo


class FakeResponse:
    def raise_for_status(self):
        return None

    def iter_lines(self, chunk_size=None, decode_unicode=False):
        assert chunk_size == 1
        assert decode_unicode is True
        return iter(
            [
                '{"type":"delta","content":"o"}',
                '{"type":"delta","content":"k"}',
                '{"type":"final","response":{"role":"assistant","content":"ok"}}',
            ]
        )


def test_director_call_sends_nested_message_and_info(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout, stream):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        captured["stream"] = stream
        return FakeResponse()

    monkeypatch.setattr("aidu.ai.director.director.requests.post", fake_post)

    director = Director()
    director.actors["chem_tutor_actor"] = {
        "service": True,
        "url": "http://actor.test",
    }
    events = Queue()
    director._add_subscriber(events)

    response = director.call(
        "chem_tutor_actor",
        Message(
            role="user",
            content="Applet event: applet-periodic-table",
            actor="gui_user_actor",
            kind="applet",
        ),
        SessionInfo(
            applet_input={
                "applet": "applet-periodic-table",
                "infoStore": {"elementName": "Lithium"},
            },
            messages=[{"role": "user", "content": "previous"}],
            session_id="session-1",
            session_context={"on_air": True, "domain": "atomic-structure"},
        ),
    )

    assert response == {"role": "assistant", "content": "ok"}
    first_delta = events.get_nowait()
    second_delta = events.get_nowait()
    assert first_delta["event"] == "message_delta"
    assert first_delta["data"]["session_id"] == "session-1"
    assert first_delta["data"]["content"] == "o"
    assert second_delta["data"]["content"] == "k"
    assert captured["url"] == "http://actor.test/run/stream"
    assert captured["stream"] is True
    request = RunRequest.model_validate(captured["json"])
    assert request.message.model_dump(exclude_none=True) == {
        "role": "user",
        "content": "Applet event: applet-periodic-table",
        "actor": "gui_user_actor",
        "kind": "applet",
    }
    assert request.info.session_id == "session-1"
    assert request.info.session_context.on_air is True
    assert request.info.session_context.domain == "atomic-structure"
    assert request.info.messages.model_dump() == [{"role": "user", "content": "previous"}]
    assert request.info.applet_input == {
        "applet": "applet-periodic-table",
        "infoStore": {"elementName": "Lithium"},
    }


def test_director_call_without_session_sends_valid_off_air_context(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout, stream):
        captured["request"] = RunRequest.model_validate(json)
        return FakeResponse()

    monkeypatch.setattr("aidu.ai.director.director.requests.post", fake_post)

    director = Director()
    director.actors["test_knowledge_actor"] = {
        "service": True,
        "url": "http://actor.test",
    }

    director.call(
        "test_knowledge_actor",
        Message(role="user", content='{"questions": []}'),
    )

    assert captured["request"].info.session_id is None
    assert captured["request"].info.session_context.on_air is False


def test_director_run_preserves_session_id_without_copying_input_metadata(monkeypatch):
    director = Director()
    director.actors["chem_tutor_actor"] = {
        "actor": object(),
        "service": True,
        "url": "http://actor.test",
        "avatar": "Robo",
    }
    director.actors["gui_user_actor"] = {
        "actor": object(),
        "service": False,
        "url": "",
        "avatar": "Buddy",
    }
    director.routes["chem_tutor_actor"] = "gui_user_actor"

    def fake_call(actor, message, info=None):
        return {"role": "assistant", "content": "rule response"}

    monkeypatch.setattr(director, "call", fake_call)

    trace = director.run(
        start_actor="chem_tutor_actor",
        message=Message(role="user", content="Applet event", kind="applet"),
        info=SessionInfo(
            session_id="session-1",
            session_context={"on_air": True},
            applet_input={"applet": "applet-build-an-atom"},
        ),
    )

    next_message = trace[-1][1]
    assert next_message.content == "rule response"
    assert next_message.source_actor == "chem_tutor_actor"
    assert next_message.recipient_actor == "gui_user_actor"
    assert next_message.session_id == "session-1"
    assert not hasattr(next_message, "kind")
    assert not hasattr(next_message, "applet_input")


from queue import Queue
