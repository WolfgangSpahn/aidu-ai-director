from aidu.ai.director.director import Director


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"role": "assistant", "content": "ok"}


def test_director_call_sends_nested_message_and_info(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("aidu.ai.director.director.requests.post", fake_post)

    director = Director()
    director.actors["chem_tutor_actor"] = {
        "service": True,
        "url": "http://actor.test",
    }

    response = director.call(
        "chem_tutor_actor",
        {
            "role": "user",
            "content": "Applet event: applet-periodic-table",
            "actor": "gui_user_actor",
            "kind": "applet",
            "applet_input": {
                "applet": "applet-periodic-table",
                "infoStore": {"elementName": "Lithium"},
            },
            "messages": [{"role": "user", "content": "previous"}],
            "session_id": "session-1",
            "session_context": {"domain": "atomic-structure"},
        },
    )

    assert response == {"role": "assistant", "content": "ok"}
    assert captured["url"] == "http://actor.test/run"
    assert captured["json"] == {
        "message": {
            "role": "user",
            "content": "Applet event: applet-periodic-table",
            "actor": "gui_user_actor",
            "kind": "applet",
            "applet_input": {
                "applet": "applet-periodic-table",
                "infoStore": {"elementName": "Lithium"},
            },
        },
        "info": {
            "summary": "",
            "messages": [{"role": "user", "content": "previous"}],
            "session_id": "session-1",
            "session_context": {"domain": "atomic-structure"},
        },
    }
