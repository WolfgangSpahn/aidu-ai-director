# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/director/director.py

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

from aidu.ai.core.session import RoutedMessage, SessionInfo
from rich.rule import Rule

import requests

from aidu.ai.core.context import Message

logger = logging.getLogger(__name__)


class RouteBuilder:
    def __init__(self, director, source: str):
        self.director = director
        self.source = source

    def send_to(self, target: str):
        self.director.routes[self.source] = target
        return self.director


class Director:
    def __init__(self):

        self.actors: dict[str, dict] = {}
        self.routes: dict[str, str] = {}
        self._event_subscribers: list[queue.Queue] = []
        self._event_lock = threading.Lock()
        self._sse_server: ThreadingHTTPServer | None = None
        self._sse_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # SSE server
    # ------------------------------------------------------------------

    def start_sse_server(self, host: str = "0.0.0.0", port: int = 8100, path: str = "/events"):

        if self._sse_server is not None:
            raise RuntimeError("SSE server is already running")

        director = self

        class SSEHandler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self):
                if self.path == "/health":
                    body = b'{"status":"ok"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path != path:
                    self.send_response(404)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                client_queue: queue.Queue = queue.Queue(maxsize=200)
                director._add_subscriber(client_queue)

                try:
                    self.wfile.write(b"event: connected\n")
                    self.wfile.write(b'data: {"status":"connected"}\n\n')
                    self.wfile.flush()

                    while True:
                        try:
                            event = client_queue.get(timeout=15)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue

                        event_name = event.get("event", "message")
                        payload = json.dumps(event.get("data", {}), default=str)

                        self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
                        for payload_line in payload.splitlines() or [payload]:
                            self.wfile.write(f"data: {payload_line}\n".encode("utf-8"))
                        self.wfile.write(b"\n")
                        self.wfile.flush()

                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    director._remove_subscriber(client_queue)

            def log_message(self, format: str, *args):
                logger.debug("[director-sse] " + format, *args)

        self._sse_server = ThreadingHTTPServer((host, port), SSEHandler)
        self._sse_thread = threading.Thread(target=self._sse_server.serve_forever, daemon=True)
        self._sse_thread.start()
        logger.info(f"[director] SSE server listening on http://{host}:{port}{path}")

    def stop_sse_server(self):

        if self._sse_server is None:
            return

        self._sse_server.shutdown()
        self._sse_server.server_close()
        self._sse_server = None

        if self._sse_thread is not None:
            self._sse_thread.join(timeout=2)
            self._sse_thread = None

    def _add_subscriber(self, subscriber: queue.Queue):

        with self._event_lock:
            self._event_subscribers.append(subscriber)

    def _remove_subscriber(self, subscriber: queue.Queue):

        with self._event_lock:
            self._event_subscribers = [q for q in self._event_subscribers if q is not subscriber]

    def _publish_event(self, event: str, data: dict[str, Any]):

        payload = {"event": event, "data": data}

        with self._event_lock:
            subscribers = list(self._event_subscribers)

        for subscriber in subscribers:
            try:
                subscriber.put_nowait(payload)
            except queue.Full:
                logger.debug("[director] dropping SSE event for a slow subscriber")

    def _publish_message(self, actor: str, message: Message):
        payload = self._serialize_message(actor=actor, message=message)
        self._publish_event(
            event="message",
            data={**payload, "message": payload},
        )

    def _serialize_message(self, actor: str, message: Message) -> dict[str, Any]:
        payload: dict[str, Any]
        if hasattr(message, "model_dump"):
            payload = message.model_dump()
        elif isinstance(message, dict):
            payload = dict(message)
        else:
            payload = {"content": str(message)}

        # Keep both actor and avatar for compatibility with existing consumers.
        payload["actor"] = actor
        payload["avatar"] = self.actor_avatar(actor)
        payload.setdefault("role", "assistant")
        payload.setdefault("content", "")
        source_actor = payload.get("source_actor")
        if source_actor:
            payload["source_avatar"] = self.actor_avatar(source_actor)
        return payload

    def get_actor(self, actor: str):
        return self.actors[actor]["actor"]

    def actor_avatar(self, actor: str) -> str:
        info = self.actors.get(actor)
        if info is None:
            return actor
        return info["avatar"]

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, actor, port: int | None = None):

        self.actors[actor.name] = {
            "actor": actor,
            "port": port,
            "url": f"http://localhost:{port}" if port is not None else "",
            "thread": None,
            "service": port is not None,
            "avatar": getattr(actor, "avatar", actor.name),
        }

    # ------------------------------------------------------------------
    # Routing DSL
    # ------------------------------------------------------------------

    def on_input(self, actor: str):

        return RouteBuilder(self, actor)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self):

        for name, info in self.actors.items():
            if not info["service"]:
                logger.info(f"registered '{name}' as non-service actor")
                continue

            logger.info(f"starting '{name}' on port {info['port']}'")

            thread = info["actor"].start(
                port=info["port"],
            )

            info["thread"] = thread

        # wait until all actors answer REST requests
        for name, info in self.actors.items():
            if not info["service"]:
                continue

            self._wait_until_ready(
                name=name,
                url=info["url"],
            )

    def _wait_until_ready(self, name: str, url: str, timeout: float = 10.0):

        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                requests.get(
                    f"{url}/docs",
                    timeout=1,
                )

                try:
                    info = requests.get(f"{url}/info", timeout=1).json()
                except Exception as exc:
                    info = {"error": str(exc)}

                logger.debug(
                    "[director] actor ready name=%s url=%s served_info=%s",
                    name,
                    url,
                    info,
                )

                return

            except Exception:
                time.sleep(0.25)

        raise RuntimeError(f"{name} failed to start at {url}")

    # ------------------------------------------------------------------
    # REST call
    # ------------------------------------------------------------------

    def call(self, actor: str, message: Message | dict[str, Any], info: SessionInfo | None = None) -> dict:

        if actor not in self.actors or not self.actors[actor]["service"]:
            raise RuntimeError(f"Actor '{actor}' is not a callable service actor")

        message = message if isinstance(message, Message) else Message.model_validate(message)

        url = self.actors[actor]["url"]
        run_info = (
            info.model_dump(exclude_none=True)
            if info is not None
            else {"summary": "", "messages": [], "session_id": None, "session_context": {}}
        )
        current_message = message.model_dump(exclude_none=True)
        current_message.setdefault("actor", message.actor or actor)
        current_message.setdefault("role", message.role or "user")
        current_message.setdefault("content", message.content or "")
        payload = {
            "message": current_message,
            "info": run_info,
        }

        logger.debug(
            "[director] call actor=%s url=%s startup_actor=%s domain=%s:%s applet=%s:%s messages=%s content_prefix=%r",
            actor,
            url,
            current_message.get("actor"),
            run_info["session_context"].get("domain"),
            run_info["session_context"].get("domain_label"),
            run_info["session_context"].get("applet_id"),
            run_info["session_context"].get("applet_name"),
            len(run_info.get("messages") or []),
            str(current_message.get("content", ""))[:240],
        )

        response = requests.post(
            f"{url}/run/stream",
            json=payload,
            timeout=300,
            stream=True,
        )
        response.raise_for_status()
        turn_id = str(uuid4())
        final_response: dict[str, Any] | None = None
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "delta":
                self._publish_event(
                    event="message_delta",
                    data={
                        "session_id": run_info.get("session_id"),
                        "turn_id": turn_id,
                        "source_actor": actor,
                        "recipient_actor": self.routes.get(actor),
                        "content": event.get("content", ""),
                    },
                )
            elif event.get("type") == "final":
                final_response = event.get("response") or {}
            elif event.get("type") == "error":
                raise RuntimeError(event.get("error") or "Streaming actor request failed")
        if final_response is None:
            raise RuntimeError(f"Actor '{actor}' stream ended without a final response")
        return final_response

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def run(
        self,
        start_actor: str,
        message: Message | dict[str, Any],
        info: SessionInfo | None = None,
        max_step: int = 5,
        console=None,
        interactive: bool = False,
    ):

        message = message if isinstance(message, Message) else Message.model_validate(message)

        trace = [(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)]
        self._publish_message(start_actor, message)

        mailbox = deque()
        mailbox.append((start_actor, message))

        step = 0
        while mailbox:
            step += 1

            if step > max_step:
                logger.debug("[director] maximum steps reached")
                self._publish_event(
                    event="run_stopped",
                    data={
                        "reason": "max_step_reached",
                        "step": step,
                    },
                )
                break

            actor_name, message = mailbox.popleft()
            logger.debug(f"[director] step {step}, mailbox len: {len(mailbox)}: calling {actor_name}")

            if actor_name not in self.actors:
                logger.debug(f"[director] actor {actor_name} is not registered")
                break

            if not self.actors[actor_name]["service"]:
                next_actor = self.routes.get(actor_name)
                if next_actor is None:
                    logger.debug(f"[director] external actor {actor_name} has no route")
                    break

                logger.debug(f"[director] route: {actor_name} -> {next_actor}")
                mailbox.append((next_actor, message))
                continue

            service_message = (
                message
                if isinstance(message, Message)
                else Message(
                    role=message.role,
                    content=message.content,
                    actor=message.source_actor,
                )
            )
            response = self.call(actor=actor_name, message=service_message, info=info)

            next_actor = self.routes.get(actor_name)
            if next_actor is None:
                logger.debug(f"[director] no route defined for {actor_name}")
                break

            next_message = RoutedMessage(
                role="assistant" if "user" not in actor_name else "user",
                content=response["content"],
                source_actor=actor_name,
                recipient_actor=next_actor,
                session_id=info.session_id if info else None,
            )
            if response.get("applet") and response.get("applet_command"):
                next_message.applet = response["applet"]
                next_message.applet_command = response["applet_command"]
            if response.get("activity_event"):
                next_message.activity_event = response["activity_event"]
                logger.info(
                    "[director] forwarding activity_event session=%s event=%s",
                    info.session_id if info else None,
                    response["activity_event"],
                )
            if response.get("backend_belief_state"):
                next_message.backend_belief_state = response["backend_belief_state"]
            if response.get("backend_progress_state"):
                next_message.backend_progress_state = response["backend_progress_state"]

            self._publish_message(next_actor, next_message)

            logger.debug(f"[director] route: {actor_name} -> {next_actor}")

            trace.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), next_message))
            if next_actor not in self.actors or not self.actors[next_actor]["service"]:
                continue

            mailbox.append((next_actor, next_message))

            # for debgugging,
            if console is not None:
                console.print(Rule(title=f"Step {step}"))
                console.print(trace)

            if interactive:
                input("Press Enter to continue...")

        return trace
