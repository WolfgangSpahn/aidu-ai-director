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
                logger.info(f"registered external actor {name}")
                continue

            logger.info(f"starting {name} on port {info['port']}")

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

                logger.debug(f"{name} ready")

                return

            except Exception:
                time.sleep(0.25)

        raise RuntimeError(f"{name} failed to start at {url}")

    # ------------------------------------------------------------------
    # REST call
    # ------------------------------------------------------------------

    def call(self, actor: str, message: dict) -> dict:

        if actor not in self.actors or not self.actors[actor]["service"]:
            raise RuntimeError(f"Actor '{actor}' is not a callable service actor")

        url = self.actors[actor]["url"]
        payload = {
            "summary": message.get("summary", ""),
            "messages": message.get("messages", [message]),
            "actor": message.get("actor", actor),
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
        }
        for key, value in message.items():
            payload.setdefault(key, value)

        response = requests.post(
            f"{url}/run",
            json=payload,
            timeout=300,
        )

        response.raise_for_status()

        return response.json()

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def run(self, start_actor: str, message: Message, max_step: int = 5, console=None, interactive: bool = False):

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
            logger.info(f"[director] step {step}, mailbox len: {len(mailbox)}: calling {actor_name}")

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

            response = self.call(actor=actor_name, message=message)

            next_actor = self.routes.get(actor_name)
            if next_actor is None:
                logger.debug(f"[director] no route defined for {actor_name}")
                break

            metadata = {
                key: value
                for key, value in message.items()
                if key not in ("role", "content", "source_actor", "recipient_actor")
            }
            next_message = Message(
                **metadata,
                role="assistant" if "user" not in actor_name else "user",
                content=response["content"],
                source_actor=actor_name,
                recipient_actor=next_actor,
            )
            if response.get("applet") and response.get("applet_command"):
                next_message["applet"] = response["applet"]
                next_message["applet_command"] = response["applet_command"]

            self._publish_message(next_actor, next_message)

            logger.debug(f"[director] route: {actor_name} -> {next_actor}")

            trace.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), next_message))
            if next_actor not in self.actors:
                continue

            mailbox.append((next_actor, next_message))

            # for debgugging,
            if console is not None:
                console.print(Rule(title=f"Step {step}"))
                console.print(trace)

            if interactive:
                input("Press Enter to continue...")

        return trace
