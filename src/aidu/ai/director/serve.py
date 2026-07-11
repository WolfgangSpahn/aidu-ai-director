# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""Frontend bridge for AIDu Director workflows.

The Director coordinates a small cast of actors, usually instances from the
``aidu-ai-actor`` package. This module provides the user-facing edge of that
system: it serves the browser frontend, streams dialog turns back to connected
clients, accepts user input, and exposes a ``/run`` endpoint so the browser user
can participate in a workflow like any other actor.

In the overall architecture, the Director remains responsible for orchestration
and routing. ``Server`` is the adapter between that orchestration layer and
human-facing frontends: browser messages are converted into actor-style
messages, and actor prompts routed to the user are held until the frontend sends
the next reply.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aidu.ai.director.config import DEFAULT_NAMING, WEB_CONFIG

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------


class UserInput(BaseModel):
    role: str = "user"
    content: str


class RunRequest(BaseModel):
    summary: str = ""
    messages: list[dict[str, Any]] = []
    actor: str = ""
    role: str = "assistant"
    content: str


# ------------------------------------------------------------------
# Server
# ------------------------------------------------------------------


class Server:
    def __init__(
        self,
        name: str,
        director_url: str = "",
        description: str = "",
        director=None,
        start_actor: str | None = None,
        max_step: int = 5,
        web_dir: str | Path | None = None,
        naming: dict[str, str] | None = None,
    ):
        """
        Serves the web frontend and lets a browser user participate as an actor.

        Browser
            ↓
        Director
            ↓
        Actors
        """

        self.name = name
        self.description = description
        self.director_url = director_url.rstrip("/")
        self.director = director
        self.start_actor = start_actor or name
        self.max_step = max_step
        self.naming = dict(DEFAULT_NAMING if naming is None else naming)
        self.turns: list[dict[str, str]] = []
        self.subscribers: set[queue.Queue[dict[str, str]]] = set()
        self.pending_inputs: queue.Queue[dict[str, str]] = queue.Queue()
        self._pending_input_requests = 0
        self._pending_input_lock = threading.Lock()
        self._director_event_queue: queue.Queue | None = None

        self.web_dir = Path(web_dir) if web_dir is not None else self._find_web_dir()

        # check if the web_dir exists
        if not self.web_dir.exists():
            raise FileNotFoundError(f"Web directory '{self.web_dir}' does not exist. Please ensure the frontend assets are built and available.")

        self.app = FastAPI(
            title=name,
            description=description,
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._register_routes()
        self._start_director_event_bridge()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @staticmethod
    def _find_web_dir() -> Path:
        package_dir = Path(__file__).parent
        candidates = [
            WEB_CONFIG.web_dir,
            package_dir / "web" / "dist",
            package_dir / "demo",
            package_dir.parents[3] / "demo" / "dist",
            package_dir.parents[3] / "demo",
        ]

        for candidate in candidates:
            if (candidate / "index.html").exists():
                return candidate

        return candidates[0]

    def _start_director_event_bridge(self):
        if self.director is None or not hasattr(self.director, "_add_subscriber"):
            return

        self._director_event_queue = queue.Queue(maxsize=200)
        self.director._add_subscriber(self._director_event_queue)

        thread = threading.Thread(
            target=self._forward_director_events,
            daemon=True,
        )
        thread.start()

    def _forward_director_events(self):
        if self._director_event_queue is None:
            return

        while True:
            event = self._director_event_queue.get()
            if event.get("event") != "message":
                continue

            data = event.get("data", {})
            message = data.get("message", data)
            turn = self._turn_from_message(
                message,
                actor=data.get("actor") or data.get("avatar"),
            )
            self._publish_turn(turn)

    def _turn_from_message(self, message: dict[str, Any], actor: str | None = None) -> dict[str, str]:
        actor_name = str(actor or message.get("actor") or message.get("avatar") or self.name)
        turn = {
            "role": str(message.get("role") or "message"),
            "content": str(message.get("content") or ""),
        }
        turn["actor"] = actor_name
        turn["avatar"] = self.naming.get(actor_name, actor_name)
        return turn

    def _publish_turn(self, turn: dict[str, str]):
        self.turns.append(turn)
        for subscriber in list(self.subscribers):
            try:
                subscriber.put_nowait(turn)
            except queue.Full:
                self.subscribers.discard(subscriber)

    def _submit_to_director(self, message: dict[str, str]):
        if self.director is None:
            turn = {
                "role": "system",
                "content": "Input received, but no director is attached to this server.",
                "actor": self.name,
                "avatar": self.name,
            }
            self._publish_turn(turn)
            return

        try:
            next_actor = self.director.routes.get(self.start_actor)
            if next_actor is None:
                self._publish_turn(
                    {
                        "role": "system",
                        "content": f"Input received, but no route is defined for {self.start_actor}.",
                        "actor": self.name,
                        "avatar": self.name,
                    }
                )
                return

            self.director.run(
                start_actor=next_actor,
                message=message,
                max_step=self.max_step,
            )
        except Exception as exc:
            logger.exception("Director run failed")
            self._publish_turn(
                {
                    "role": "system",
                    "content": f"Director run failed: {exc}",
                    "actor": self.name,
                    "avatar": self.name,
                }
            )

    def _register_routes(self):

        # --------------------------------------------------
        # Static frontend
        # --------------------------------------------------

        if (self.web_dir / "assets").exists():
            self.app.mount(
                "/assets",
                StaticFiles(directory=self.web_dir / "assets"),
                name="assets",
            )

        @self.app.get("/")
        def index():
            return FileResponse(
                self.web_dir / "index.html",
                headers={"Cache-Control": "no-store"},
            )

        @self.app.get("/app.js")
        def app_js():
            return FileResponse(
                self.web_dir / "app.js",
                media_type="application/javascript",
                headers={"Cache-Control": "no-store"},
            )

        # --------------------------------------------------
        # Health
        # --------------------------------------------------

        @self.app.get("/health")
        def health():
            return {
                "name": self.name,
                "status": "running",
            }

        @self.app.get("/info")
        def info():
            return {
                "name": self.name,
                "description": self.description,
                "director": self.director_url,
                "start_actor": self.start_actor,
                "forwarding": self.director is not None,
            }

        @self.app.get("/naming")
        def naming():
            return self.naming

        # --------------------------------------------------
        # Browser sends user input
        # --------------------------------------------------

        @self.app.post("/input")
        def input(msg: UserInput):
            logger.info(
                "Received frontend input: role=%s content=%r",
                msg.role,
                msg.content,
            )
            turn = self._turn_from_message(
                {
                    "role": msg.role,
                    "content": msg.content,
                },
                actor=self.start_actor,
            )
            self._publish_turn(turn)

            message = {
                "role": msg.role,
                "content": msg.content,
            }

            with self._pending_input_lock:
                has_pending_actor_request = self._pending_input_requests > 0

            if has_pending_actor_request:
                self.pending_inputs.put(message)
                return {
                    "status": "accepted",
                    "mode": "pending_actor_response",
                    "turn": turn,
                }

            threading.Thread(
                target=self._submit_to_director,
                args=(message,),
                daemon=True,
            ).start()

            return {
                "status": "accepted",
                "mode": "director_run_started",
                "turn": turn,
            }

        @self.app.post("/run")
        def run(req: RunRequest):
            incoming = self._turn_from_message(
                {
                    "role": req.role,
                    "content": req.content,
                },
                actor=req.actor or self.name,
            )
            self._publish_turn(incoming)

            with self._pending_input_lock:
                self._pending_input_requests += 1

            try:
                response = self.pending_inputs.get()
            finally:
                with self._pending_input_lock:
                    self._pending_input_requests -= 1

            return {
                "role": response["role"],
                "content": response["content"],
            }

        @self.app.get("/events")
        def events():
            subscriber: queue.Queue[dict[str, str]] = queue.Queue(maxsize=100)
            self.subscribers.add(subscriber)

            def stream():
                try:
                    for turn in self.turns:
                        yield f"data: {json.dumps(turn)}\n\n"

                    while True:
                        try:
                            turn = subscriber.get(timeout=15)
                            yield f"data: {json.dumps(turn)}\n\n"
                        except queue.Empty:
                            yield ": keepalive\n\n"
                finally:
                    self.subscribers.discard(subscriber)

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # --------------------------------------------------
        # SPA fallback
        # --------------------------------------------------

        @self.app.get("/{path:path}")
        def spa(path: str):
            return FileResponse(
                self.web_dir / "index.html",
                headers={"Cache-Control": "no-store"},
            )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8100,
        reload: bool = False,
    ):

        import uvicorn

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            access_log=False,
            log_config=None,
        )

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8100,
    ):

        thread = threading.Thread(
            target=self.serve,
            kwargs={
                "host": host,
                "port": port,
                "reload": False,
            },
            daemon=True,
        )

        thread.start()

        return thread
