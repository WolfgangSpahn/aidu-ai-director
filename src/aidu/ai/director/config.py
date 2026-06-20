from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SSEConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8100
    path: str = "/events"


SSE_CONFIG = SSEConfig()


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    web_dir: Path = Path(__file__).parent / "web" / "dist"


WEB_CONFIG = WebConfig()


DEFAULT_NAMING = {
    "tui_user_actor": "Daisy",
    "echo_actor": "Bruce",
    "math_tutor_actor": "Bruno",
    "math_student_actor": "Buddy",
    "gui_user_actor": "Bella",
    "dummy_director_server": "System",
}
