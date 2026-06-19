#!/usr/bin/env python3

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from typing import Iterator

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal SSE monitor: print time, event, and data.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8100/events",
        help="SSE endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP read timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Seconds to wait before reconnecting on disconnect (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed connection errors in status messages",
    )
    return parser.parse_args()


def iter_sse_events(response: requests.Response) -> Iterator[tuple[str, str]]:
    current_event = "message"
    data_lines: list[str] = []

    def flush() -> tuple[str, str] | None:
        nonlocal current_event, data_lines
        if not data_lines:
            return None
        payload_text = "\n".join(data_lines)
        event_name = current_event or "message"
        current_event = "message"
        data_lines = []
        return event_name, payload_text

    # Use tiny chunks to avoid buffered delays in streamed SSE output.
    for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
        if raw is None or raw == "":
            event = flush()
            if event is not None:
                yield event
            continue

        line = raw
        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            current_event = line[len("event:") :].strip() or "message"
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
            continue

    event = flush()
    if event is not None:
        yield event


def print_header(
    console: Console,
    *,
    url: str,
    status: str,
    retry_count: int,
    event_count: int,
    right_note: str = "",
    detail: str = "",
) -> None:
    status_color = "green" if status == "connected" else "yellow"
    first_line = Text.from_markup(
        f"[bold]SSE Monitor[/bold]  [dim]|[/dim]  "
        f"url: [cyan]{url}[/cyan]  [dim]|[/dim]  "
        f"status: [{status_color}]{status}[/{status_color}]  [dim]|[/dim]  "
        f"retries: [bold]{retry_count}[/bold]  [dim]|[/dim]  "
        f"events: [bold]{event_count}[/bold]"
    )

    header_grid = Table.grid(expand=True)
    header_grid.add_column(ratio=1)
    header_grid.add_column(justify="right", no_wrap=True)
    side_note = right_note or detail
    header_grid.add_row(first_line, Text(side_note))

    console.print(Panel(header_grid, border_style="blue", padding=(0, 1)))


def main() -> int:
    args = parse_args()
    console = Console()
    should_stop = False
    last_status: str | None = None
    disconnected_at: datetime | None = None
    retry_count = 0
    event_count = 0

    def _stop(_sig, _frame):
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while not should_stop:
        try:
            with requests.get(args.url, stream=True, timeout=(5.0, args.timeout)) as response:
                response.raise_for_status()

                if last_status != "connected":
                    if disconnected_at is not None and retry_count > 0:
                        down_for = int((datetime.now() - disconnected_at).total_seconds())
                        print_header(
                            console,
                            url=args.url,
                            status="connected",
                            retry_count=retry_count,
                            event_count=event_count,
                            right_note=f"recovered after {down_for}s",
                        )
                    else:
                        print_header(
                            console,
                            url=args.url,
                            status="connected",
                            retry_count=retry_count,
                            event_count=event_count,
                        )
                last_status = "connected"
                disconnected_at = None
                retry_count = 0

                for event_name, data in iter_sse_events(response):
                    event_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    console.print(f"{timestamp}\t{event_name}\t{data}")
                    if should_stop:
                        break

        except requests.RequestException as exc:
            retry_count += 1
            if last_status != "disconnected":
                disconnected_at = datetime.now()
                if args.verbose:
                    detail = f"{exc} | retry every {max(0.1, args.retry_delay)}s"
                else:
                    detail = f"retry every {max(0.1, args.retry_delay)}s"
                print_header(
                    console,
                    url=args.url,
                    status="disconnected",
                    retry_count=retry_count,
                    event_count=event_count,
                    detail=detail,
                )
                last_status = "disconnected"
            if should_stop:
                break
            time.sleep(max(0.1, args.retry_delay))

    return 0


if __name__ == "__main__":
    sys.exit(main())
