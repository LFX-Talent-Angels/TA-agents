"""Show that a turn is being worked on, how long it has taken, and let it go.

A turn can spend several seconds inside a provider call. Without a visible
indicator the terminal is indistinguishable from a crash, and a stalled
provider is indistinguishable from a slow one — the process even sits at 0%
CPU while it waits on the socket.

A spinner answers "is it alive". The elapsed counter answers "should I still
be waiting": 3s reads as working, 40s reads as stuck. Escape answers "can I
stop", which until now had no answer other than killing the session and losing
the conversation.

On cancel the work is abandoned, not killed. A Python thread cannot be
interrupted from outside, so the provider call runs to its own conclusion and
its result is discarded. The thread is a daemon, so it can never hold the
process open, and the user gets the prompt back immediately — which is the
part they actually wanted.
"""

from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from typing import TypeVar

from rich.console import Console

T = TypeVar("T")

# Below this a turn reads as instant and a counter is only noise.
SHOW_ELAPSED_AFTER_SECONDS = 2.0
# Escape is offered a little later still: a turn that ends quickly should not
# invite the reader to interrupt it.
OFFER_CANCEL_AFTER_SECONDS = 3.0
_POLL_SECONDS = 0.1
_CANCEL_KEYS = {"\x1b", "\x03"}  # Esc, Ctrl-C


class Cancelled(Exception):
    """The reader stopped waiting for this turn."""


def _can_watch_keys() -> bool:
    return sys.stdin.isatty()


def _status_text(label: str, elapsed: float) -> str:
    if elapsed < SHOW_ELAPSED_AFTER_SECONDS:
        return f"[cyan]{label}[/]"
    if elapsed < OFFER_CANCEL_AFTER_SECONDS:
        return f"[cyan]{label}[/] [dim]{elapsed:.0f}s[/]"
    return f"[cyan]{label}[/] [dim]{elapsed:.0f}s · esc to cancel[/]"


def run_with_status(console: Console, label: str, work: Callable[[], T]) -> T:
    """Run `work` behind a spinner. Raises `Cancelled` if the reader escapes.

    `work` runs on a background thread so this one stays free to watch the
    keyboard; without that the terminal cannot notice a keypress at all while
    a provider call blocks.
    """
    started = time.monotonic()
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["value"] = work()
        except BaseException as exc:  # re-raised on this thread below
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    watching = _can_watch_keys()
    settings = termios.tcgetattr(sys.stdin) if watching else None
    try:
        if watching:
            tty.setcbreak(sys.stdin.fileno())
        with console.status(_status_text(label, 0.0), spinner="dots") as status:
            while worker.is_alive():
                elapsed = time.monotonic() - started
                status.update(_status_text(label, elapsed))
                if not watching:
                    worker.join(_POLL_SECONDS)
                    continue
                ready, _, _ = select.select([sys.stdin], [], [], _POLL_SECONDS)
                if ready and sys.stdin.read(1) in _CANCEL_KEYS:
                    raise Cancelled
    finally:
        if settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]
