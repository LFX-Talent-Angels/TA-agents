"""A keyboard-driven picker: type to filter, arrows to move, Enter to choose.

Printing a numbered list and asking for the number means reading 22 lines,
finding the one you want, remembering an integer, and typing a second command.
Typing part of the name is how anyone who already knows the model would ask
for it.

Matching is on any part of the name, not the start. Model ids are
`vendor/model:tag`, so an author who remembers "laguna" should not have to
recall that it lives under "poolside". Characters must appear in order, so
"nemultra" finds "nemotron-3-ultra" without an exact substring.

cbreak mode is entered only when stdin and stdout are both a terminal. Piped
input and captured output fall back to the numbered list, which is what the
tests and any non-interactive caller get.

Every row is printed without wrapping. The frame is erased by moving the
cursor up as many lines as were written, so one wrapped row would leave the
count short and the previous frame would smear down the screen.
"""

from __future__ import annotations

import os
import select as select_module
import sys
import termios
import tty
from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

VISIBLE_ROWS = 8

_ENTER = {"\r", "\n"}
_BACKSPACE = {"\x7f", "\b"}
_CANCEL = {"\x1b", "\x03"}  # Esc, Ctrl-C
_UP = "\x1b[A"
_DOWN = "\x1b[B"
_ESCAPE_WAIT_SECONDS = 0.2


@dataclass(frozen=True)
class Option:
    """One selectable row. `key` is what gets matched, `label` what is shown."""

    key: str
    label: str


def subsequence_match(needle: str, haystack: str) -> bool:
    """True when every character of `needle` appears in order in `haystack`.

    Looser than a substring so a half-remembered name still lands, but ordered
    so it does not degenerate into matching everything.
    """
    if not needle:
        return True
    position = 0
    for character in needle.casefold():
        position = haystack.casefold().find(character, position)
        if position == -1:
            return False
        position += 1
    return True


def filter_options(options: Sequence[Option], query: str) -> list[Option]:
    """Exact substrings first — they are almost always what was meant."""
    lowered = query.casefold().strip()
    if not lowered:
        return list(options)
    substring = [o for o in options if lowered in o.key.casefold()]
    fuzzy = [o for o in options if o not in substring and subsequence_match(lowered, o.key)]
    return substring + fuzzy


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_key() -> str:
    """One keypress, decoding the three-byte arrow sequences."""
    descriptor = sys.stdin.fileno()
    first = os.read(descriptor, 1)
    if first != b"\x1b":
        return first.decode(errors="replace")
    # Esc alone (cancel) versus Esc [ A (an arrow): wait briefly for the rest
    # of an escape sequence, but never block a bare Escape. Read through the
    # descriptor rather than TextIOWrapper: its internal buffer can hide `[A`
    # and `[B` from select after the leading Escape was already consumed.
    ready, _, _ = select_module.select([descriptor], [], [], _ESCAPE_WAIT_SECONDS)
    if not ready:
        return "\x1b"
    second = os.read(descriptor, 1)
    if second != b"[":
        return "\x1b"
    ready, _, _ = select_module.select([descriptor], [], [], _ESCAPE_WAIT_SECONDS)
    if not ready:
        return "\x1b"
    return "\x1b[" + os.read(descriptor, 1).decode(errors="replace")


def _render(console: Console, options: list[Option], query: str, cursor: int, title: str) -> int:
    """Draw the frame. Returns how many lines were written, so it can be erased."""
    console.print(Text(title, style="bold"), no_wrap=True, overflow="ellipsis")
    console.print(Text(f"› {query}", style="cyan"), no_wrap=True, overflow="ellipsis")
    written = 2

    if not options:
        console.print(Text("  no match", style="dim"))
        return written + 1

    start = max(0, min(cursor - VISIBLE_ROWS // 2, len(options) - VISIBLE_ROWS))
    start = max(start, 0)
    for index, option in enumerate(options[start : start + VISIBLE_ROWS], start=start):
        selected = index == cursor
        console.print(
            Text(
                f"{'❯' if selected else ' '} {option.label}",
                style="bold cyan" if selected else "default",
            ),
            no_wrap=True,
            overflow="ellipsis",
        )
        written += 1

    hidden = len(options) - min(len(options), start + VISIBLE_ROWS)
    if hidden > 0:
        console.print(Text(f"  … {hidden} more", style="dim"))
        written += 1
    return written


def select(
    console: Console,
    options: Sequence[Option],
    *,
    title: str = "Type to filter · ↑↓ to move · Enter to pick · Esc to cancel",
) -> Option | None:
    """Run the picker. Returns the chosen option, or None if cancelled."""
    if not options:
        return None

    query = ""
    cursor = 0
    visible = list(options)
    settings = termios.tcgetattr(sys.stdin)
    lines = 0
    try:
        # cbreak, not raw: raw also disables output post-processing, so a
        # newline stops returning the carriage and every row starts where the
        # previous one ended, drawing the list diagonally down the screen.
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if lines:
                # Walk back over the previous frame and clear it, so the picker
                # updates in place instead of scrolling the scrollback away.
                sys.stdout.write(f"\x1b[{lines}A\x1b[J")
                sys.stdout.flush()
            lines = _render(console, visible, query, cursor, title)

            try:
                key = _read_key()
            except KeyboardInterrupt:
                # cbreak leaves ISIG on, so Ctrl-C arrives as a signal here
                # rather than as a byte to compare against.
                return None
            if key in _CANCEL:
                return None
            if key in _ENTER:
                return visible[cursor] if visible else None
            if key == _UP:
                cursor = max(0, cursor - 1)
                continue
            if key == _DOWN:
                cursor = min(len(visible) - 1, cursor + 1) if visible else 0
                continue
            if key in _BACKSPACE:
                query = query[:-1]
            elif key.isprintable():
                query += key
            else:
                continue
            visible = filter_options(options, query)
            cursor = 0
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        if lines:
            sys.stdout.write(f"\x1b[{lines}A\x1b[J")
        sys.stdout.write("\r")
        sys.stdout.flush()
