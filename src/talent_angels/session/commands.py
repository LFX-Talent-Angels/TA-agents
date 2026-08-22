"""Slash-command parsing for the interactive ta-agent TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandName = Literal["help", "quit", "save", "resume", "clear"]

_NAME_MAP: dict[str, CommandName] = {
    "help": "help",
    "quit": "quit",
    "exit": "quit",
    "save": "save",
    "resume": "resume",
    "clear": "clear",
}


class UnknownCommand(ValueError):
    """Raised when a slash line names a command that is not allowed."""


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    name: CommandName
    argument: str | None


def parse_command(text: str) -> ParsedCommand | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped.split(maxsplit=1)
    raw_name = parts[0][1:].lower()
    name = _NAME_MAP.get(raw_name)
    if name is None:
        raise UnknownCommand(parts[0])

    argument = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    if name == "resume" and argument is None:
        argument = "last"
    return ParsedCommand(name=name, argument=argument)
