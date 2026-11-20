"""Typed input constraints for the passive Connect skill."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectRequest:
    """The subject and graph constraint required by one Connect operation."""

    subject: str
    rel_types: tuple[str, ...] = ()
    relation_kind: str | None = None
