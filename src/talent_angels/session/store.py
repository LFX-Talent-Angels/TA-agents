"""File-backed session save/load under ``<memory home>/sessions`` (or TA_SESSIONS_DIR)."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.memory.paths import default_sessions_dir
from talent_angels.session.models import (
    AreaChoice,
    LastBinding,
    PendingChoice,
    SessionState,
    TranscriptLine,
)

_LAST_POINTER = "_last"
_META = "meta.json"
_TRANSCRIPT = "transcript.jsonl"
_BINDING = "binding.json"


def sessions_dir() -> Path:
    """``TA_SESSIONS_DIR`` or ``<memory home>/sessions`` — never cwd-relative."""
    raw = os.environ.get("TA_SESSIONS_DIR", "").strip()
    return Path(raw).expanduser() if raw else default_sessions_dir()


_SESSION_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_session_key(key: str) -> str:
    """A session key is a directory name: no separators, no dots, bounded."""
    if not _SESSION_KEY.fullmatch(key):
        raise ValueError(f"invalid session id {key!r}")
    return key


def session_exists(key: str) -> bool:
    try:
        validate_session_key(key)
    except ValueError:
        return False
    return (sessions_dir() / key / _META).is_file()


def new_session() -> SessionState:
    return SessionState(session_id=uuid4().hex[:12])


def save_session(state: SessionState, *, name: str | None = None, update_last: bool = True) -> Path:
    """Persist a session. ``update_last=False`` for API sessions, so a web
    client's conversation never becomes the one the TUI resumes."""
    if name is not None:
        state.name = name
    key = validate_session_key(state.name or state.session_id)
    root = sessions_dir()
    path = root / key
    path.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": state.session_id,
        "name": state.name,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    (path / _META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    with (path / _TRANSCRIPT).open("w", encoding="utf-8") as fh:
        for line in state.transcript:
            fh.write(line.model_dump_json() + "\n")

    binding_payload = {
        "node": state.binding.node.model_dump() if state.binding is not None else None,
        "pending": [choice.model_dump() for choice in state.pending],
        "last_result": state.last_result.model_dump() if state.last_result is not None else None,
        "bindings": {suite: node.model_dump() for suite, node in state.bindings.items()},
        "last_results": [r.model_dump() for r in state.last_results],
        "recent": list(state.recent),
        "pending_profile_intent": state.pending_profile_intent,
        "areas": [area.model_dump() for area in state.areas],
        "list_topic": state.list_topic,
        "pending_compare": list(state.pending_compare),
        "language": state.language,
    }
    (path / _BINDING).write_text(json.dumps(binding_payload, indent=2) + "\n", encoding="utf-8")

    if update_last:
        (root / _LAST_POINTER).write_text(key + "\n", encoding="utf-8")
    return path


def load_session(name: str) -> SessionState:
    path = sessions_dir() / validate_session_key(name)
    meta = json.loads((path / _META).read_text(encoding="utf-8"))

    transcript: list[TranscriptLine] = []
    transcript_path = path / _TRANSCRIPT
    if transcript_path.is_file():
        for raw in transcript_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                transcript.append(TranscriptLine.model_validate_json(raw))

    binding: LastBinding | None = None
    pending: list[PendingChoice] = []
    last_result = None
    bindings: dict[str, NodeRef] = {}
    last_results: list[AgentResult] = []
    recent: list[str] = []
    pending_profile_intent: str | None = None
    areas: list[AreaChoice] = []
    list_topic = ""
    pending_compare: list[str] = []
    language = "en"
    binding_path = path / _BINDING
    if binding_path.is_file():
        payload = json.loads(binding_path.read_text(encoding="utf-8"))
        node = payload.get("node")
        if node is not None:
            binding = LastBinding.model_validate({"node": node})
        pending = [PendingChoice.model_validate(item) for item in payload.get("pending") or []]
        raw_result = payload.get("last_result")
        if raw_result is not None:
            last_result = AgentResult.model_validate(raw_result)
        for suite, raw_node in (payload.get("bindings") or {}).items():
            bindings[suite] = NodeRef.model_validate(raw_node)
        for raw_r in payload.get("last_results") or []:
            last_results.append(AgentResult.model_validate(raw_r))
        recent = [str(title) for title in payload.get("recent") or []]
        pending_profile_intent = payload.get("pending_profile_intent")
        areas = [AreaChoice.model_validate(item) for item in payload.get("areas") or []]
        list_topic = str(payload.get("list_topic") or "")
        pending_compare = [str(item) for item in payload.get("pending_compare") or []]
        language = str(payload.get("language") or "en")

    return SessionState(
        session_id=meta["session_id"],
        name=meta.get("name"),
        transcript=transcript,
        binding=binding,
        pending=pending,
        last_result=last_result,
        bindings=bindings,
        last_results=last_results,
        recent=recent,
        pending_profile_intent=pending_profile_intent,
        areas=areas,
        list_topic=list_topic,
        pending_compare=pending_compare,
        language=language,
    )


def load_last() -> SessionState:
    pointer = sessions_dir() / _LAST_POINTER
    name = pointer.read_text(encoding="utf-8").strip()
    return load_session(name)


def clear_conversation(state: SessionState) -> SessionState:
    return SessionState(
        session_id=state.session_id,
        name=state.name,
        transcript=[],
        binding=None,
        pending=[],
        last_result=None,
    )
