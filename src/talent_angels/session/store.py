"""File-backed session save/load under data/local/sessions (or TA_SESSIONS_DIR)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from talent_angels.contracts import AgentResult
from talent_angels.session.models import (
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
    raw = os.environ.get("TA_SESSIONS_DIR", "data/local/sessions")
    return Path(raw)


def new_session() -> SessionState:
    return SessionState(session_id=uuid4().hex[:12])


def save_session(state: SessionState, *, name: str | None = None) -> Path:
    if name is not None:
        state.name = name
    key = state.name or state.session_id
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
    }
    (path / _BINDING).write_text(json.dumps(binding_payload, indent=2) + "\n", encoding="utf-8")

    (root / _LAST_POINTER).write_text(key + "\n", encoding="utf-8")
    return path


def load_session(name: str) -> SessionState:
    path = sessions_dir() / name
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

    return SessionState(
        session_id=meta["session_id"],
        name=meta.get("name"),
        transcript=transcript,
        binding=binding,
        pending=pending,
        last_result=last_result,
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
