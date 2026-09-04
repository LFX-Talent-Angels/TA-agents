"""Conversation kernel: classify a line, mutate session, call run_turn only for map-work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from talent_angels.assistant.answer import is_unimplemented_pathfind
from talent_angels.assistant.connect_request import followup_connect_request
from talent_angels.assistant.intent import CAPABILITY_CONNECT
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm import LLMClient
from talent_angels.session.budget import model_view
from talent_angels.session.catalog import FreeModel
from talent_angels.session.commands import UnknownCommand, parse_command
from talent_angels.session.copy import (
    ADVICE_REFUSE,
    CATALOGUE_REFUSE,
    COMMANDS_BLOCK,
    GREETING,
    HELP_INTRO,
    HELP_TEXT,
    LOCATE_MISS,
    MAP_NEXT_STEP,
    UNKNOWN_COMMAND,
)
from talent_angels.session.followup import (
    can_expand_connect,
    can_expand_locate,
    is_bare_yes,
    is_expand_list,
    parse_skill_mention,
    render_connect_list,
    skill_from_connect,
    skill_index_by_label,
)
from talent_angels.session.models import LastBinding, PendingChoice, SessionState, TranscriptLine
from talent_angels.session.phrase import (
    ambiguous_intro_card,
    connect_card,
    locate_card,
    phrase_chat,
    phrase_map,
    skill_card,
    uses_chat_phrasing,
)
from talent_angels.session.picker import bind_pick, choices_from_result, render_picker
from talent_angels.session.router import route_line
from talent_angels.session.store import (
    clear_conversation,
    load_last,
    load_session,
    save_session,
    sessions_dir,
)
from talent_angels.session.switch import SwitchError, apply, load_catalogue, resolve

_NO_PENDING = "There's no numbered list to pick from. Type a job title first."
_BAD_PICK = "That number isn't in the list. Reply with a number from the options."
_BAD_SKILL = "There's no skill {number} in that list. Use a number from the skills I just listed."
_CLEARED = "Conversation cleared."
_PICKER_EVENTS = "picker-events.jsonl"
_PAYLOAD_CLAUSE = "full list is in the result payload"
_DETAILS_CLAUSE = "full list is in query details"


@dataclass(frozen=True)
class ChatReply:
    text: str
    quit: bool = False
    source_note: str | None = None  # e.g. "ESCO" only for map facts
    bound_label: str | None = None
    pending_count: int = 0
    # Set by /model. The caller owns the client for the process lifetime, so a
    # switch has to travel back out rather than be applied in here.
    new_llm_client: LLMClient | None = None
    # /login. The kernel has no console, and the key must not travel through
    # the normal input path, so the prompt is the caller's job.
    request_key: bool = False
    # /model with no argument. The picker needs a terminal, which the kernel
    # does not have, so the caller runs it and reports back.
    request_model_pick: bool = False


class TurnRunner(Protocol):
    def __call__(
        self,
        question: str,
        *,
        bound_node: NodeRef | None = None,
        force_capability: str | None = None,
    ) -> TurnOutcome: ...


def handle_line(
    state: SessionState,
    text: str,
    *,
    runner: TurnRunner,
    llm_client: LLMClient | None = None,
) -> ChatReply:
    """Mutate state (transcript, binding, pending). Never search for a bare number."""
    routed = route_line(text)
    if routed.kind == "command":
        return _handle_command(state, text)
    if routed.kind == "help_plain":
        intro = phrase_chat(
            llm_client,
            user_text=model_view(state, text),
            fallback=HELP_INTRO,
            hint=(
                "User asked what you can do. One short paragraph, then they will see commands."
                + _bound_title_hint(state)
            ),
        )
        body = f"{intro}\n\n{COMMANDS_BLOCK}" if uses_chat_phrasing(llm_client) else HELP_TEXT
        return _finish(state, text, body)
    if routed.kind == "greet":
        said = phrase_chat(
            llm_client,
            user_text=model_view(state, text),
            fallback=GREETING,
            hint=(
                "User greeted you. Invite them to name a job or skill. Do not look anything up."
                + _bound_title_hint(state)
            ),
        )
        return _finish(state, text, said)
    if routed.kind == "advice":
        said = phrase_chat(
            llm_client,
            user_text=text,
            fallback=ADVICE_REFUSE,
            hint="Refuse personal advice. Offer to locate a title or list skills of a bound job.",
        )
        return _finish(state, text, said)
    if routed.kind == "catalogue":
        said = phrase_chat(
            llm_client,
            user_text=text,
            fallback=CATALOGUE_REFUSE,
            hint=(
                "User asked to list every job or occupation. Refuse a full dump. "
                "Offer to pin one title. Do not invent a catalogue."
                + _bound_title_hint(state)
            ),
        )
        return _finish(state, text, said)
    if routed.kind == "pick":
        return _handle_pick(state, text, routed.pick or 0, runner=runner, llm_client=llm_client)
    if is_expand_list(text) or (is_bare_yes(text) and can_expand_connect(state.last_result)):
        return _handle_expand(state, text, runner=runner)
    mention = parse_skill_mention(text)
    stored = state.last_result
    if mention is None and stored is not None and can_expand_connect(stored):
        mention = skill_index_by_label(stored, text)
    if mention is not None and can_expand_connect(stored):
        return _focus_skill(state, mention, text, runner=runner, llm_client=llm_client)
    return _handle_map(state, routed.text, runner=runner, llm_client=llm_client)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _record(state: SessionState, role: Literal["user", "assistant", "system"], text: str) -> None:
    state.transcript.append(TranscriptLine(role=role, text=text, ts=_now()))


def _reply(
    state: SessionState,
    text: str,
    *,
    quit: bool = False,
    source_note: str | None = None,
    new_llm_client: LLMClient | None = None,
    request_key: bool = False,
    request_model_pick: bool = False,
) -> ChatReply:
    return ChatReply(
        text=text,
        quit=quit,
        source_note=source_note,
        new_llm_client=new_llm_client,
        request_key=request_key,
        request_model_pick=request_model_pick,
        bound_label=state.binding.node.pref_label if state.binding is not None else None,
        pending_count=len(state.pending),
    )


def _finish(state: SessionState, user_text: str, assistant_text: str) -> ChatReply:
    _record(state, "user", user_text)
    _record(state, "assistant", assistant_text)
    return _reply(state, assistant_text)


def _copy_into(state: SessionState, loaded: SessionState) -> None:
    state.session_id = loaded.session_id
    state.name = loaded.name
    state.transcript = list(loaded.transcript)
    state.binding = loaded.binding
    state.pending = list(loaded.pending)
    state.last_result = loaded.last_result


def _handle_command(state: SessionState, text: str) -> ChatReply:
    try:
        command = parse_command(text)
    except UnknownCommand as exc:
        return _finish(state, text, UNKNOWN_COMMAND.format(token=str(exc)))

    assert command is not None
    if command.name == "quit":
        return _reply(state, "", quit=True)
    if command.name == "help":
        return _finish(state, text, HELP_TEXT)
    if command.name == "save":
        _record(state, "user", text)
        path = save_session(state, name=command.argument)
        message = f"Saved session to {path}"
        _record(state, "assistant", message)
        return _reply(state, message)
    if command.name == "resume":
        loaded = (
            load_last() if command.argument == "last" else load_session(command.argument or "last")
        )
        _copy_into(state, loaded)
        message = f"Resumed session {state.name or state.session_id}."
        return _finish(state, text, message)
    if command.name == "clear":
        _copy_into(state, clear_conversation(state))
        return _finish(state, text, _CLEARED)
    if command.name == "model":
        if command.argument is None:
            _record(state, "user", text)
            return _reply(state, "", request_model_pick=True)
        return _handle_model(state, text, command.argument)
    if command.name == "login":
        if command.argument is not None:
            return _finish(
                state,
                "/login",
                "For security, do not paste a key after /login. Run /login, then paste it at the "
                "hidden prompt.",
            )
        _record(state, "user", "/login")
        return _reply(state, "", request_key=True)
    return _finish(state, text, UNKNOWN_COMMAND.format(token=text.split()[0]))


def _handle_model(state: SessionState, text: str, argument: str) -> ChatReply:
    """List free models, or switch to one. Never leaves the session clientless."""
    catalogue: list[FreeModel] = []
    if argument.isascii() and argument.isdigit():
        catalogue, note = load_catalogue()
        if note:
            return _finish(state, text, f"Did not switch: {note}")

    try:
        choice = resolve(argument, catalogue)
        client = apply(choice)
    except SwitchError as exc:
        return _finish(state, text, f"Did not switch: {exc}")

    _record(state, "user", text)
    message = f"Now using {choice.label()}."
    _record(state, "assistant", message)
    return _reply(state, message, new_llm_client=client)


def _last_map_query(state: SessionState) -> str:
    for line in reversed(state.transcript):
        if line.role == "user" and route_line(line.text).kind == "map":
            return line.text
    return ""


def _write_picker_event(
    state: SessionState, *, query: str, pending: list[PendingChoice], chosen_id: str
) -> None:
    key = state.name or state.session_id
    directory = sessions_dir() / key
    directory.mkdir(parents=True, exist_ok=True)
    event = {
        "query": query,
        "candidates": [choice.node.id for choice in pending],
        "chosen_id": chosen_id,
        "timestamp": _now(),
    }
    with (directory / _PICKER_EVENTS).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def _bound_title_hint(state: SessionState) -> str:
    if state.binding is None:
        return ""
    return f" Bound title this session: {state.binding.node.pref_label}. Do not pretend you forgot."


def _handle_pick(
    state: SessionState,
    text: str,
    number: int,
    *,
    runner: TurnRunner,
    llm_client: LLMClient | None,
) -> ChatReply:
    if state.pending:
        _record(state, "user", text)
        try:
            node = bind_pick(state.pending, number)
        except ValueError:
            if can_expand_connect(state.last_result):
                return _focus_skill(
                    state,
                    number,
                    text,
                    runner=runner,
                    llm_client=llm_client,
                    record_user=False,
                )
            _record(state, "assistant", _BAD_PICK)
            return _reply(state, _BAD_PICK)
        state.binding = LastBinding(node=node)
        _write_picker_event(
            state,
            query=_last_map_query(state),
            pending=state.pending,
            chosen_id=node.id,
        )
        message = f"Bound {node.pref_label}."
        _record(state, "assistant", message)
        return _reply(state, message)
    if can_expand_connect(state.last_result):
        return _focus_skill(state, number, text, runner=runner, llm_client=llm_client)
    _record(state, "user", text)
    _record(state, "assistant", _NO_PENDING)
    return _reply(state, _NO_PENDING)


def _focus_skill(
    state: SessionState,
    number: int,
    text: str,
    *,
    runner: TurnRunner,
    llm_client: LLMClient | None,
    record_user: bool = True,
) -> ChatReply:
    if record_user:
        _record(state, "user", text)
    stored = state.last_result
    miss = _BAD_SKILL.format(number=number)
    if stored is None:
        _record(state, "assistant", miss)
        return _reply(state, miss)
    try:
        skill = skill_from_connect(stored, number)
    except ValueError:
        _record(state, "assistant", miss)
        return _reply(state, miss)

    occupation = (
        state.binding.node.pref_label if state.binding is not None else stored.nodes[0].pref_label
    )
    neighbors: list[NodeRef] = []
    phrase_result = stored
    lowered = text.casefold()
    wants_neighbors = any(
        word in lowered for word in ("neighbor", "neighbour", "related", "who uses")
    )
    if wants_neighbors and followup_connect_request("neighbors of that", skill) is not None:
        outcome = runner(
            "neighbors of that",
            bound_node=skill,
            force_capability=CAPABILITY_CONNECT,
        )
        phrase_result = outcome.result
        neighbors = phrase_result.nodes[1:] if phrase_result.nodes else []
    else:
        phrase_result = AgentResult(
            capability="locate",
            suite=skill.suite,
            nodes=[skill],
            evidence=list(stored.evidence),
            confidence=stored.confidence,
        )

    fallback = f"Skill {number} on {occupation}: {skill.pref_label} (id {skill.id})"
    message = phrase_map(
        llm_client,
        question=text,
        result=phrase_result,
        fallback=fallback,
        card=skill_card(skill, number=number, occupation=occupation, neighbors=neighbors),
    )
    _record(state, "assistant", message)
    source = phrase_result.suite.upper() if phrase_result.suite else None
    return _reply(state, message, source_note=source)


def _map_answer(text: str) -> str:
    """Kernel-only rewrite of CLI JSON prose for chat."""
    return text.replace(_PAYLOAD_CLAUSE, _DETAILS_CLAUSE)


def _from_outcome(
    state: SessionState,
    outcome: TurnOutcome,
    *,
    question: str,
    llm_client: LLMClient | None,
) -> ChatReply:
    result = outcome.result
    state.last_result = result
    text = outcome.answer
    if "ambiguous" in result.warnings:
        pending = choices_from_result(result)
        omitted = max(0, len(result.nodes) - len(pending))
        state.pending = pending
        state.binding = None
        intro = phrase_chat(
            llm_client,
            user_text=question,
            fallback=f'I found several matches for "{question}". Which one did you mean?',
            hint=ambiguous_intro_card(
                question, [c.node.pref_label for c in pending], omitted=omitted
            ),
            mode="intro",
        )
        text = render_picker(question, pending, omitted=omitted, intro=intro)
    elif is_unimplemented_pathfind(result):
        # Do not let the phrasing model invent a gap, curriculum, or skills.
        # The graph did not walk; the typed warning is the answer.
        text = outcome.answer
    elif "not_found" in result.warnings:
        text = phrase_chat(
            llm_client,
            user_text=question,
            fallback=LOCATE_MISS,
            hint=(
                "Search missed. One or two sentences. It is a miss, not a maybe. "
                "Do not name occupations or skills as facts. Do not list related jobs."
            ),
            mode="miss",
        )
    elif result.capability == "locate" and result.nodes:
        state.binding = LastBinding(node=result.nodes[0])
        state.pending = []
        record = f"{_map_answer(outcome.answer)}\n\n{MAP_NEXT_STEP}"
        phrased = phrase_map(
            llm_client,
            question=question,
            result=result,
            fallback=record,
            card=locate_card(result),
        )
        if uses_chat_phrasing(llm_client) and phrased.strip() != record.strip():
            text = f"{phrased}\n\n{record}"
        else:
            text = record
    elif result.capability == "connect" and result.nodes:
        state.binding = LastBinding(node=result.nodes[0])
        fallback = _map_answer(outcome.answer)
        if not uses_chat_phrasing(llm_client):
            fallback = f"{fallback}\n\n{MAP_NEXT_STEP}"
        text = phrase_map(
            llm_client,
            question=question,
            result=result,
            fallback=fallback,
            card=connect_card(result),
        )
    else:
        text = _map_answer(outcome.answer)
    source = result.suite.upper() if result.suite else None
    return _reply(state, text, source_note=source)


def _handle_expand(state: SessionState, text: str, *, runner: TurnRunner) -> ChatReply:
    """Replay the last Connect list, or widen the last occupation picker."""
    _record(state, "user", text)
    result = state.last_result
    if can_expand_locate(result):
        assert result is not None
        pending = choices_from_result(result, limit=len(result.nodes))
        omitted = max(0, len(result.nodes) - len(pending))
        state.pending = pending
        query = _last_map_query(state) or text
        message = render_picker(query, pending, omitted=omitted)
        _record(state, "assistant", message)
        source = result.suite.upper() if result.suite else None
        return _reply(state, message, source_note=source)
    if not can_expand_connect(result) and state.binding is not None:
        outcome = runner(
            "list the skills",
            bound_node=state.binding.node,
            force_capability=CAPABILITY_CONNECT,
        )
        result = outcome.result
        state.last_result = result
        if result.nodes:
            state.binding = LastBinding(node=result.nodes[0])
    if not can_expand_connect(result):
        message = (
            "I don't have a skill list stored for this session yet. "
            "Name a job title, then ask for its skills."
        )
        _record(state, "assistant", message)
        return _reply(state, message)
    assert result is not None
    message = render_connect_list(result)
    _record(state, "assistant", message)
    source = result.suite.upper() if result.suite else None
    return _reply(state, message, source_note=source)


def _handle_map(
    state: SessionState,
    text: str,
    *,
    runner: TurnRunner,
    llm_client: LLMClient | None,
) -> ChatReply:
    _record(state, "user", text)
    bound = state.binding.node if state.binding is not None else None
    if bound is not None and followup_connect_request(text, bound) is not None:
        outcome = runner(text, bound_node=bound, force_capability=CAPABILITY_CONNECT)
    else:
        outcome = runner(text, bound_node=bound)
    reply = _from_outcome(state, outcome, question=text, llm_client=llm_client)
    _record(state, "assistant", reply.text)
    return reply
