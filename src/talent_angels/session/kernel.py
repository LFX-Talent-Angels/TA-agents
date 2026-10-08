"""Conversation kernel: classify a line, mutate session, call run_turn only for map-work."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from talent_angels.assistant.answer import CONNECT_PREVIEW_CAP, summarize_result
from talent_angels.assistant.connect_request import followup_connect_request, is_describe_followup
from talent_angels.assistant.intent import CAPABILITY_CONNECT
from talent_angels.assistant.llm_plan import PlanDraft, denied_subject, is_statement
from talent_angels.assistant.merge import suite_heading
from talent_angels.assistant.suite_select import resolve_show_token
from talent_angels.assistant.synthesize import synthesize
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm import LLMClient
from talent_angels.llm.protocol import turn_cancelled, turn_deadline
from talent_angels.memory.episodes import recent_episodes
from talent_angels.memory.erase import erase_all, erase_session, erase_summary
from talent_angels.memory.profile import (
    drop_standing_matching,
    profile_facts,
    profile_titles,
    write_goal,
    write_rejected,
    write_standing,
)
from talent_angels.session.advice import advice_plan
from talent_angels.session.budget import model_view
from talent_angels.session.catalog import FreeModel
from talent_angels.session.commands import UnknownCommand, parse_command
from talent_angels.session.compare_view import render_compare
from talent_angels.session.copy import (
    ADVICE_REFUSE,
    CATALOGUE_REFUSE,
    COMMANDS_BLOCK,
    CONNECT_MORE_HINT,
    GREETING,
    HELP_INTRO,
    HELP_TEXT,
    LOCATE_MISS,
    MAP_NEXT_STEP,
    PATHFIND_REDIRECT,
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
from talent_angels.session.lead import lead, pick_intent
from talent_angels.session.models import (
    AreaChoice,
    LastBinding,
    PendingChoice,
    SessionState,
    TranscriptLine,
)
from talent_angels.session.narrow import area_by_letter, area_choices, narrow_decision, render_areas
from talent_angels.session.phrase import (
    connect_card,
    locate_card,
    phrase_chat,
    phrase_map,
    skill_card,
    uses_chat_phrasing,
)
from talent_angels.session.picker import PICKER_LIMIT, bind_pick, choices_from_result, render_picker
from talent_angels.session.recall import recall_reply
from talent_angels.session.router import route_line
from talent_angels.session.store import (
    clear_conversation,
    load_last,
    load_session,
    new_session,
    save_session,
    sessions_dir,
)
from talent_angels.session.switch import SwitchError, apply, load_catalogue, resolve
from talent_angels.session.working_set import pair_followup, refers_to_one, remember
from talent_angels.skills.connect.compare import CAPABILITY_COMPARE
from talent_angels.skills.locate.areas import AreaRequest

_NO_PENDING = "There's no numbered list to pick from. Type a job title first."
_BAD_PICK = "That number isn't in the list. Reply with a number from the options."
_BAD_SKILL = "There's no skill {number} in that list. Use a number from the skills I just listed."
_PICKER_EVENTS = "picker-events.jsonl"
_PAYLOAD_CLAUSE = "full list is in the result payload"
_DETAILS_CLAUSE = "full list is in query details"


def _session_dir_for(state: SessionState) -> Path:
    """Where this session's files live — the same key save_session() writes to.

    Resolved before the state is replaced by a fresh session, so a reset
    deletes the files of the conversation being discarded rather than the new
    empty one's, which does not exist on disk yet.
    """
    return sessions_dir() / (state.name or state.session_id)


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
        bound_nodes: dict[str, NodeRef] | None = None,
        force_capability: str | None = None,
        area: AreaRequest | None = None,
    ) -> TurnOutcome: ...


def handle_line(
    state: SessionState,
    text: str,
    *,
    runner: TurnRunner,
    llm_client: LLMClient | None = None,
) -> ChatReply:
    """Mutate state (transcript, binding, pending). Never search for a bare number.

    One line is one turn with one clock (``turn_deadline``): every model call it
    makes, from narrowing to phrasing, shares it.
    """
    with turn_deadline():
        return _handle_line(state, text, runner=runner, llm_client=llm_client)


def _handle_line(
    state: SessionState,
    text: str,
    *,
    runner: TurnRunner,
    llm_client: LLMClient | None = None,
) -> ChatReply:
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
        thanked = text.strip().casefold().startswith("thank")
        said = phrase_chat(
            llm_client,
            user_text=model_view(state, text),
            fallback="You're welcome. Name another job or skill whenever you like."
            if thanked
            else GREETING,
            hint=(
                "User thanked you. One short sentence; offer a next step. Do not welcome "
                "them again or introduce yourself."
                if thanked
                else "User greeted you. Invite them to name a job or skill. "
                "Do not look anything up."
            )
            + _bound_title_hint(state),
        )
        return _finish(state, text, said)
    if routed.kind == "advice":
        current, goal = profile_titles()
        compared = (state.recent[-2], state.recent[-1]) if _compared_last(state) else None
        grounded = advice_plan(text, current=current, goal=goal, compared=compared)
        if grounded is not None:
            return _handle_map(
                state,
                text,
                runner=runner,
                llm_client=llm_client,
                question=grounded.question,
                preface=grounded.preface,
            )
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
                "Offer to pin one title. Do not invent a catalogue." + _bound_title_hint(state)
            ),
        )
        return _finish(state, text, said)
    if routed.kind == "pick":
        return _handle_pick(state, text, routed.pick or 0, runner=runner, llm_client=llm_client)
    if is_expand_list(text) or (is_bare_yes(text) and can_expand_connect(state.last_result)):
        return _handle_expand(state, text, runner=runner)
    if routed.kind == "show_suite":
        return _handle_show(state, text, routed.show_token or "")
    if routed.kind == "chat":
        return _handle_chat(state, text, llm_client=llm_client)
    if routed.kind == "recall":
        return _finish(state, text, recall_reply(text, state.recent, recent_episodes(limit=30)))
    if (state.pending or state.areas) and _may_narrow(state, routed.text):
        narrowed = _handle_narrow(state, routed.text, runner=runner, llm_client=llm_client)
        if narrowed is not None:
            return narrowed
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


def _set_bind(state: SessionState, node: NodeRef) -> None:
    label = (node.pref_label or "").strip()
    if not label or label.casefold() == "none":
        return
    state.bindings[node.suite] = node
    state.binding = LastBinding(node=node)


def _apply_profile_intent(intent: str | None, node: NodeRef) -> None:
    """Write what the user said about themselves; a plain lookup writes nothing."""
    if turn_cancelled():
        return  # the user stopped waiting: an abandoned turn writes nothing
    if intent == "goal":
        write_goal(node)
    elif intent == "reject":
        write_rejected(node)
    elif intent == "standing":
        write_standing(node)


def _bound_status(state: SessionState) -> str | None:
    if len(state.bindings) > 1:
        return " · ".join(
            f"{suite_heading(suite)} {node.pref_label}" for suite, node in state.bindings.items()
        )
    if state.binding is not None:
        return state.binding.node.pref_label
    if state.bindings:
        return next(iter(state.bindings.values())).pref_label
    return None


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
        bound_label=_bound_status(state),
        pending_count=len(state.pending),
    )


def _finish(state: SessionState, user_text: str, assistant_text: str) -> ChatReply:
    _record(state, "user", user_text)
    _record(state, "assistant", assistant_text)
    return _reply(state, assistant_text)


def _copy_into(state: SessionState, loaded: SessionState) -> None:
    """Replace every field of ``state`` in place.

    Field-by-field copying missed ``bindings`` and ``last_results``: after
    ``/reset`` the old per-suite binding kept steering turns and was saved back
    to disk. Iterating the model's fields means a new field cannot be missed.
    """
    for field_name in SessionState.model_fields:
        setattr(state, field_name, copy.deepcopy(getattr(loaded, field_name)))


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
        previous_name = state.name
        try:
            path = save_session(state, name=command.argument)
        except ValueError:
            state.name = previous_name
            message = "Session names use letters, digits, - and _ only (up to 64)."
            _record(state, "assistant", message)
            return _reply(state, message)
        message = f"Saved session to {path}"
        _record(state, "assistant", message)
        return _reply(state, message)
    if command.name == "resume":
        try:
            loaded = (
                load_last()
                if command.argument in (None, "", "last")
                else load_session(command.argument)
            )
        except (OSError, ValueError, KeyError):
            return _finish(state, text, f"No saved session named {command.argument!r}.")
        _copy_into(state, loaded)
        message = f"Resumed session {state.name or state.session_id}."
        return _finish(state, text, message)
    if command.name == "reset":
        # Session scope: forget the conversation, keep who the user is. The
        # session's own files go too — the transcript is the largest store of
        # the user's own words and previously survived every reset, since
        # erase_person() only covered the profile and the episode table.
        # A session the user named with /save is something they asked to keep:
        # step away from it instead of erasing it.
        saved_name = state.name
        old_dir = _session_dir_for(state)
        fresh = new_session()
        _copy_into(state, clear_conversation(fresh))
        if saved_name:
            return _finish(
                state,
                text,
                f"Starting a fresh conversation. Saved session {saved_name!r} is kept; "
                f"/resume {saved_name} to go back.",
            )
        erased = erase_session(old_dir)
        return _finish(state, text, f"{erase_summary(erased)} Starting a fresh conversation.")
    if command.name == "reset-all":
        # Full scope: conversation, profile, episode history, any externally
        # configured run-log, and a VACUUM so the bytes leave the file. A mentee
        # who asks to be forgotten must actually be.
        old_dir = _session_dir_for(state)
        fresh = new_session()
        _copy_into(state, clear_conversation(fresh))
        erased = erase_all(session_dir=old_dir)
        return _finish(state, text, f"{erase_summary(erased)} Forgotten.")
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
        _set_bind(state, node)
        state.recent = remember(state.recent, node.pref_label)
        _apply_profile_intent(state.pending_profile_intent, node)
        state.pending_profile_intent = None
        state.areas = []
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


def _turn_results(outcome: TurnOutcome) -> tuple[AgentResult, ...]:
    if outcome.results:
        return outcome.results
    return (outcome.result,)


def _source_note_for(results: tuple[AgentResult, ...]) -> str | None:
    names: list[str] = []
    for result in results:
        if not result.suite:
            continue
        label = suite_heading(result.suite)
        if label not in names:
            names.append(label)
    return " · ".join(names) if names else None


def _connect_bridge_intro(results: Sequence[AgentResult]) -> str:
    """GAP B: frame stacked per-suite connect blocks as one answer, not two.

    Skills never merge across suites here — ARCHITECTURE.md is explicit that
    node ids are suite-scoped and there is no cross-suite identity without an
    explicit crosswalk, so pretending ESCO's and O*NET's skill lists are "the
    same list" would be a real correctness bug, not a formatting nicety. This
    only adds one framing sentence above the full, still-separate per-suite
    cards that follow — no information is dropped or merged.

    Deterministic, no LLM call: the framing is fixed prose, and the team is
    already watching latency, so this does not add a network round trip to
    every multi-suite connect turn.
    """
    hits = [r for r in results if r.capability == "connect" and r.nodes]
    headings: list[str] = []
    for result in hits:
        heading = suite_heading(result.suite) if result.suite else "Map"
        if heading not in headings:
            headings.append(heading)
    if len(headings) < 2:
        return ""
    if len(headings) == 2:
        joined = f"{headings[0]} and {headings[1]}"
    else:
        joined = ", ".join(headings[:-1]) + f", and {headings[-1]}"
    return (
        f"{joined} both answer this — separate official records, not one shared id. "
        "Here is what each map's skills say:"
    )


def _connect_more_hint(result: AgentResult) -> str:
    """GAP F: name the words that actually widen a truncated skill list.

    ``connect_card``/``summarize_result`` already preview CONNECT_PREVIEW_CAP
    skills; beyond that, ``is_expand_list`` already understands "show more
    skills" (session.followup) — the gap was that nothing ever told the user
    those words work. Returns "" when there is nothing left to expand to.
    """
    if result.capability != "connect" or not result.nodes:
        return ""
    if len(result.nodes) - 1 <= CONNECT_PREVIEW_CAP:
        return ""
    return CONNECT_MORE_HINT


def _renumber_pending(choices: list[PendingChoice], *, start: int) -> list[PendingChoice]:
    return [
        PendingChoice(number=start + index, node=choice.node, group_label=choice.group_label)
        for index, choice in enumerate(choices)
    ]


def _prose_then_facts(phrased: str, fallback: str, facts: str) -> str:
    """Model prose may introduce the facts; it never replaces them.

    With a real model the phrased text used to *replace* the deterministic list,
    so a connect turn showed no skill names at all (the prompt forbids numbered
    lists) and a locate turn lost its id/confidence line although the prompt
    promised it is "printed under your text".
    """
    if phrased.strip() == fallback.strip():
        return fallback
    return f"{phrased}\n\n{facts}"


def _locate_fact_line(result: AgentResult) -> str:
    """The graph facts under a phrased locate answer — no raw ids in user text."""
    if not result.nodes:
        return ""
    top = result.nodes[0]
    confidence = f", confidence {result.confidence:.0%}" if result.confidence is not None else ""
    return f"Map record: **{top.pref_label}** ({top.kind}{confidence})"


def _render_unique_block(
    result: AgentResult,
    *,
    question: str,
    llm_client: LLMClient | None,
    heading: str | None,
) -> str:
    fallback = _map_answer(summarize_result(result))
    if result.capability == "connect" and result.nodes:
        phrased = phrase_map(
            llm_client,
            question=question,
            result=result,
            fallback=fallback,
            card=connect_card(result),
        )
        body = _prose_then_facts(
            phrased, fallback, render_connect_list(result, cap=CONNECT_PREVIEW_CAP)
        )
        hint = _connect_more_hint(result)
        if hint and hint not in body:
            body = f"{body}\n\n{hint}"
    else:
        # The CLI summary carries node ids; chat shows the record line instead.
        fallback = _locate_fact_line(result) or fallback
        phrased = phrase_map(
            llm_client,
            question=question,
            result=result,
            fallback=fallback,
            card=locate_card(result),
        )
        body = _prose_then_facts(phrased, fallback, _locate_fact_line(result))
    if heading:
        # Plain ATX heading — TUI treats **bold** lines as picker group names.
        return f"## {heading}\n\n{body}"
    return body


def _remember_topic(state: SessionState, draft: PlanDraft | None, node: NodeRef) -> None:
    """Add this turn's title to the working set, as the user searched it."""
    subject = draft.subject if draft is not None and draft.subject else node.pref_label
    state.recent = remember(state.recent, subject)


def _compared_last(state: SessionState) -> bool:
    return (
        not state.bindings
        and len(state.recent) >= 2
        and any(result.capability == CAPABILITY_COMPARE for result in state.last_results)
    )


def _remember_compared(state: SessionState, draft: PlanDraft | None, result: AgentResult) -> None:
    if draft is not None and draft.subject and draft.secondary_subject:
        titles = [draft.subject, draft.secondary_subject]
    else:
        titles = [node.pref_label for node in result.nodes[:2]]
    for title in titles:
        state.recent = remember(state.recent, title)
    # After a compare, "it" could be either title: nothing stays bound.
    state.binding = None
    state.bindings.clear()


def _searched_for(question: str, draft: PlanDraft | None) -> str:
    """The words the search used ("nurse"), not the whole question, for a picker."""
    subject = draft.subject if draft is not None else None
    return subject.strip() if subject and subject.strip() else question


def _from_single_outcome(
    state: SessionState,
    result: AgentResult,
    outcome: TurnOutcome,
    *,
    question: str,
    llm_client: LLMClient | None,
) -> ChatReply:
    state.last_result = result
    text = outcome.answer
    draft = getattr(outcome, "plan_draft", None)
    state.pending_profile_intent = None
    state.areas = []
    if "ambiguous" not in result.warnings:
        # A turn without a new list closes the old one: a later hint must not
        # narrow a list from several turns back.
        state.pending = []
        state.list_topic = ""
    if "ambiguous" in result.warnings:
        pending = choices_from_result(result)
        omitted = max(0, len(result.nodes) - len(pending))
        state.pending = pending
        state.pending_profile_intent = pick_intent(draft)
        state.binding = None
        state.bindings.clear()
        searched = _searched_for(question, draft)
        state.list_topic = searched
        # What was understood and what happens next, written in code (lead.py).
        intro = lead(question, draft, [result])
        text = render_picker(searched, pending, omitted=omitted, intro=intro)
        text = _with_areas(state, text, outcome, searched=searched, draft=draft)
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
        _set_bind(state, result.nodes[0])
        _remember_topic(state, draft, result.nodes[0])
        _draft = getattr(outcome, "plan_draft", None)
        _apply_profile_intent(_draft.profile_intent if _draft else None, result.nodes[0])
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
            text = f"{phrased}\n\n{_locate_fact_line(result)}\n\n{MAP_NEXT_STEP}"
        else:
            text = record
    elif result.capability == "connect" and result.nodes:
        _set_bind(state, result.nodes[0])
        _remember_topic(state, draft, result.nodes[0])
        fallback = _map_answer(outcome.answer)
        if not uses_chat_phrasing(llm_client):
            fallback = f"{fallback}\n\n{MAP_NEXT_STEP}"
        phrased = phrase_map(
            llm_client,
            question=question,
            result=result,
            fallback=fallback,
            card=connect_card(result),
        )
        text = _prose_then_facts(
            phrased, fallback, render_connect_list(result, cap=CONNECT_PREVIEW_CAP)
        )
        hint = _connect_more_hint(result)
        if hint and hint not in text:
            text = f"{text}\n\n{hint}"
    elif result.capability == CAPABILITY_COMPARE:
        state.pending = []
        _remember_compared(state, draft, result)
        text = render_compare(result)
    elif any(w.startswith("capability_not_implemented") for w in result.warnings):
        text = PATHFIND_REDIRECT
    else:
        text = _map_answer(outcome.answer)
    source = result.suite.upper() if result.suite else None
    return _reply(state, text, source_note=source)


def _with_areas(
    state: SessionState,
    text: str,
    outcome: TurnOutcome,
    *,
    searched: str,
    draft: PlanDraft | None,
) -> str:
    """Offer "which area?" under a pick list when the search was broad."""
    kind = draft.kind if draft is not None and draft.kind else "occupation"
    state.areas = area_choices(getattr(outcome, "areas", None) or {}, query=searched, kind=kind)
    if not state.areas:
        return text
    return f"{text}\n\n{render_areas(state.areas)}"


def _narrowed_reply(
    state: SessionState, text: str, choices: list[PendingChoice], intro: str
) -> ChatReply:
    numbered = _renumber_pending(choices[:PICKER_LIMIT], start=1)
    state.pending = numbered
    state.areas = []
    state.binding = None
    state.bindings.clear()
    message = render_picker(
        text,
        numbered,
        omitted=max(0, len(choices) - len(numbered)),
        intro=intro,
        include_source=False,
    )
    _record(state, "assistant", message)
    return _reply(state, message, source_note=_source_note_for(tuple(state.last_results)))


def _area_titles(state: SessionState, choice: AreaChoice, *, runner: TurnRunner) -> list[NodeRef]:
    """Every title of the search inside one area; the shown list when unsupported."""
    request = AreaRequest(
        suite=choice.suite, code=choice.code, query=choice.query, kind=choice.kind
    )
    outcome = runner(choice.query, area=request)
    result = outcome.result
    if result.nodes:
        state.last_result = result
        state.last_results = [result]
        return list(result.nodes)
    return [c.node for c in state.pending if c.group_label == choice.label]


def _may_narrow(state: SessionState, text: str) -> bool:
    """Only a list still waiting for a choice is narrowed, and never by a statement.

    After a pick the list stays for number corrections ("actually 3"), but a
    sentence then is a new request. "I am not a nurse" or "compare the two" is
    never a hint about the list.
    """
    if state.binding is not None and not state.areas:
        return False
    return not is_statement(text) and pair_followup(text, state.recent) is None


def _names_title(text: str, label: str) -> bool:
    """The user typed the title itself, singular or plural."""
    said = text.strip().rstrip(".!?").casefold()
    title = label.casefold()
    return said in (title, f"{title}s") or f"{said}s" == title


def _handle_narrow(
    state: SessionState,
    text: str,
    *,
    runner: TurnRunner,
    llm_client: LLMClient | None,
) -> ChatReply | None:
    """Narrow the open pick list, or None when ``text`` is a new request."""
    letter = area_by_letter(text, state.areas)
    if letter is not None:
        chosen_areas = [letter]
        options: list[PendingChoice] = []
    else:
        decision = narrow_decision(llm_client, text, state.pending, state.areas)
        if decision.action == "new":
            return None
        if not (decision.options or decision.areas):
            # About the list, but nothing on it fits: search again, keeping
            # the topic the list was for ("something with children, in healthcare").
            topic = state.list_topic
            if not topic:
                return None
            intent = state.pending_profile_intent
            reply = _handle_map(
                state, text, runner=runner, llm_client=llm_client, question=f"{text}, in {topic}"
            )
            if state.pending and state.pending_profile_intent is None:
                # "I want to become an engineer" … "something outdoors": a pick
                # from the new list is still the goal.
                state.pending_profile_intent = intent
            return reply
        options = [c for c in state.pending if c.number in decision.options]
        chosen_areas = [a for a in state.areas if a.letter in decision.areas]
        named = [c for c in options if _names_title(text, c.node.pref_label)]
        if len(named) == 1:
            # The user typed a listed title itself: that is a pick, not a hint.
            return _handle_pick(state, text, named[0].number, runner=runner, llm_client=llm_client)
    _record(state, "user", text)
    nodes = [c.node for c in options]
    for area in chosen_areas:
        nodes.extend(_area_titles(state, area, runner=runner))
    unique: list[NodeRef] = []
    for node in nodes:
        if all(node.id != kept.id for kept in unique):
            unique.append(node)
    if not unique:
        message = (
            "None of the titles on the list fit that. Describe it another way, or name a title."
        )
        _record(state, "assistant", message)
        return _reply(state, message)
    choices = [PendingChoice(number=i, node=n) for i, n in enumerate(unique, start=1)]
    if chosen_areas and not options:
        names = ", ".join(f"**{a.label}**" for a in chosen_areas)
        intro = f'{names}: titles matching "{chosen_areas[0].query}". Which one did you mean?'
    else:
        intro = f'These fit "{text.strip()}". Which one did you mean?'
    return _narrowed_reply(state, text, choices, intro)


def _stacked_cards(
    results: tuple[AgentResult, ...],
    *,
    question: str,
    llm_client: LLMClient | None,
) -> str:
    blocks: list[str] = []
    for result in results:
        heading = suite_heading(result.suite) if result.suite else "Map"
        if result.nodes and "ambiguous" not in result.warnings:
            blocks.append(
                _render_unique_block(
                    result, question=question, llm_client=llm_client, heading=heading
                )
            )
        else:
            blocks.append(f"## {heading}\n\n{_map_answer(summarize_result(result))}")
    return "\n\n---\n\n".join(blocks)


def _handle_show(state: SessionState, text: str, token: str) -> ChatReply:
    _record(state, "user", text)
    known = tuple(dict.fromkeys([*[item.suite for item in state.last_results], *state.bindings]))
    if not known:
        message = "I don't have a map card stored yet. Name a job title first."
        _record(state, "assistant", message)
        return _reply(state, message)
    resolved = resolve_show_token(token, known)
    if resolved is None:
        attached = ", ".join(suite_heading(name) for name in known)
        message = f"I don't have a source called {token!r}. Attached: {attached}."
        _record(state, "assistant", message)
        return _reply(state, message)
    if resolved == "all":
        message = _stacked_cards(tuple(state.last_results), question=token, llm_client=None)
        _record(state, "assistant", message)
        return _reply(state, message, source_note=_source_note_for(tuple(state.last_results)))
    match = next((item for item in state.last_results if item.suite == resolved), None)
    if match is None:
        message = f"No stored card for {suite_heading(resolved)}."
        _record(state, "assistant", message)
        return _reply(state, message)
    heading = suite_heading(resolved)
    message = _render_unique_block(match, question=token, llm_client=None, heading=heading)
    _record(state, "assistant", message)
    return _reply(state, message, source_note=heading)


def _profile_reply() -> str:
    facts = profile_facts()
    if not facts:
        return (
            'I don\'t have much about you yet. Tell me your current job ("I am a …") '
            "or a job you're aiming for (\"my goal is …\") and I'll note it."
        )
    lines = "\n".join(f"- {fact}" for fact in facts)
    return (
        f"Here is what you've told me:\n\n{lines}\n\n"
        "That is all I keep about you. Tell me your goal or current job to change it."
    )


def _handle_chat(
    state: SessionState,
    text: str,
    *,
    llm_client: LLMClient | None,  # noqa: ARG001 — signature matches other handlers
) -> ChatReply:
    """Questions about the user answer from the profile, written in code.

    A model asked to summarise the profile mixed it with the conversation and
    invented roles the user never claimed; the profile is data, so it is shown.
    """
    return _finish(state, text, _profile_reply())


def _from_outcome(
    state: SessionState,
    outcome: TurnOutcome,
    *,
    question: str,
    llm_client: LLMClient | None,
) -> ChatReply:
    results = _turn_results(outcome)
    state.last_results = list(results)
    state.areas = []
    if len(results) <= 1:
        return _from_single_outcome(
            state,
            results[0] if results else outcome.result,
            outcome,
            question=question,
            llm_client=llm_client,
        )

    preferred = next(
        (item for item in results if item.nodes and "ambiguous" not in item.warnings),
        results[0],
    )
    state.last_result = preferred

    blocks: list[str] = []
    #: Per-suite "no match" / "not available" blocks. Kept separately so they
    #: are shown whatever the other suites returned — they used to be dropped
    #: whenever another suite hit, while the footer still named both suites.
    miss_blocks: list[str] = []
    unique_cards: list[str] = []
    pending_all: list[PendingChoice] = []
    unique_bind: NodeRef | None = None
    any_hit = False
    all_miss = True
    _draft = getattr(outcome, "plan_draft", None)
    searched = _searched_for(question, _draft)

    cards_shown = any(
        (r.capability in ("connect", CAPABILITY_COMPARE) and r.nodes)
        or ("ambiguous" in r.warnings and r.nodes)
        for r in results
    )
    for result in results:
        heading = suite_heading(result.suite) if result.suite else "Map"
        if "ambiguous" in result.warnings and result.nodes:
            all_miss = False
            any_hit = True
            choices = _renumber_pending(choices_from_result(result), start=len(pending_all) + 1)
            omitted = max(0, len(result.nodes) - len(choices))
            # Written in code: a model call per suite for one fixed sentence
            # cost 4-9 seconds each and said nothing the list does not.
            intro = (
                f"{heading} titles that fit your request. Which one did you mean?"
                if "guided" in result.warnings
                else f'I found several {heading} matches for "{searched}". Which one did you mean?'
            )
            picker = render_picker(
                searched,
                choices,
                omitted=omitted,
                intro=intro,
                include_source=False,
            )
            blocks.append(f"## {heading}\n\n{picker}")
            pending_all.extend(choices)
            continue
        if any(w.startswith("capability_not_implemented") for w in result.warnings):
            blocks.append(f"## {heading}\n\n{PATHFIND_REDIRECT}")
            all_miss = False
            continue
        if not result.nodes or "not_found" in result.warnings:
            # A follow-up ("its skills") with nothing chosen in this suite is
            # not a search miss: the user still has to pick from its list.
            message = (
                f"No {heading} occupation is chosen yet. Pick one from its list or name one, "
                "then ask again."
                if {"bind_required", "no_subject"} & set(result.warnings)
                else LOCATE_MISS
            )
            blocks.append(f"## {heading}\n\n{message}")
            miss_blocks.append(f"## {heading}\n\n{message}")
            continue
        all_miss = False
        any_hit = True
        if result.capability == CAPABILITY_COMPARE:
            # Two titles, neither is "it": the compare binds nothing.
            unique_cards.append(f"## {heading}\n\n{render_compare(result)}")
            continue
        _set_bind(state, result.nodes[0])
        if unique_bind is None:
            unique_bind = result.nodes[0]
        unique_cards.append(
            _render_unique_block(
                result,
                question=question,
                # Cards are only printed beside a connect, compare or pick list;
                # otherwise synthesize() writes the answer and a phrased card
                # would be a paid model call thrown away.
                llm_client=llm_client if cards_shown else None,
                heading=heading,
            )
        )

    # A title the user named and the map resolved carries what they said about
    # it; a pick from suggestions only carries what pick_intent allows.
    said = _draft.profile_intent if _draft is not None else None
    intent = pick_intent(_draft)
    if said == "standing":
        # One current occupation per suite: every suite that resolved records it.
        for item in results:
            if item.nodes and "ambiguous" not in item.warnings and "not_found" not in item.warnings:
                _apply_profile_intent(said, item.nodes[0])
    elif unique_bind is not None:
        _apply_profile_intent(said, unique_bind)
        intent = None  # goal/reject are written once, not again on a later pick
    state.pending_profile_intent = intent if pending_all else None
    if unique_bind is not None:
        _remember_topic(state, _draft, unique_bind)
    compared = next((r for r in results if r.capability == CAPABILITY_COMPARE), None)
    if compared is not None:
        _remember_compared(state, _draft, compared)

    if pending_all:
        state.pending = pending_all
        state.list_topic = searched
        if not state.bindings:
            state.binding = None
    else:
        # No new list this turn (a compare, a miss, a unique hit): close the old
        # one, so a later hint cannot narrow a list from several turns back.
        state.pending = []
        state.list_topic = ""

    not_implemented = any(
        w.startswith("capability_not_implemented") for result in results for w in result.warnings
    )
    if not_implemented and not any_hit and not pending_all:
        # GAP-PF (multi-suite): every attached suite classifies the SAME question
        # to the SAME capability, so "not implemented" here is never partial —
        # if one suite says it, all of them do. Show the honest redirect instead
        # of falling through to synthesize(), which cannot tell "not implemented"
        # apart from a genuine search miss: both are zero-nodes results, and
        # _hit_phrase() treats any zero-nodes result as "no hit" (GAP-PF was
        # closed for the tool loop and the single-suite reply in agent_loop.py /
        # _from_single_outcome; this merge path had the same bug independently).
        text = PATHFIND_REDIRECT
    elif all_miss and not any_hit:
        text = phrase_chat(
            llm_client,
            user_text=question,
            fallback=LOCATE_MISS,
            hint=(
                "Search missed. One or two sentences. It is a miss, not a maybe. "
                "Do not name occupations or skills as facts. Do not list related jobs."
                + _bound_title_hint(state)
            ),
            mode="miss",
        )
    elif pending_all:
        # Deterministic, never the free-form synthesize() LLM narrative: an
        # ambiguous locate result's fact card has only candidate occupation
        # titles, no skill data (Connect never ran — there was no unique node
        # to hop from). Live testing against a real model found it invent a
        # specific skill list for one candidate it picked unprompted anyway,
        # in 1 of 5 identical runs, presented as graph fact, then still asked
        # the user to choose among all candidates in the very next paragraph.
        # synthesize_structured() produces the same safe framing
        # (`_hit_phrase` already says "<suite> has several matches" for an
        # ambiguous hit) with no model call, so there is nothing left to
        # hallucinate.
        # The short answer first: what was understood, what each map has, what
        # next. Code-written, like the rest of a pick list (lead.py).
        text = lead(question, _draft, results)
        extras = [*unique_cards]
        picker_blocks = [
            block for block in blocks if "I won't pick" in block or "Which one" in block
        ]
        extras.extend(picker_blocks)
        extras.extend(miss_blocks)
        if extras:
            text = text + "\n\n---\n\n" + "\n\n---\n\n".join(extras)
        text = _with_areas(state, text, outcome, searched=searched, draft=_draft)
    else:
        has_connect = any(
            r.capability in ("connect", CAPABILITY_COMPARE) and r.nodes for r in results
        )
        if has_connect and unique_cards:
            text = "\n\n---\n\n".join(unique_cards)
            intro = _connect_bridge_intro(results)
            if intro:
                text = f"{intro}\n\n{text}"
        else:
            text = synthesize(results, question=question, llm_client=llm_client)
        if has_connect and unique_cards and miss_blocks:
            text = text + "\n\n---\n\n" + "\n\n---\n\n".join(miss_blocks)
        if unique_bind is not None and MAP_NEXT_STEP not in text:
            text = f"{text}\n\n{MAP_NEXT_STEP}"

    return _reply(state, text, source_note=_source_note_for(results))


def _handle_expand(state: SessionState, text: str, *, runner: TurnRunner) -> ChatReply:
    """Replay the last Connect list, or widen the last occupation picker."""
    _record(state, "user", text)
    result = state.last_result
    pickers = [r for r in state.last_results if can_expand_locate(r)]
    if not pickers and can_expand_locate(result):
        assert result is not None
        pickers = [result]
    if pickers:
        # Every suite's picker, numbered continuously: widening only one suite
        # (and renumbering it from 1) broke the other suite's numbers.
        query = _last_map_query(state) or text
        pending: list[PendingChoice] = []
        picker_blocks: list[str] = []
        for item in pickers:
            choices = _renumber_pending(
                choices_from_result(item, limit=len(item.nodes)), start=len(pending) + 1
            )
            pending.extend(choices)
            picker = render_picker(query, choices, omitted=0, include_source=False)
            heading = suite_heading(item.suite) if item.suite else "Map"
            picker_blocks.append(f"## {heading}\n\n{picker}" if len(pickers) > 1 else picker)
        state.pending = pending
        message = "\n\n---\n\n".join(picker_blocks)
        _record(state, "assistant", message)
        return _reply(state, message, source_note=_source_note_for(tuple(pickers)))
    if not can_expand_connect(result) and state.binding is not None:
        bn = dict(state.bindings) if len(state.bindings) > 1 else None
        outcome = runner(
            "list the skills",
            bound_nodes=bn,
            bound_node=state.binding.node if not bn else None,
            force_capability=CAPABILITY_CONNECT,
        )
        result = outcome.result
        state.last_result = result
        if outcome.results:
            state.last_results = list(outcome.results)
        if result.nodes and (result.nodes[0].pref_label or "").strip():
            state.binding = LastBinding(node=result.nodes[0])
    if not can_expand_connect(result):
        message = (
            "I don't have a skill list stored for this session yet. "
            "Name a job title, then ask for its skills."
        )
        _record(state, "assistant", message)
        return _reply(state, message)
    assert result is not None
    # The bound suite's list first ("skill N" follow-ups index into it), then
    # every other suite's list — "show more" used to list only one suite.
    lists = [
        result,
        *(r for r in state.last_results if r.suite != result.suite and can_expand_connect(r)),
    ]
    if len(lists) == 1:
        message = render_connect_list(result)
    else:
        message = "\n\n---\n\n".join(
            f"## {suite_heading(r.suite) if r.suite else 'Map'}\n\n{render_connect_list(r)}"
            for r in lists
        )
    _record(state, "assistant", message)
    return _reply(state, message, source_note=_source_note_for(tuple(lists)))


def _describe_bound(
    state: SessionState,
    text: str,
    *,
    llm_client: LLMClient | None,  # noqa: ARG001 — signature matches other handlers
) -> ChatReply:
    """Explain already-bound occupations. No new Locate."""
    blocks: list[str] = []
    for suite, node in state.bindings.items():
        heading = suite_heading(suite)
        desc = (node.description or "").strip()
        if desc:
            body = desc
        else:
            body = f"{node.pref_label} is on the {heading} map. I don't have a definition stored."
        blocks.append(f"## {heading}\n\n**{node.pref_label}**\n\n{body}")
    message = "\n\n---\n\n".join(blocks)
    _record(state, "assistant", message)
    names = [suite_heading(s) for s in state.bindings]
    source_note = " · ".join(names) if names else None
    return _reply(state, message, source_note=source_note)


def _handle_map(
    state: SessionState,
    text: str,
    *,
    runner: TurnRunner,
    llm_client: LLMClient | None,
    question: str | None = None,
    preface: str = "",
) -> ChatReply:
    """``question`` replaces ``text`` for the search when code already rewrote it."""
    _record(state, "user", text)
    bound = state.binding.node if state.binding is not None else None
    bound_nodes = dict(state.bindings) if state.bindings else None
    # "compare the two" names its titles in code, from the working set.
    denied = denied_subject(text)
    dropped = drop_standing_matching(denied) if denied else []
    if dropped:
        # The denial names a saved current job: correct the profile, no search.
        titles = ", ".join(f"**{title}**" for title in dropped)
        gone = {title.casefold() for title in dropped}
        for suite, node in list(state.bindings.items()):
            if node.pref_label.casefold() in gone:
                del state.bindings[suite]
        if state.binding is not None and state.binding.node.pref_label.casefold() in gone:
            state.binding = None
        message = f"Noted: {titles} is no longer saved as your current job."
        _record(state, "assistant", message)
        return _reply(state, message)
    pair = question or pair_followup(text, state.recent)
    if pair is None and _compared_last(state) and refers_to_one(text):
        first, second = state.recent[-2:]
        message = (
            f"Which one do you mean: **{first}** or **{second}**? Name it and I'll look it up."
        )
        _record(state, "assistant", message)
        return _reply(state, message)
    if pair is None and bound_nodes and is_describe_followup(text, bound_nodes):
        return _describe_bound(state, text, llm_client=llm_client)
    question = pair or text
    outcome = runner(question, bound_node=bound, bound_nodes=bound_nodes)
    # Semantic suite switch: LLM detected user wants to see cached results on a specific suite.
    # Re-render from state.last_results (previous turn) without running a new query.
    _draft = getattr(outcome, "plan_draft", None)
    if _draft and _draft.suite_override and state.last_results:
        token = _draft.suite_override
        known = tuple(
            dict.fromkeys([item.suite for item in state.last_results] + list(state.bindings))
        )
        resolved = resolve_show_token(token, known)
        if resolved and resolved != "all":
            cached = next((r for r in state.last_results if r.suite == resolved), None)
            if cached is not None:
                heading = suite_heading(resolved)
                # Use structured output (no LLM) to match _handle_show — avoids the
                # LLM reading the profile prefix instead of the skill card.
                message = _render_unique_block(
                    cached, question=text, llm_client=None, heading=heading
                )
                _record(state, "assistant", message)
                return _reply(state, message, source_note=heading)
    reply = _from_outcome(state, outcome, question=text, llm_client=llm_client)
    if preface:
        reply = replace(reply, text=f"{preface}\n\n{reply.text}")
    _record(state, "assistant", reply.text)
    return reply
