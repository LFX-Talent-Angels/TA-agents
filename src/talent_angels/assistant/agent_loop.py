"""Main-assistant tool loop: the model chooses Locate/Connect suite tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from talent_angels.assistant.answer import (
    PATHFIND_UNAVAILABLE,
    PATHFIND_UNIMPLEMENTED_WARNING,
    is_terminal_locate,
    is_unimplemented_pathfind,
    summarize_result,
)
from talent_angels.assistant.intent import (
    CAPABILITY_CONNECT,
    CAPABILITY_LOCATE,
    CAPABILITY_PATHFIND,
    Capability,
    classify_capability,
    extract_locate_subject,
)
from talent_angels.assistant.llm_call import measure_complete
from talent_angels.assistant.planning import ExecutionPlan, build_plan_for_capability
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm import LLMClient, Message, ToolInvocation
from talent_angels.runlog import StageUsage, ToolCall
from talent_angels.skills.connect import connect
from talent_angels.skills.connect.models import ConnectRequest
from talent_angels.skills.locate import locate
from talent_angels.suites.measured import MeasuredSuite
from talent_angels.suites.protocol import SuiteTools

MAX_TOOL_ROUNDS = 4
MAX_COMPACT_NODES = 8
MAX_COMPACT_EDGES = 8

LOOP_SYSTEM = """You are the LFX Talent Angels main assistant.
Return ONLY a JSON object each turn. Do not invent node IDs or skills.

Call a graph tool:
{"tool":"search_nodes","text":"software developer","kind":"occupation"}
{"tool":"get_neighbors","node_id":"esco:occupation:...","rel_types":["HAS_SKILL"],"relation_filter":"essential"}

Or finish with the user-facing answer:
{"final":"one sentence using only returned labels, ids, and confidence"}

Search the occupation or skill phrase from the question (not the whole sentence).
Skills questions: search_nodes first, then get_neighbors with HAS_SKILL if the search is unique.
If TOOL_RESULT warnings include ambiguous or not_found, return final and stop. Do not search again.
If node_count is larger than the listed nodes, mention the count and a few examples.
Do not offer to fetch, paginate, or retrieve the rest.
"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_XML_TOOL = re.compile(
    r"<tool_call>\s*([A-Za-z0-9_]+)\s*(.*?)</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_XML_ARG = re.compile(
    r"<arg_key>\s*([^<]+?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class AgentLoopOutcome:
    answer: str
    result: AgentResult
    plan: ExecutionPlan
    stages: list[StageUsage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)


def _compact_result(result: AgentResult) -> dict[str, object]:
    omitted_nodes = max(0, len(result.nodes) - MAX_COMPACT_NODES)
    omitted_edges = max(0, len(result.edges) - MAX_COMPACT_EDGES)
    payload: dict[str, object] = {
        "capability": result.capability,
        "confidence": result.confidence,
        "warnings": result.warnings,
        "node_count": len(result.nodes),
        "edge_count": len(result.edges),
        "nodes": [
            {"id": node.id, "kind": node.kind, "pref_label": node.pref_label}
            for node in result.nodes[:MAX_COMPACT_NODES]
        ],
        "edges": [
            {
                "type": edge.type,
                "from": edge.source_node_id,
                "to": edge.target_node_id,
                "properties": edge.properties,
            }
            for edge in result.edges[:MAX_COMPACT_EDGES]
        ],
    }
    if omitted_nodes or omitted_edges:
        payload["truncated"] = True
        payload["omitted_nodes"] = omitted_nodes
        payload["omitted_edges"] = omitted_edges
    return payload


def _search_text_for(question: str, requested: str, *, already_searched: bool) -> str:
    canonical = extract_locate_subject(question)
    raw = requested.strip()
    if not already_searched:
        return canonical or raw
    if raw.casefold() == question.strip().casefold():
        return canonical or raw
    return raw or canonical


def _execute_tool(
    invocation: ToolInvocation,
    *,
    suite: MeasuredSuite,
    suite_name: str,
    located: AgentResult | None,
    question: str,
    already_searched: bool,
) -> AgentResult:
    args = invocation.arguments
    if invocation.name == "search_nodes":
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("search_nodes requires text")
        kind = args.get("kind")
        kind_value = kind if isinstance(kind, str) else None
        return locate(
            suite,
            suite_name,
            _search_text_for(question, text, already_searched=already_searched),
            kind=kind_value,
        )

    if invocation.name == "get_neighbors":
        node_id = args.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("get_neighbors requires node_id")
        rel_raw = args.get("rel_types")
        rel_types: tuple[str, ...]
        if isinstance(rel_raw, list) and rel_raw:
            rel_types = tuple(str(item) for item in rel_raw)
        else:
            rel_types = ("HAS_SKILL",)
        relation = args.get("relation_filter")
        relation_kind = relation if isinstance(relation, str) else None
        center = None
        if located is not None:
            center = next((node for node in located.nodes if node.id == node_id), None)
        if center is None:
            center = NodeRef(
                id=node_id,
                suite=suite_name,
                source=suite_name,
                source_id=node_id,
                kind="Occupation",
                pref_label="",
            )
        return connect(
            suite,
            suite_name,
            center,
            request=ConnectRequest(
                subject=center.pref_label or node_id,
                rel_types=rel_types,
                relation_kind=relation_kind,
            ),
            confidence=located.confidence if located is not None else None,
            locate_evidence=located.evidence if located is not None else (),
        )

    raise ValueError(f"unknown tool: {invocation.name}")


def _coerce_arg(raw: str) -> object:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_xml_tool_calls(text: str) -> list[ToolInvocation]:
    found: list[ToolInvocation] = []
    for match in _XML_TOOL.finditer(text):
        name = match.group(1).strip()
        arguments: dict[str, object] = {}
        for key, value in _XML_ARG.findall(match.group(2)):
            arguments[key.strip()] = _coerce_arg(value)
        found.append(ToolInvocation(id=name, name=name, arguments=arguments))
    return found


def looks_like_tool_markup(text: str) -> bool:
    lowered = text.lower()
    return "<tool_call>" in lowered or "<arg_key>" in lowered


def parse_loop_turn(text: str) -> tuple[list[ToolInvocation], str | None]:
    """Return tool invocations and/or a user-facing final string."""
    xml_calls = _parse_xml_tool_calls(text)
    if xml_calls:
        return xml_calls, None

    match = _JSON_OBJECT.search(text.strip())
    if match is not None:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            tool = payload.get("tool")
            if isinstance(tool, str) and tool:
                arguments = {key: value for key, value in payload.items() if key != "tool"}
                return [ToolInvocation(id=tool, name=tool, arguments=arguments)], None
            final = payload.get("final")
            if isinstance(final, str) and final.strip() and not looks_like_tool_markup(final):
                return [], final.strip()

    cleaned = text.strip()
    if not cleaned or looks_like_tool_markup(cleaned):
        return [], None
    return [], cleaned


def _structured_fallback(result: AgentResult) -> str:
    return summarize_result(result)


def _is_unique_locate(result: AgentResult) -> bool:
    return bool(result.nodes) and not is_terminal_locate(result)


def _capability_from_result(result: AgentResult | None) -> Capability:
    if result is None:
        return CAPABILITY_LOCATE
    if result.capability == CAPABILITY_CONNECT:
        return CAPABILITY_CONNECT
    if result.capability == CAPABILITY_PATHFIND:
        return CAPABILITY_PATHFIND
    return CAPABILITY_LOCATE


def _unimplemented_pathfind(suite_name: str) -> AgentLoopOutcome:
    result = AgentResult(
        capability=CAPABILITY_PATHFIND,
        suite=suite_name,
        warnings=[PATHFIND_UNIMPLEMENTED_WARNING],
    )
    return AgentLoopOutcome(
        answer=PATHFIND_UNAVAILABLE,
        result=result,
        plan=build_plan_for_capability(CAPABILITY_PATHFIND, suites=(suite_name,)),
    )


def run_tool_loop(
    *,
    question: str,
    suite: SuiteTools,
    suite_name: str,
    llm_client: LLMClient,
    kind: str | None = None,
) -> AgentLoopOutcome:
    if classify_capability(question) == CAPABILITY_PATHFIND:
        return _unimplemented_pathfind(suite_name)

    measured = MeasuredSuite(suite)
    messages: list[Message] = [
        Message(role="system", content=LOOP_SYSTEM),
        Message(role="user", content=question if kind is None else f"{question}\nkind={kind}"),
    ]
    stages: list[StageUsage] = []
    last_result: AgentResult | None = None
    located: AgentResult | None = None
    answer = ""
    searched: set[tuple[str, str | None]] = set()
    intent = classify_capability(question)
    stop_tools = False

    for _round_index in range(MAX_TOOL_ROUNDS):
        # JSON actions, not OpenAI tools: Nvidia/OpenRouter free models reject
        # native tool schemas with 400 "missing field function".
        llm_result, stage = measure_complete(llm_client, messages, stage="act")
        invocations, final = parse_loop_turn(llm_result.text)
        if invocations:
            stage = stage.model_copy(update={"stage": "act"})
        else:
            stage = stage.model_copy(update={"stage": "answer"})
        stages.append(stage)

        if not invocations:
            answer = final or ""
            break

        messages.append(Message(role="assistant", content=llm_result.text))
        for invocation in invocations:
            try:
                if invocation.name == "search_nodes":
                    raw = invocation.arguments.get("text")
                    kind_arg = invocation.arguments.get("kind")
                    kind_value = kind_arg if isinstance(kind_arg, str) else None
                    text = (
                        _search_text_for(
                            question,
                            raw if isinstance(raw, str) else "",
                            already_searched=bool(searched),
                        )
                        if isinstance(raw, str)
                        else extract_locate_subject(question)
                    )
                    key = (text.casefold(), kind_value)
                    if key in searched and located is not None:
                        tool_result = located
                    else:
                        tool_result = _execute_tool(
                            invocation,
                            suite=measured,
                            suite_name=suite_name,
                            located=located,
                            question=question,
                            already_searched=bool(searched),
                        )
                        searched.add(key)
                else:
                    tool_result = _execute_tool(
                        invocation,
                        suite=measured,
                        suite_name=suite_name,
                        located=located,
                        question=question,
                        already_searched=bool(searched),
                    )
                if tool_result.capability == CAPABILITY_LOCATE:
                    located = tool_result
                    if is_terminal_locate(tool_result):
                        stop_tools = True
                last_result = tool_result
                payload: object = _compact_result(tool_result)
            except (ValueError, KeyError) as exc:
                payload = {"error": str(exc)}
            messages.append(Message(role="user", content="TOOL_RESULT " + json.dumps(payload)))
            if stop_tools:
                break
        if stop_tools:
            answer = ""
            break
    else:
        answer = ""

    if (
        intent == CAPABILITY_CONNECT
        and located is not None
        and _is_unique_locate(located)
        and (last_result is None or last_result.capability != CAPABILITY_CONNECT)
    ):
        last_result = connect(
            measured,
            suite_name,
            located.nodes[0],
            request=ConnectRequest(
                subject=located.nodes[0].pref_label,
                rel_types=("HAS_SKILL",),
                relation_kind="essential",
            ),
            confidence=located.confidence,
            locate_evidence=located.evidence,
        )
        answer = summarize_result(last_result)

    if last_result is None:
        last_result = AgentResult(
            capability=CAPABILITY_LOCATE,
            suite=suite_name,
            warnings=["not_found"],
        )

    if intent == CAPABILITY_CONNECT and is_terminal_locate(last_result):
        last_result = last_result.model_copy(update={"capability": CAPABILITY_CONNECT})

    if is_unimplemented_pathfind(last_result) or is_terminal_locate(last_result):
        answer = summarize_result(last_result)
    elif (
        (last_result.capability == CAPABILITY_CONNECT and len(last_result.edges) > 5)
        or not answer
        or looks_like_tool_markup(answer)
    ):
        answer = _structured_fallback(last_result)

    capability = _capability_from_result(last_result)
    if intent == CAPABILITY_CONNECT and capability == CAPABILITY_LOCATE:
        capability = CAPABILITY_CONNECT
        last_result = last_result.model_copy(update={"capability": capability})
    else:
        last_result = last_result.model_copy(update={"capability": capability})
    return AgentLoopOutcome(
        answer=answer,
        result=last_result,
        plan=build_plan_for_capability(capability, suites=(suite_name,)),
        stages=stages,
        tool_calls=measured.tool_calls,
    )
