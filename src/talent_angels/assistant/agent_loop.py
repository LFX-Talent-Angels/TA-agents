"""Main-assistant tool loop: the model chooses Locate/Connect suite tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from talent_angels.assistant.intent import CAPABILITY_CONNECT, CAPABILITY_LOCATE, Capability
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

LOOP_SYSTEM = """You are the Talent Angels main assistant.
Return ONLY a JSON object each turn. Do not invent node IDs or skills.

Call a graph tool:
{"tool":"search_nodes","text":"software developer","kind":"occupation"}
{"tool":"get_neighbors","node_id":"esco:occupation:...","rel_types":["HAS_SKILL"],"relation_filter":"essential"}

Or finish with the user-facing answer:
{"final":"one sentence using only returned labels, ids, and confidence"}

Skills questions: search_nodes first, then get_neighbors with HAS_SKILL.
If search is ambiguous, return final and ask the user to choose.
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
    return {
        "capability": result.capability,
        "confidence": result.confidence,
        "warnings": result.warnings,
        "nodes": [
            {"id": node.id, "kind": node.kind, "pref_label": node.pref_label}
            for node in result.nodes[:20]
        ],
        "edges": [
            {
                "type": edge.type,
                "from": edge.source_node_id,
                "to": edge.target_node_id,
                "properties": edge.properties,
            }
            for edge in result.edges[:30]
        ],
    }


def _execute_tool(
    invocation: ToolInvocation,
    *,
    suite: MeasuredSuite,
    suite_name: str,
    located: AgentResult | None,
) -> AgentResult:
    args = invocation.arguments
    if invocation.name == "search_nodes":
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("search_nodes requires text")
        kind = args.get("kind")
        kind_value = kind if isinstance(kind, str) else None
        return locate(suite, suite_name, text.strip(), kind=kind_value)

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
    if not result.nodes:
        warning = result.warnings[0] if result.warnings else "not_found"
        return f"No match found for capability '{result.capability}' ({warning})."
    confidence = f"{result.confidence:.0%}" if result.confidence is not None else "unknown"
    if result.capability == CAPABILITY_CONNECT and len(result.nodes) > 1:
        neighbors = ", ".join(node.pref_label for node in result.nodes[1:6])
        extra = len(result.nodes) - 6
        more = f" (+{extra} more)" if extra > 0 else ""
        return (
            f"{result.nodes[0].pref_label} — confidence {confidence}; "
            f"{len(result.nodes) - 1} direct connection(s): {neighbors}{more}"
        )
    top = result.nodes[0]
    return f"{top.pref_label} ({top.kind}, id={top.id}) — confidence {confidence}"


def _capability_from_result(result: AgentResult | None) -> Capability:
    if result is not None and result.capability == CAPABILITY_CONNECT:
        return CAPABILITY_CONNECT
    return CAPABILITY_LOCATE


def run_tool_loop(
    *,
    question: str,
    suite: SuiteTools,
    suite_name: str,
    llm_client: LLMClient,
    kind: str | None = None,
) -> AgentLoopOutcome:
    measured = MeasuredSuite(suite)
    messages: list[Message] = [
        Message(role="system", content=LOOP_SYSTEM),
        Message(role="user", content=question if kind is None else f"{question}\nkind={kind}"),
    ]
    stages: list[StageUsage] = []
    last_result: AgentResult | None = None
    located: AgentResult | None = None
    answer = ""

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
                tool_result = _execute_tool(
                    invocation,
                    suite=measured,
                    suite_name=suite_name,
                    located=located,
                )
                if tool_result.capability == CAPABILITY_LOCATE:
                    located = tool_result
                last_result = tool_result
                payload: object = _compact_result(tool_result)
            except (ValueError, KeyError) as exc:
                payload = {"error": str(exc)}
            messages.append(Message(role="user", content="TOOL_RESULT " + json.dumps(payload)))
    else:
        answer = ""

    if last_result is None:
        last_result = AgentResult(
            capability=CAPABILITY_LOCATE,
            suite=suite_name,
            warnings=["not_found"],
        )
    if not answer or looks_like_tool_markup(answer):
        answer = _structured_fallback(last_result)

    capability = _capability_from_result(last_result)
    last_result = last_result.model_copy(update={"capability": capability})
    return AgentLoopOutcome(
        answer=answer,
        result=last_result,
        plan=build_plan_for_capability(capability, suites=(suite_name,)),
        stages=stages,
        tool_calls=measured.tool_calls,
    )
