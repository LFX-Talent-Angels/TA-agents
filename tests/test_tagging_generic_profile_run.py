"""Runner e2e for the generic (business-agnostic) profile over a table input.

Unlike `test_tagging_run_model.py` (CONOCER's `competency_standard` profile,
a Neo4j tree), this drives a flat CSV through `--profile generic` with three
fake suites attached, to prove the profile/reader split is real: every row
tags as `item`, the document sheet is named "Document", and the rewrite
system prompt is built from the *profile's* own wording ("lines of a
document") and lists every attached suite's display name — including a
third suite nothing else in the test fixtures uses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
import pytest

from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.suites.schema import SuiteSchema
from talent_angels.tagging.models import RunOptions
from talent_angels.tagging.runner import run
from tests.fakes.tagging_suite import FakeCandidate, FakeNode, FakeToolResult, table_registry

_SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL",),
    optional_rel_values=frozenset(),
    group_rel_type=None,
    group_node_kinds=frozenset(),
)
_HIT_KINDS = frozenset({"skill", "task", "knowledge"})


@dataclass
class AnyKindSuite:
    """Answers every query with one fixed hit, for every kind `item` routes to."""

    name: str
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    @property
    def suite_schema(self) -> SuiteSchema:
        return _SCHEMA

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.calls.append((text, kind))
        if kind is None or kind not in _HIT_KINDS:
            return FakeToolResult(warnings=[f"unknown_kind:{kind}"])
        node = FakeNode(id=f"{self.name}:{kind}:fixed", kind="Skill", label=text)
        return FakeToolResult(
            candidates=[FakeCandidate(node=node, confidence=0.9, method="exact_pref")]
        )

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["node_not_found"])


class _ExactEveryTimeClient:
    """Scripted client: rewrite calls get a trivial phrase, choose calls always pick #1."""

    provider = "litellm"
    model = "test-model"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        self.calls.append(messages)
        user = messages[1].content
        if '<row id="' in user:
            row_ids = re.findall(r'<row id="(r\d+)"', user)
            rows = {
                rid: {"labels": ["do a thing"], "description": "does a thing"} for rid in row_ids
            }
            text = json.dumps({"rows": rows})
        else:
            text = json.dumps({"fit": "exact", "picks": [{"n": 1}], "reason": "matches"})
        return LLMResult(text=text, provider=self.provider, model=self.model, usage=LLMUsage(5, 5))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    (project_dir / "inputs").mkdir(parents=True)
    (project_dir / "inputs" / "items.csv").write_text(
        "text\n"
        "Registers guest arrivals in the property management system\n"
        "Handles complaints with empathy\n"
    )
    return project_dir


def _registry():
    return table_registry(
        {
            "esco": AnyKindSuite("esco"),
            "onet": AnyKindSuite("onet"),
            "extra": AnyKindSuite("extra"),
        },
        default="esco",
    )


def test_generic_profile_tags_every_row_as_item(project: Path) -> None:
    registry = _registry()
    client = _ExactEveryTimeClient()
    summary = run(
        project,
        registry=registry,
        opts=RunOptions(no_llm=False, profile="generic", fresh=True),
        llm_client=client,
    )

    assert summary.units == 2
    [output] = summary.outputs
    workbook = openpyxl.load_workbook(output)
    assert "Document" in workbook.sheetnames
    assert "Standard" not in workbook.sheetnames

    standard_ws = workbook["Document"]
    kind_col = next(i for i, cell in enumerate(standard_ws[2], start=1) if cell.value == "Kind")
    kinds = [
        value for row in standard_ws.iter_rows(min_row=3) if (value := row[kind_col - 1].value)
    ]
    assert kinds == ["item", "item"]


def test_rewrite_system_prompt_uses_profile_wording_and_lists_every_suite(project: Path) -> None:
    registry = _registry()
    client = _ExactEveryTimeClient()
    run(
        project,
        registry=registry,
        opts=RunOptions(no_llm=False, profile="generic", fresh=True),
        llm_client=client,
    )

    rewrite_calls = [c for c in client.calls if '<row id="' in c[1].content]
    assert rewrite_calls, "expected at least one rewrite call"
    system_prompt = rewrite_calls[0][0].content
    assert "lines of a document" in system_prompt
    assert "ESCO" in system_prompt
    assert "O*NET" in system_prompt
    assert "EXTRA" in system_prompt  # the third suite, display_name's upper() fallback
