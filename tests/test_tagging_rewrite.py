"""`rewrite.rewrite_group`: id mapping, fallback, split-retry — and the runner's
cache/skip wrapping around it (`_rewrite_phase`, `_needs_rewrite`)."""

from __future__ import annotations

from pathlib import Path

from talent_angels.llm.protocol import LLMError, LLMResult, LLMUsage, Message
from talent_angels.tagging.keys import REWRITE_VERSION
from talent_angels.tagging.ledger import Ledger
from talent_angels.tagging.models import TagUnit
from talent_angels.tagging.rewrite import Rewrite, rewrite_cache_key, rewrite_group
from talent_angels.tagging.runner import _needs_rewrite, _rewrite_phase


class ScriptedLLMClient:
    provider = "litellm"
    model = "test-model"

    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(
            text=str(reply),
            provider=self.provider,
            model=self.model,
            usage=LLMUsage(input_tokens=5, output_tokens=5),
        )


def _unit(unit_key: str, text: str, **overrides: object) -> TagUnit:
    base: dict[str, object] = {
        "unit_key": unit_key,
        "doc_key": "doc",
        "row_id": unit_key,
        "kind": "performance",
        "role": "leaf",
        "text": text,
        "clean_text": text,
        "context_path": [],
        "group_key": "g1",
        "seed_phrases": [],
    }
    base.update(overrides)
    return TagUnit(**base)  # type: ignore[arg-type]


def test_rewrite_group_maps_local_ids_to_units_in_order() -> None:
    units = [_unit("k1", "define criteria"), _unit("k2", "run tests")]
    client = ScriptedLLMClient(
        [
            '{"rows": {"r1": {"labels": ["define criteria"], "description": "defines it"}, '
            '"r2": {"labels": ["run tests"], "description": "runs it"}}}'
        ]
    )
    rewrites, stages = rewrite_group(client, units, [])
    assert rewrites["k1"].labels == ["define criteria"]
    assert rewrites["k2"].labels == ["run tests"]
    assert len(stages) == 1
    assert len(client.calls) == 1


def test_missing_id_falls_back_to_empty_rewrite() -> None:
    units = [_unit("k1", "define criteria"), _unit("k2", "run tests")]
    client = ScriptedLLMClient(
        ['{"rows": {"r1": {"labels": ["define criteria"], "description": "defines it"}}}']
    )
    rewrites, _stages = rewrite_group(client, units, [])
    assert rewrites["k1"].labels == ["define criteria"]
    assert rewrites["k2"] == Rewrite()


def test_extra_ids_in_reply_are_ignored() -> None:
    units = [_unit("k1", "define criteria")]
    client = ScriptedLLMClient(
        [
            '{"rows": {"r1": {"labels": ["define criteria"], "description": "d"}, '
            '"r2": {"labels": ["bogus"], "description": "d"}}}'
        ]
    )
    rewrites, _stages = rewrite_group(client, units, [])
    assert list(rewrites) == ["k1"]


def test_bad_json_splits_group_and_retries_each_half_once() -> None:
    units = [_unit("k1", "define criteria"), _unit("k2", "run tests")]
    client = ScriptedLLMClient(
        [
            "not json at all",  # first whole-group attempt fails
            '{"rows": {"r1": {"labels": ["define criteria"], "description": "d"}}}',  # k1 half
            '{"rows": {"r1": {"labels": ["run tests"], "description": "d"}}}',  # k2 half
        ]
    )
    rewrites, stages = rewrite_group(client, units, [])
    assert rewrites["k1"].labels == ["define criteria"]
    assert rewrites["k2"].labels == ["run tests"]
    assert len(client.calls) == 3
    assert len(stages) == 3  # the failed whole-group call was billed too


def test_llm_error_with_single_unit_falls_back_without_split() -> None:
    units = [_unit("k1", "define criteria")]
    client = ScriptedLLMClient([LLMError("boom")])
    rewrites, stages = rewrite_group(client, units, [])
    assert rewrites["k1"] == Rewrite()
    assert stages == []
    assert len(client.calls) == 1


def test_bad_json_still_failing_after_split_falls_back_per_unit() -> None:
    units = [_unit("k1", "define criteria"), _unit("k2", "run tests")]
    client = ScriptedLLMClient(["garbage", "garbage again", "garbage once more"])
    rewrites, _stages = rewrite_group(client, units, [])
    assert rewrites["k1"] == Rewrite()
    assert rewrites["k2"] == Rewrite()
    assert len(client.calls) == 3


def test_needs_rewrite_skips_occupation_row_rules_with_seed_phrases() -> None:
    unit = _unit(
        "k1",
        "Associated Occupations: Prompt Architect; Prompt Engineer",
        role="row_rule",
        seed_phrases=["Prompt Architect", "Prompt Engineer"],
    )
    assert _needs_rewrite(unit) is False


def test_needs_rewrite_true_for_a_plain_leaf() -> None:
    assert _needs_rewrite(_unit("k1", "define criteria")) is True


def test_rewrite_phase_skips_seed_phrase_rows_entirely(tmp_path: Path) -> None:
    occupation_unit = _unit(
        "k1",
        "Associated Occupations: Prompt Architect",
        role="row_rule",
        seed_phrases=["Prompt Architect"],
    )
    leaf_unit = _unit("k2", "define criteria")
    client = ScriptedLLMClient(
        ['{"rows": {"r1": {"labels": ["define criteria"], "description": "d"}}}']
    )
    ledger = Ledger(tmp_path / ".ta-tag")

    rewrites, stages = _rewrite_phase(
        client, [occupation_unit, leaf_unit], [], ledger, "test-model"
    )

    assert "k1" not in rewrites
    assert rewrites["k2"].labels == ["define criteria"]
    assert len(stages) == 1


def test_rewrite_phase_cache_hit_makes_zero_model_calls(tmp_path: Path) -> None:
    unit = _unit("k1", "define criteria")
    ledger = Ledger(tmp_path / ".ta-tag")
    key = rewrite_cache_key(unit, "test-model", REWRITE_VERSION)
    ledger.rewrite_append(key, Rewrite(labels=["cached label"], description="cached"))

    client = ScriptedLLMClient([])  # any call would raise IndexError
    rewrites, stages = _rewrite_phase(client, [unit], [], ledger, "test-model")

    assert rewrites["k1"].labels == ["cached label"]
    assert stages == []
    assert client.calls == []
