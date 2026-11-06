"""`keys.py`: identity stable across re-export, fingerprints sensitive to inputs."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import pytest

from talent_angels.tagging.keys import fingerprint, taxonomies_commit, unit_key, versions
from talent_angels.tagging.models import TagUnit, Versions
from talent_angels.tagging.readers import read_document
from talent_angels.tagging.select import select_units

FIXTURE = Path(__file__).parent / "fixtures" / "tagging" / "standard_small.json"


def test_unit_key_stable_across_identity_renumbering(tmp_path: Path) -> None:
    from talent_angels.tagging.profile import load_profile

    original = json.loads(FIXTURE.read_text())
    # Renumber every identity by adding a large constant offset, as a re-export
    # from Neo4j would (new `identity` values, same structure/text).
    offset = 100_000

    def bump(node: dict) -> dict:
        new = dict(node)
        new["identity"] = node["identity"] + offset
        return new

    renumbered = []
    for row in original:
        new_row = dict(row)
        new_row["n"] = bump(row["n"])
        new_row["m"] = bump(row["m"])
        new_row["r"] = {
            **row["r"],
            "start": row["r"]["start"] + offset,
            "end": row["r"]["end"] + offset,
        }
        renumbered.append(new_row)

    renumbered_path = tmp_path / "renumbered.json"
    renumbered_path.write_text(json.dumps(renumbered))

    profile = load_profile()
    doc_a = read_document(FIXTURE)
    doc_b = read_document(renumbered_path)
    units_a, _ = select_units(doc_a, profile)
    units_b, _ = select_units(doc_b, profile)

    keys_a = {unit.unit_key for unit in units_a}
    keys_b = {unit.unit_key for unit in units_b}
    assert keys_a == keys_b
    assert len(keys_a) == len(units_a)  # every key distinct


def test_unit_key_differs_for_identical_text_siblings() -> None:
    from talent_angels.tagging.profile import load_profile

    profile = load_profile()
    doc = read_document(FIXTURE)
    units, _ = select_units(doc, profile)
    duplicates = [u for u in units if u.clean_text == "Clean and sanitized work area"]
    assert len(duplicates) == 2
    assert duplicates[0].unit_key != duplicates[1].unit_key


def test_unit_key_depends_on_occurrence_index() -> None:
    k0 = unit_key("doc", ["a", "b"], "same text", 0)
    k1 = unit_key("doc", ["a", "b"], "same text", 1)
    assert k0 != k1
    assert k0.startswith("k_")
    assert len(k0) == len("k_") + 16


def _unit(**overrides: object) -> TagUnit:
    base = dict(
        unit_key="k_abc",
        doc_key="doc",
        row_id="1",
        kind="performance",
        role="leaf",
        text="raw",
        clean_text="clean",
        context_path=["a", "b"],
        group_key="parent",
    )
    base.update(overrides)
    return TagUnit.model_validate(base)


def _versions(**overrides: object) -> Versions:
    base = dict(
        pipeline="tag-1", policy="tag-v1", prompts="rw-1/ch-1", ta_taxonomies="unknown", model="m"
    )
    base.update(overrides)
    return Versions.model_validate(base)


def test_fingerprint_format() -> None:
    fp = fingerprint(_unit(), "esco", _versions())
    assert fp.startswith("f_")
    assert len(fp) == len("f_") + 24


def test_fingerprint_changes_with_hint() -> None:
    unit = _unit()
    v = _versions()
    assert fingerprint(unit, "esco", v, hint="try again") != fingerprint(unit, "esco", v)


def test_fingerprint_changes_with_attempt() -> None:
    unit = _unit()
    v = _versions()
    assert fingerprint(unit, "esco", v, attempt=1) != fingerprint(unit, "esco", v, attempt=0)


def test_fingerprint_changes_with_suite() -> None:
    unit = _unit()
    v = _versions()
    assert fingerprint(unit, "esco", v) != fingerprint(unit, "onet", v)


def test_fingerprint_changes_with_version_fields() -> None:
    unit = _unit()
    assert fingerprint(unit, "esco", _versions()) != fingerprint(
        unit, "esco", _versions(model="other-model")
    )
    assert fingerprint(unit, "esco", _versions()) != fingerprint(
        unit, "esco", _versions(policy="tag-v2")
    )


def test_taxonomies_commit_returns_a_string() -> None:
    assert isinstance(taxonomies_commit(), str)


def test_taxonomies_commit_falls_back_to_unknown_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_not_found(name: str) -> None:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", raise_not_found)
    assert taxonomies_commit() == "unknown"


def test_versions_fills_pipeline_policy_prompts() -> None:
    v = versions("claude-sonnet-5")
    assert v.pipeline == "tag-1"
    assert v.policy == "tag-v1"
    assert v.prompts == "rw-1/ch-1"
    assert v.model == "claude-sonnet-5"
    assert isinstance(v.ta_taxonomies, str)
