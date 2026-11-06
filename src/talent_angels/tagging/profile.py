"""Business rules as data: how raw headings/rows map to a tagging `kind`.

Kept out of code so CONOCER's "competency standard" shape — and a future
business's different shape — are both just a YAML file; `select.py` never
special-cases a kind by name.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr


class RowRule(BaseModel):
    """A row-text rule that assigns a kind regardless of ancestor headings."""

    pattern: str
    kind: str
    split: str | None = None


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


class Profile(BaseModel):
    """One business's tagging rules (CONOCER's `competency_standard`, ...)."""

    name: str
    version: int
    depth: Literal["leaves", "headings", "all"] = "leaves"
    kinds: dict[str, list[str]] = Field(default_factory=dict)
    row_rules: list[RowRule] = Field(default_factory=list)
    include_kinds: list[str] = Field(default_factory=list)
    report_kinds: list[str] = Field(default_factory=list)
    cleaners: list[str] = Field(default_factory=list)
    routing: dict[str, list[str]] = Field(default_factory=dict)
    doc_context: list[str] = Field(default_factory=list)

    _kind_patterns: list[tuple[str, re.Pattern[str]]] = PrivateAttr(default_factory=list)
    _row_rule_patterns: list[tuple[re.Pattern[str], RowRule]] = PrivateAttr(default_factory=list)
    _cleaner_patterns: list[re.Pattern[str]] = PrivateAttr(default_factory=list)
    _doc_context_patterns: list[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        try:
            self._kind_patterns = [
                (kind, _compile(pattern))
                for kind, patterns in self.kinds.items()
                for pattern in patterns
            ]
            self._row_rule_patterns = [(_compile(rule.pattern), rule) for rule in self.row_rules]
            self._cleaner_patterns = [_compile(pattern) for pattern in self.cleaners]
            self._doc_context_patterns = [_compile(pattern) for pattern in self.doc_context]
        except re.error as exc:
            raise ValueError(f"invalid regex in profile {self.name!r}: {exc}") from exc

    def kind_of_heading(self, text: str) -> str | None:
        """The kind this row's own text names, if its text is a kind heading."""
        stripped = text.strip()
        for kind, pattern in self._kind_patterns:
            if pattern.fullmatch(stripped):
                return kind
        return None

    def rule_for(self, text: str) -> RowRule | None:
        """The row rule matching this row's own text, if any."""
        for pattern, rule in self._row_rule_patterns:
            if pattern.match(text):
                return rule
        return None

    def clean(self, text: str) -> str:
        """`text` with every cleaner applied in order, then whitespace-collapsed."""
        cleaned = text
        for pattern in self._cleaner_patterns:
            cleaned = pattern.sub("", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def matches_doc_context(self, text: str) -> bool:
        return any(pattern.match(text) for pattern in self._doc_context_patterns)


@lru_cache(maxsize=8)
def _builtin_profile_data(name: str) -> dict[str, object]:
    package = resources.files("talent_angels.tagging.profiles")
    text = package.joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{name}.yaml did not parse to a mapping")
    return data


def load_profile(project_dir: Path | None = None, name: str = "competency_standard") -> Profile:
    """Built-in `<name>.yaml` merged with an optional `<project_dir>/tag.yaml`.

    Top-level keys in `tag.yaml` replace the matching built-in key wholesale
    (a project overriding `cleaners` doesn't need to repeat `kinds`).
    """
    data = dict(_builtin_profile_data(name))
    source: Path | str = name
    if project_dir is not None:
        override_path = project_dir / "tag.yaml"
        if override_path.exists():
            source = override_path
            try:
                override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError(f"{override_path}: invalid YAML ({exc})") from exc
            if override is not None:
                if not isinstance(override, dict):
                    raise ValueError(f"{override_path}: expected a mapping at the top level")
                data.update(override)
    try:
        return Profile.model_validate(data)
    except ValueError as exc:
        raise ValueError(f"invalid profile ({source}): {exc}") from exc
