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
    #: What this profile calls its input ("competency standard", "document", ...) —
    #: fills the model prompts' "lines of a {document_type}" wording (prompts.py).
    document_type: str
    #: What this profile calls its input on the workbook: the `<context>` prefix in
    #: the rewrite prompt, and the Standard/Document sheet's own name.
    document_label: str
    depth: Literal["leaves", "headings", "all"] = "leaves"
    #: Also tag each heading *as a group*: its own text plus its direct
    #: children's, one request — the group-level answer next to the steps'.
    groups: bool = True
    #: How many times a reviewer may "Retry with hint"/"Reject" one answer.
    #: Each retry adds more of the document's own context (parent, then
    #: grandparent + siblings), so a third rarely finds anything new.
    max_retries: int = Field(default=2, ge=0)
    kinds: dict[str, list[str]] = Field(default_factory=dict)
    #: Graph exports: kind -> regexes searched (case-insensitive) in a row's
    #: relationship to its parent, then in its node type ("PRODUCES_PRODUCT",
    #: "WorkProduct" -> product). Tree and table exports carry no relationship.
    relations: dict[str, list[str]] = Field(default_factory=dict)
    row_rules: list[RowRule] = Field(default_factory=list)
    #: A row with no kind heading above it (and no row rule) gets this kind
    #: instead of being skipped as metadata — `select.py`. `None` keeps the
    #: competency_standard behaviour: no ancestor kind means skip.
    default_kind: str | None = None
    include_kinds: list[str] = Field(default_factory=list)
    #: Kinds whose line is itself a name to look up (a job title): searched by
    #: its own cleaned text, with no model rewrite into skill phrases.
    verbatim_kinds: list[str] = Field(default_factory=list)
    report_kinds: list[str] = Field(default_factory=list)
    cleaners: list[str] = Field(default_factory=list)
    routing: dict[str, list[str]] = Field(default_factory=dict)
    doc_context: list[str] = Field(default_factory=list)

    _kind_patterns: list[tuple[str, re.Pattern[str]]] = PrivateAttr(default_factory=list)
    _relation_patterns: list[tuple[str, re.Pattern[str]]] = PrivateAttr(default_factory=list)
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
            self._relation_patterns = [
                (kind, _compile(pattern))
                for kind, patterns in self.relations.items()
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

    def kind_of_relation(self, relation: str, node_type: str) -> str | None:
        """The kind a graph row's relationship names, else its node type; None if neither.

        Only graph rows carry a relationship. Without one (tree and table
        exports) this is always None: their node labels never decide a kind.
        """
        if not relation:
            return None
        for value in (relation, node_type):
            for kind, pattern in self._relation_patterns:
                if pattern.search(value):
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


_DEFAULT_PROFILE_NAME = "competency_standard"


@lru_cache(maxsize=8)
def _builtin_profile_data(name: str) -> dict[str, object]:
    package = resources.files("talent_angels.tagging.profiles")
    text = package.joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{name}.yaml did not parse to a mapping")
    return data


def builtin_profile_names() -> list[str]:
    """Every `<name>.yaml` shipped under `tagging/profiles/` (no path, no extension)."""
    package = resources.files("talent_angels.tagging.profiles")
    return sorted(
        entry.name[: -len(".yaml")]
        for entry in package.iterdir()
        if entry.is_file() and entry.name.endswith(".yaml")
    )


def _load_base(name: str) -> tuple[dict[str, object], Path | str]:
    """Built-in `<name>.yaml`, or `name` itself when it is a path to a `.yaml` file."""
    maybe_path = Path(name)
    if maybe_path.suffix.lower() == ".yaml" and maybe_path.is_file():
        try:
            data = yaml.safe_load(maybe_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"{maybe_path}: invalid YAML ({exc})") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{maybe_path}: did not parse to a mapping")
        return data, maybe_path
    builtins = builtin_profile_names()
    if name not in builtins:
        available = ", ".join(builtins)
        raise ValueError(f"unknown profile {name!r}; available built-in profiles: {available}")
    return dict(_builtin_profile_data(name)), name


def load_profile(project_dir: Path | None = None, name: str | None = None) -> Profile:
    """A base profile merged with an optional `<project_dir>/tag.yaml`.

    The base profile is, in order: the explicit `name` argument (CLI
    `--profile`, a built-in name or a path to a `.yaml` file) if given; else
    `tag.yaml`'s own `base_profile:` key; else `"competency_standard"`.

    Top-level keys in `tag.yaml` replace the matching built-in key wholesale
    (a project overriding `cleaners` doesn't need to repeat `kinds`).
    """
    override: dict[str, object] | None = None
    override_path: Path | None = None
    if project_dir is not None:
        override_path = project_dir / "tag.yaml"
        if override_path.exists():
            try:
                loaded = yaml.safe_load(override_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError(f"{override_path}: invalid YAML ({exc})") from exc
            if loaded is not None:
                if not isinstance(loaded, dict):
                    raise ValueError(f"{override_path}: expected a mapping at the top level")
                override = loaded

    base_name = name
    if base_name is None and override is not None:
        base_profile = override.get("base_profile")
        if base_profile is not None:
            if not isinstance(base_profile, str):
                raise ValueError(f"{override_path}: 'base_profile' must be a string")
            base_name = base_profile
    if base_name is None:
        base_name = _DEFAULT_PROFILE_NAME

    data, source = _load_base(base_name)
    data = dict(data)
    if override is not None:
        data.update({key: value for key, value in override.items() if key != "base_profile"})
        source = override_path if override_path is not None else source
    try:
        return Profile.model_validate(data)
    except ValueError as exc:
        raise ValueError(f"invalid profile ({source}): {exc}") from exc
