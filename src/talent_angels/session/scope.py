"""Out-of-scope refusals and BLS-only pay questions, checked in code before routing."""

from __future__ import annotations

import re

from talent_angels.assistant.llm_plan import strip_suite_suffix
from talent_angels.session.copy import SCOPE_REFUSE

_A = r"(?:(?:an?|the)\s+)?"
_END = r"\s*[?.!]*\s*$"

# A job title ("curriculum developer"), or the user's own list of one, is not a request to write it.
_NOT_A_REQUEST = (
    r"(?!\s+(?:manager|designer|director|instructor|superintendent|attendant|keeper|developer"
    r"|planner|ranger|specialist|coordinator|consultant|list|file|data|map)s?\b)"
)
_PLAN_DOC = r"\b(?:syllab(?:us|i)|curricul(?:um|a)|learning\s+outcomes?|lesson\s+plans?)\b"
_WRITE = (
    r"\b(?:write|create|design|build|draft|make|generate|develop|prepare|propose|recommend"
    r"|suggest|list|give\s+me|i\s+need|i\s+want)\b"
)
# "make up" is a request only where an instruction starts ("skills that make up a job" is not).
_ASK = r"(?:^|[.!?;:]\s*|\b(?:please|can\s+you|could\s+you|you\s+to|just|and|then)\s+)"

_SCOPE = re.compile(
    rf"{_WRITE}[^.?!]*{_PLAN_DOC}{_NOT_A_REQUEST}"
    rf"|\b(?:what|which)\s+(?:online\s+)?courses?\b{_NOT_A_REQUEST}"
    rf"|\b(?:recommend|suggest|write|create|design|build|draft)\b[^.?!]*\bcourses?\b{_NOT_A_REQUEST}"
    rf"|{_ASK}(?:invent|make\s+up|fabricate|hallucinate|pretend)\b"
    r"|\bmade[\s-]up\s+(?:skills?|jobs?|titles?|list|data|answer)"
    r"|\bignore\b[^.?!]*\b(?:taxonom\w*|instructions?|rules?|graph|sources?)\b",
    re.I,
)

_SALARY = (
    re.compile(
        # "pay" the noun only: "how do I pay for training" is not a wage question.
        r"\b(?:salary|salaries|wages?|earnings?"
        r"|(?<!\bi )(?<!\bwe )(?<!\bto )(?<!\byou )(?<!\bthey )pay)"
        r"(?:\s+(?:range|level|scale|rate|data|figures?|and\s+outlook))?"
        rf"\s+(?:of|for)\s+{_A}(.+?){_END}",
        re.I,
    ),
    re.compile(
        rf"\bhow\s+much\s+(?:money\s+)?(?:does|do|will|would|can|could)\s+{_A}(.+?)\s+"
        r"(?:make|earn|get\s+paid)\b",
        re.I,
    ),
)
_PRONOUNS = {"i", "you", "we", "they", "he", "she", "me", "one", "someone", "people"}


def out_of_scope(text: str) -> str | None:
    """Return a fixed code refusal for out-of-scope requests, else None."""
    return SCOPE_REFUSE if _SCOPE.search(text) else None


def salary_subject(text: str) -> str | None:
    """None: not a pay question. "": pay question with no job named. Else the job."""
    for rx in _SALARY:
        if match := rx.search(text):
            subject = strip_suite_suffix(match.group(1)).strip(" ?.!\"'")
            ok = subject.casefold() not in _PRONOUNS and 1 <= len(subject.split()) <= 6
            return subject if ok else ""
    return None
