"""Model phrasing keeps to sentences; the lists and record lines are code's job."""

from __future__ import annotations

from talent_angels.assistant.prose import prose_only

# Replies seen live on 2026-10-07: a skills table above the code-printed list,
# a placeholder table, and record fields the code already prints.
_TABLE_REPLY = """Chefs need creative and operational skills, such as **plan menus**.

---

| Skill | Tag |
|---|---|
| plan menus | essential |
| manage staff | essential |"""

_PLACEHOLDER_REPLY = """A data analyst cleans and models data.

| Skill | Type |
|-------|------|
| *(skills table prints here)* | |

---
**Title:** Data Analyst
**Confidence:** 95%"""

_FIELD_REPLY = """Software developers build software systems.

`occupation_id: software_developer | confidence: —`"""

_SOURCES_REPLY = """Two separate official records cover this work.

```
Sources used:
- ESCO: chef
- O*NET: Chefs and Head Cooks
```"""


def test_drops_a_skills_table() -> None:
    assert prose_only(_TABLE_REPLY) == (
        "Chefs need creative and operational skills, such as **plan menus**."
    )


def test_drops_placeholder_table_and_record_fields() -> None:
    assert prose_only(_PLACEHOLDER_REPLY) == "A data analyst cleans and models data."


def test_drops_a_backticked_field_line() -> None:
    assert prose_only(_FIELD_REPLY) == "Software developers build software systems."


def test_drops_a_code_block() -> None:
    assert prose_only(_SOURCES_REPLY) == "Two separate official records cover this work."


def test_drops_three_or_more_bullets_but_keeps_a_short_mention() -> None:
    assert prose_only("Includes:\n- a\n- b\n- c\nAsk for more.") == "Includes:\nAsk for more."
    assert prose_only("Includes:\n- a\n- b") == "Includes:\n- a\n- b"


def test_plain_prose_is_unchanged() -> None:
    text = "Plumbers install pipes.\n\nThey follow safety rules - always."
    assert prose_only(text) == text
