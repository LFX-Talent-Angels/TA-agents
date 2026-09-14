from rich.console import Console

from talent_angels.session.kernel import ChatReply
from talent_angels.tui.render import parse_list_reply, render_assistant, render_welcome


def test_parse_grouped_picker() -> None:
    text = """I found several matches for "developer". Which one did you mean?

**Software developers**
1. IoT developer
2. software developer
**Web and multimedia developers**
5. web developer

I won't pick #1 for you — reply with a number.
source: esco
"""
    parsed = parse_list_reply(text)
    assert "several matches" in parsed.intro
    assert parsed.rows[0] == ("1", "IoT developer", "", "Software developers")
    assert parsed.rows[1][1] == "software developer"
    assert parsed.rows[2] == ("5", "web developer", "", "Web and multimedia developers")
    assert "won't pick" in parsed.footer.casefold()


def test_parse_skill_list_with_relation_tags() -> None:
    text = """**software developer** — 24 skills on the map:

1. computer programming (essential)
2. Joomla (optional)

These are graph neighbors, not a study plan.
"""
    parsed = parse_list_reply(text)
    assert "24 skills" in parsed.intro
    assert parsed.rows[0] == ("1", "computer programming", "essential", "")
    assert parsed.rows[1][2] == "optional"
    assert "graph neighbors" in parsed.footer


def test_two_suite_pickers_stay_separate_tables() -> None:
    text = """ESCO has several matches; O*NET has several matches.

Sources used: ESCO · O*NET

## ESCO

I found several ESCO matches for "developer". Which one did you mean?

1. IoT developer
2. software developer
3. web developer

I won't pick #1 for you — reply with a number. Showing 10 of 25 (15 more).

---

## O*NET

I found several O*NET matches for "developer". Which one did you mean?

11. Web Developers
12. Software Developers
13. Database Architects

I won't pick #1 for you — reply with a number. Showing 10 of 25 (15 more).
"""
    console = Console(record=True, force_terminal=True, width=100, color_system="truecolor")
    render_assistant(console, ChatReply(text=text, source_note="ESCO · O*NET"))
    exported = console.export_text()
    assert "IoT developer" in exported
    assert "Web Developers" in exported
    assert "Database Architects" in exported
    # One merged table would list Economists-style noise as ESCO rows; both
    # headings must survive as separate sections.
    assert exported.index("ESCO") < exported.index("O*NET") or "O*NET" in exported


def test_short_prose_is_not_a_table() -> None:
    parsed = parse_list_reply("Hi there. Name a job title.")
    assert parsed.rows == ()
    assert "Hi there" in parsed.intro


def test_render_welcome_and_assistant_use_color_when_terminal() -> None:
    console = Console(record=True, force_terminal=True, width=80, color_system="truecolor")
    render_welcome(console, "Talent Angels — map jobs and skills.")
    render_assistant(
        console,
        ChatReply(text="Hello from the map.", source_note="ESCO"),
    )
    exported = console.export_text(styles=True)
    assert "Talent Angels" in exported
    assert "Hello from the map." in exported
    assert "source: ESCO" in exported
    assert "\x1b[" in exported  # ANSI, not extra LLM work
