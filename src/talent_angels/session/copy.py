"""User-facing copy for the interactive ta-agent TUI.

Talent Angels voice — no taxonomy product names in identity lines.
"""

GREETING = "Hey. I'm here.\n\nTry a job title, or ask what I can do."

WELCOME = (
    "Talent Angels — map jobs and skills from the bound taxonomy suite.\n\n"
    "Type a job title to locate it, or /help for commands."
)

HELP_INTRO = (
    "I look up occupations and skills, then list what's connected once we "
    "agree which title you mean. I won't guess a match."
)

COMMANDS_BLOCK = (
    "**Commands**\n"
    "- `/help` — this list\n"
    "- `/save [name]` — save this session\n"
    "- `/resume [name]` — resume a saved session (default: last)\n"
    "- `/clear` — clear transcript and binding\n"
    "- `/quit` or `/exit` — leave"
)

HELP_TEXT = (
    f"{HELP_INTRO}\n\n{COMMANDS_BLOCK}\n\n"
    "Type a job title, ask about essential skills, or pick a numbered choice."
)

ADVICE_REFUSE = (
    "I don't give personal career advice. "
    "I can locate a job title and show the skills tied to it — "
    "try naming a role, or ask what skills it needs."
)

UNKNOWN_COMMAND = "{token} isn't a command. /help lists the ones I know."

LOCATE_MISS = "No node for that phrase with today's search. That's a miss, not a maybe."

MAP_NEXT_STEP = "You can ask for essential skills, optional skills, or pick another number."
