"""User-facing copy for the interactive ta-agent TUI.

LFX Talent Angels voice — no taxonomy product names in identity lines.
"""

GREETING = "Hey. I'm here.\n\nTry a job title, or ask what I can do."

WELCOME = (
    "LFX Talent Angels — map jobs and skills across the attached taxonomies.\n\n"
    "Type a job title to locate it, or /help for commands."
)

HELP_INTRO = (
    "I look up occupations and skills, then list what's connected once we "
    "agree which title you mean. I won't guess a match."
)

COMMANDS_BLOCK = (
    "**Commands**\n"
    "- `/help` — this list\n"
    "- `/model` — pick a model; type to filter, arrows to move\n"
    "- `/connect` — use your own LLM (OpenAI, Anthropic, Gemini, OpenRouter, Azure, a local "
    "Ollama, or any LiteLLM model); `/login` does the same\n"
    "- `/save [name]` — save this session\n"
    "- `/resume [name]` — resume a saved session (default: last)\n"
    "- `/reset` — forget this conversation (your profile and history stay)\n"
    "- `/reset-all` — forget everything: conversation, profile, and history\n"
    "- `/map NAME [--depth N --breadth N --suites a,b --kinds k1,k2]` — map a skill or job; "
    "`/map --route FROM | TO` maps around the best route\n"
    "- `/discover FILE [--depth N --breadth N --suites a,b]` — how a list of roles and "
    "skills connects\n"
    "- `/roadmap FROM | TO` — what to learn to get from one job to another, in order\n"
    "- `/successors SKILL [--depth N --breadth N]` — what comes after a skill\n"
    "- `/tag PATH [--no-llm --limit N --suites a,b]` — tag documents against the taxonomies "
    "(can take minutes)\n"
    "- `/quit` or `/exit` — leave\n"
    "\n"
    "`/map`, `/discover`, `/roadmap`, `/successors` and `/tag` print a short summary and "
    "write the full result as files under your TA home, `outputs/` (JSON, and a web page "
    "to open in a browser); they never call the model (except `/tag` without `--no-llm`).\n\n"
    "Press `esc` while a question or a long command is running to stop it.\n\n"
    "Or just say it: “roadmap from data analyst to data scientist”, “what comes after Python”, "
    "“map Python 2 levels”, “connect these: SQL, Python, statistics”, “tag ./my-folder”."
)
STOPPED = "Stopped. Nothing was written."

HELP_TEXT = (
    f"{HELP_INTRO}\n\n{COMMANDS_BLOCK}\n\n"
    "Type a job title, ask about essential skills, or pick a numbered choice."
)

ADVICE_REFUSE = (
    "I don't give personal career advice. "
    "I can locate a job title and show the skills tied to it — "
    "try naming a role, or ask what skills it needs."
)

CATALOGUE_REFUSE = (
    "I don't list every occupation on the map. "
    "Name a job title and I'll pin it, then we can see the skills tied to it."
)

UNKNOWN_COMMAND = "{token} isn't a command. /help lists the ones I know."

LOCATE_MISS = "No node for that phrase with today's search. That's a miss, not a maybe."

MAP_NEXT_STEP = "You can ask for essential skills, optional skills, or pick another number."

# GAP F: the initial connect answer previews CONNECT_PREVIEW_CAP skills; beyond
# that, the only way to see the rest was an undocumented "say list all skills"
# follow-up. Say the words that actually work.
CONNECT_MORE_HINT = 'Ask "show more skills" to see the rest.'

SCOPE_REFUSE = (
    "I only report what the taxonomies hold: occupations, skills and how they connect. I don't "
    "write courses, syllabi or learning outcomes, I don't invent or guess skills, and I can't set "
    "the taxonomy data aside. Name a job or a skill and I'll show what the map lists, with its "
    "source."
)
SALARY_NOTE = "Pay comes only from what BLS publishes, so this is the BLS record."
SALARY_REFUSE = (
    "I only report pay that BLS publishes on its occupation records. Name the job, "
    "for example: “registered nurse in BLS”."
)
SECRET_REFUSE = (
    "That looks like an API key, so I didn't keep it or send it anywhere. Use "
    "/connect to add your model; its key prompt is hidden."
)
ROUTE_CLARIFY = (
    "I couldn't tell which of these you want: “roadmap from data analyst to data "
    "scientist”, “what comes after Python”, “map Python 2 levels”, “connect these: "
    "SQL, Python, statistics”, or “tag ./my-folder”. Nothing was run."
)
ROADMAP_CLARIFY = (
    "Which two jobs? Say: roadmap from data analyst to data scientist. Nothing was run."
)
SUCCESSORS_CLARIFY = "After which skill? Say: what comes after Python. Nothing was run."
MAP_CLARIFY = "Map what? Say: map Python, or map the route from nurse to midwife. Nothing was run."
NAME_CLARIFY = (
    "That doesn't look like one job or skill name. Give just the name, for example: "
    "map Python. Nothing was run."
)
NO_SUCH_PATH = (
    "I can't find {path}. Give the full path to an existing file or folder. Nothing was run."
)
UNATTACHED = "{names} isn't attached here. Attached: {available}. Nothing was run."
CANT_RUN = "I can't run {cap}: {error}. Nothing was run."
TAG_CONFIRM = (
    "Tagging with your model ({model}) makes about one model call per row for each of {count} "
    "taxonomies ({suites}), across {files} file(s), plus some rewrites. There is no exact estimate "
    "before the run; the cost is reported at the end. Reply **yes** to start. Anything else "
    "cancels; “tag {path}” alone runs free, without the model."
)
TAG_CANCELLED = "Not started. Nothing was tagged and nothing was spent."
TAG_NO_MODEL = "No model is connected, so this runs without one (free). /connect adds yours."
NO_MODEL_NOTICE = (
    "No model connected: every capability still works, deterministically. "
    "/connect adds your own LLM."
)
