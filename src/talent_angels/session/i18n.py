"""The sentences code writes, in the user's language.

The model already answers in the user's language when told to; the lead line,
pick lists, areas and fixed messages are written in code and were always
English, so a Spanish question got a half-English reply. The planner reports
the language of each message; the kernel keeps it for the turn
(``set_language``) and every code-written sentence goes through ``t``. English
is the fallback for a language or a key without a translation. Job titles stay
as the taxonomy names them.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

#: Languages with code-written sentences. Others get English sentences and
#: model-written text in their own language.
SUPPORTED = ("en", "es")

_LANG: ContextVar[str] = ContextVar("reply_language", default="en")

_EN: dict[str, str] = {
    # lead line
    "your_request": "your request",
    "read_as_titles": "I read {quoted} as {titles}.",
    "looked_up": 'I looked up "{subject}".',
    "read_as_subject": 'I read {quoted} as "{subject}".',
    "or": "or",
    "and": "and",
    "matching_title": "matching title",
    "matching_titles": "matching titles",
    "close_title": "close title",
    "close_titles": "close titles",
    "area_title": "title",
    "area_titles": "titles",
    "found_list": "{suite} has {count} {noun}",
    "found_close": "{suite} has no exact match; {count} {noun}",
    "found_one": "{suite} matched {title}",
    "found_none": "{suite} has no match",
    "next_goal": "Pick the one you mean and I'll save it as your goal.",
    "next_standing": "Pick the one you mean and I'll save it as your current job.",
    "next_reject": "Pick the one you mean and I'll note that it is not your job.",
    "next_default": "Pick the one you mean, or tell me more and I'll narrow it.",
    # pick lists
    "picker_heading": 'I found several matches for "{question}". Which one did you mean?',
    "picker_empty": 'I found several matches for "{question}", but none to show.',
    "picker_one": "I won't pick for you — reply with {first}.",
    "picker_range": "I won't pick for you — reply with {first}–{last}.",
    "picker_showing": (
        "Showing {shown} of {total} ({omitted} more). The full list is in query details."
    ),
    "intro_guided": "{suite} titles that fit your request. Which one did you mean?",
    "intro_several": 'I found several {suite} matches for "{searched}". Which one did you mean?',
    "areas_head": "**Or narrow it down by area:**",
    "areas_foot": "Reply with a letter, or describe the work you have in mind.",
    "bound": "Bound {title}.",
    "narrow_fit": 'These fit "{text}". Which one did you mean?',
    "narrow_area": '{names}: titles matching "{query}". Which one did you mean?',
    "narrow_none": (
        "None of the titles on the list fit that. Describe it another way, or name a title."
    ),
    # fixed messages
    "map_next": "You can ask for essential skills, optional skills, or pick another number.",
    "miss": "No node for that phrase with today's search. That's a miss, not a maybe.",
    "no_subject": (
        "Which occupation or skill do you mean? For example: "
        "“What skills does a nurse need?” or “Where is data scientist?”"
    ),
    "pathfind": (
        "Learning paths between two roles aren't available yet — they're coming. "
        "For now, try 'compare teacher and data analyst' to see which skills they "
        "share and which only one of them needs."
    ),
    # skill lists and compares
    "skill": "skill",
    "skills": "skills",
    "tool": "tool",
    "tools": "tools",
    "shared_skill": "shared skill",
    "shared_skills_noun": "shared skills",
    "shared_tool": "shared tool",
    "shared_tools_noun": "shared tools",
    "connect_head": "**{title}** — {counts} on the map:",
    "connect_more": "{count} more are in query details.",
    "connect_foot": "These are graph neighbors, not a study plan.",
    "compare_head": "**{first}** vs **{second}** — {shared}.",
    "compare_shared": "Shared",
    "compare_only": "Only {title}",
    "compare_none": "none",
    "compare_more": "(+{count} more)",
    "compare_foot": "These are graph neighbours, not a recommendation.",
    # profile
    "profile_empty": (
        'I don\'t have much about you yet. Tell me your current job ("I am a …") '
        "or a job you're aiming for (\"my goal is …\") and I'll note it."
    ),
    "profile_head": "Here is what you've told me:",
    "profile_foot": "That is all I keep about you. Tell me your goal or current job to change it.",
    "profile_goal": "Goal: {title}",
    "profile_current": "Current job ({suite}): {title}",
    "profile_rejected": "Not your job (you said so): {title}",
    "noted_not_job": "Noted: {titles} is no longer saved as your current job.",
}

_ES: dict[str, str] = {
    "your_request": "tu solicitud",
    "read_as_titles": "Entendí {quoted} como {titles}.",
    "looked_up": 'Busqué "{subject}".',
    "read_as_subject": 'Entendí {quoted} como "{subject}".',
    "or": "o",
    "and": "y",
    "matching_title": "título que coincide",
    "matching_titles": "títulos que coinciden",
    "close_title": "título cercano",
    "close_titles": "títulos cercanos",
    "area_title": "título",
    "area_titles": "títulos",
    "found_list": "{suite} tiene {count} {noun}",
    "found_close": "{suite} no tiene una coincidencia exacta; {count} {noun}",
    "found_one": "{suite} encontró {title}",
    "found_none": "{suite} no tiene coincidencias",
    "next_goal": "Elige el que buscas y lo guardaré como tu meta.",
    "next_standing": "Elige el que buscas y lo guardaré como tu trabajo actual.",
    "next_reject": "Elige el que buscas y anotaré que no es tu trabajo.",
    "next_default": "Elige el que buscas, o cuéntame más y acotaré la lista.",
    "picker_heading": 'Encontré varias coincidencias para "{question}". ¿Cuál buscas?',
    "picker_empty": 'Encontré varias coincidencias para "{question}", pero ninguna para mostrar.',
    "picker_one": "No elegiré por ti: responde con {first}.",
    "picker_range": "No elegiré por ti: responde con un número del {first} al {last}.",
    "picker_showing": (
        "Mostrando {shown} de {total} ({omitted} más). "
        "La lista completa está en los detalles de la consulta."
    ),
    "intro_guided": "Títulos de {suite} que se ajustan a tu solicitud. ¿Cuál buscas?",
    "intro_several": 'Encontré varias coincidencias en {suite} para "{searched}". ¿Cuál buscas?',
    "areas_head": "**O acota por área:**",
    "areas_foot": "Responde con una letra o describe el trabajo que tienes en mente.",
    "bound": "Seleccionado: {title}.",
    "narrow_fit": 'Estos se ajustan a "{text}". ¿Cuál buscas?',
    "narrow_area": '{names}: títulos que coinciden con "{query}". ¿Cuál buscas?',
    "narrow_none": (
        "Ninguno de los títulos de la lista encaja con eso. "
        "Descríbelo de otra forma o nombra un título."
    ),
    "map_next": "Puedes pedir las habilidades esenciales u opcionales, o elegir otro número.",
    "miss": "No encontré nada para esa frase con la búsqueda de hoy. Es un no, no un tal vez.",
    "no_subject": (
        "¿A qué ocupación o habilidad te refieres? Por ejemplo: "
        "“¿Qué habilidades necesita una enfermera?” o “¿Qué es un científico de datos?”"
    ),
    "pathfind": (
        "Las rutas de aprendizaje entre dos puestos aún no están disponibles; pronto lo "
        "estarán. Mientras tanto, prueba 'compara docente y analista de datos' para ver "
        "qué habilidades comparten y cuáles necesita solo uno."
    ),
    "skill": "habilidad",
    "skills": "habilidades",
    "tool": "herramienta",
    "tools": "herramientas",
    "shared_skill": "habilidad en común",
    "shared_skills_noun": "habilidades en común",
    "shared_tool": "herramienta en común",
    "shared_tools_noun": "herramientas en común",
    "connect_head": "**{title}** — {counts} en el mapa:",
    "connect_more": "Hay {count} más en los detalles de la consulta.",
    "connect_foot": "Son vecinos en el grafo, no un plan de estudio.",
    "compare_head": "**{first}** frente a **{second}** — {shared}.",
    "compare_shared": "En común",
    "compare_only": "Solo {title}",
    "compare_none": "ninguna",
    "compare_more": "(+{count} más)",
    "compare_foot": "Son vecinos en el grafo, no una recomendación.",
    "profile_empty": (
        'Aún no sé mucho de ti. Dime tu trabajo actual ("soy …") o el trabajo al '
        'que aspiras ("mi meta es …") y lo anotaré.'
    ),
    "profile_head": "Esto es lo que me has contado:",
    "profile_foot": (
        "Es todo lo que guardo sobre ti. Dime tu meta o tu trabajo actual para cambiarlo."
    ),
    "profile_goal": "Meta: {title}",
    "profile_current": "Trabajo actual ({suite}): {title}",
    "profile_rejected": "No es tu trabajo (lo dijiste tú): {title}",
    "noted_not_job": "Anotado: {titles} ya no está guardado como tu trabajo actual.",
}

_TABLES: dict[str, dict[str, str]] = {"en": _EN, "es": _ES}


def normalize_language(code: object) -> str | None:
    """A two-letter code we write sentences in, or None."""
    if not isinstance(code, str):
        return None
    code = code.strip().casefold()[:2]
    return code if code in SUPPORTED else None


def set_language(code: object) -> None:
    """The language of this turn's code-written sentences (English if unknown)."""
    _LANG.set(normalize_language(code) or "en")


def language() -> str:
    return _LANG.get()


def t(key: str, **values: Any) -> str:
    """The sentence ``key`` in this turn's language, filled with ``values``."""
    table = _TABLES.get(language(), _EN)
    return table.get(key, _EN[key]).format(**values)


def plural(count: int, one: str, many: str) -> str:
    """ "1 title", "3 titles" in this turn's language (keys of the table)."""
    return t(one) if count == 1 else t(many)
