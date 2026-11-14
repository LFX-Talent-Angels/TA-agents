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

import re
from contextvars import ContextVar
from typing import Any

#: Languages with code-written sentences. Others get English sentences and
#: model-written text in their own language.
SUPPORTED = ("en", "es", "sv")

_LANG: ContextVar[str] = ContextVar("reply_language", default="en")
#: "sv", "es-MX", "sv_SE".
_LANGUAGE_CODE = re.compile(r"([a-z]{2})(?:[-_][a-z0-9]{2,8})?")

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
    "map_record": "Map record: **{title}** ({kind}{confidence})",
    "map_confidence": ", confidence {percent}",
    "map_next": "You can ask for essential skills, optional skills, or pick another number.",
    "miss": "No node for that phrase with today's search. That's a miss, not a maybe.",
    "no_published_facts": (
        "{suite} publishes no pay, outlook or education figures for this occupation."
    ),
    "no_subject": (
        "Which occupation or skill do you mean? For example: "
        "“What skills does a nurse need?” or “Where is data scientist?”"
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
    "transitions_head": "**{title}** — {count} jobs to move to ({suite}):",
    "related_head": "**{title}** — {count} related occupations ({suite}):",
    "long_input": (
        "Your message is long, so I read its start and its end ({limit} characters in all)."
    ),
    "filtered_count": "{count} {kind} (of {total} in all)",
    "none_marked_count": "none marked {kind}; all {counts}",
    "transitions_foot": (
        "The percentage is how far the taxonomy says the experience carries over. "
        "This is the map, not career advice."
    ),
    "no_transitions": "{suite} publishes no job-to-job moves for **{title}**.",
    "connect_foot_moves": (
        "Related occupations as the taxonomy lists them. This is the map, not career advice."
    ),
    "connect_more": "{count} more are in query details.",
    "connect_foot": "These are graph neighbors, not a study plan.",
    "compare_head": "**{first}** vs **{second}** — {shared}.",
    "compare_shared": "Shared",
    "compare_only": "Only {title}",
    "compare_none": "none",
    "compare_more": "(+{count} more)",
    "compare_foot": "These are graph neighbours, not a recommendation.",
    "pathfind_head": "**From {start} to {target}:**",
    "pathfind_essential": "Essential skills needed",
    "pathfind_optional": "Optional skills needed",
    "pathfind_software": "Software/tools needed",
    "pathfind_gap_none": "none",
    "pathfind_more": "(+{count} more)",
    "pathfind_full_retrain": (
        "This gap is large enough to be a full retrain, not a quick transition."
    ),
    "pathfind_route": "Route {number}: {chain}",
    "pathfind_more_routes": "(+{count} more routes)",
    "pathfind_no_route": "No route found within {steps} steps. The gap above still stands.",
    "pathfind_no_facts": "No published pay, outlook or education figures for this occupation.",
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
    "compare_needs_pick": (
        "To compare {first} and {second}, I first need to know which {subject} you mean."
    ),
    "comparing": "Comparing {first} with {second}:",
    "still_waiting": "{suite} still needs a pick to compare there too:",
    "bound_many": "Bound {titles}.",
    "one_per_suite": "Pick at most one title per taxonomy, e.g. {example}.",
    "pick_per_suite": "You can pick one title from each list in one reply, e.g. {example}.",
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
    "map_record": "Registro del mapa: **{title}** ({kind}{confidence})",
    "map_confidence": ", confianza {percent}",
    "map_next": "Puedes pedir las habilidades esenciales u opcionales, o elegir otro número.",
    "miss": "No encontré nada para esa frase con la búsqueda de hoy. Es un no, no un tal vez.",
    "no_published_facts": (
        "{suite} no publica cifras de salario, perspectivas ni formación para esta ocupación."
    ),
    "no_subject": (
        "¿A qué ocupación o habilidad te refieres? Por ejemplo: "
        "“¿Qué habilidades necesita una enfermera?” o “¿Qué es un científico de datos?”"
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
    "transitions_head": "**{title}** — {count} empleos a los que pasar ({suite}):",
    "related_head": "**{title}** — {count} ocupaciones relacionadas ({suite}):",
    "long_input": (
        "Tu mensaje es largo: leí el principio y el final ({limit} caracteres en total)."
    ),
    "filtered_count": "{count} {kind} (de {total} en total)",
    "none_marked_count": "ninguna marcada como {kind}; todas: {counts}",
    "transitions_foot": (
        "El porcentaje indica cuánto de la experiencia se aprovecha según la taxonomía. "
        "Es el mapa, no un consejo profesional."
    ),
    "no_transitions": "{suite} no publica pasos de un empleo a otro para **{title}**.",
    "connect_foot_moves": (
        "Ocupaciones relacionadas según la taxonomía. Es el mapa, no un consejo profesional."
    ),
    "connect_more": "Hay {count} más en los detalles de la consulta.",
    "connect_foot": "Son vecinos en el grafo, no un plan de estudio.",
    "compare_head": "**{first}** frente a **{second}** — {shared}.",
    "compare_shared": "En común",
    "compare_only": "Solo {title}",
    "compare_none": "ninguna",
    "compare_more": "(+{count} más)",
    "compare_foot": "Son vecinos en el grafo, no una recomendación.",
    "pathfind_head": "**De {start} a {target}:**",
    "pathfind_essential": "Habilidades esenciales necesarias",
    "pathfind_optional": "Habilidades opcionales necesarias",
    "pathfind_software": "Software/herramientas necesarias",
    "pathfind_gap_none": "ninguna",
    "pathfind_more": "(+{count} más)",
    "pathfind_full_retrain": (
        "Esta brecha es lo bastante grande como para ser una reconversión "
        "completa, no una transición rápida."
    ),
    "pathfind_route": "Ruta {number}: {chain}",
    "pathfind_more_routes": "(+{count} rutas más)",
    "pathfind_no_route": (
        "No se encontró ninguna ruta en {steps} pasos. La brecha de arriba sigue siendo válida."
    ),
    "pathfind_no_facts": (
        "No hay datos publicados de salario, perspectivas o educación para este puesto."
    ),
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
    "compare_needs_pick": (
        "Para comparar {first} y {second}, primero necesito saber a qué {subject} te refieres."
    ),
    "comparing": "Comparando {first} con {second}:",
    "still_waiting": "{suite} todavía necesita que elijas uno para comparar ahí también:",
    "bound_many": "Seleccionados: {titles}.",
    "one_per_suite": "Elige como máximo un título por taxonomía, por ejemplo {example}.",
    "pick_per_suite": (
        "Puedes elegir un título de cada lista en una sola respuesta, por ejemplo {example}."
    ),
}

_SV: dict[str, str] = {
    "your_request": "din fråga",
    "read_as_titles": "Jag tolkade {quoted} som {titles}.",
    "looked_up": 'Jag sökte på "{subject}".',
    "read_as_subject": 'Jag tolkade {quoted} som "{subject}".',
    "or": "eller",
    "and": "och",
    "matching_title": "matchande titel",
    "matching_titles": "matchande titlar",
    "close_title": "närliggande titel",
    "close_titles": "närliggande titlar",
    "area_title": "titel",
    "area_titles": "titlar",
    "found_list": "{suite} har {count} {noun}",
    "found_close": "{suite} har ingen exakt träff; {count} {noun}",
    "found_one": "{suite} matchade {title}",
    "found_none": "{suite} har ingen träff",
    "next_goal": "Välj den du menar så sparar jag den som ditt mål.",
    "next_standing": "Välj den du menar så sparar jag den som ditt nuvarande yrke.",
    "next_reject": "Välj den du menar så noterar jag att det inte är ditt yrke.",
    "next_default": "Välj den du menar, eller berätta mer så smalnar jag av listan.",
    "picker_heading": 'Jag hittade flera träffar för "{question}". Vilken menade du?',
    "picker_empty": 'Jag hittade flera träffar för "{question}", men inga att visa.',
    "picker_one": "Jag väljer inte åt dig — svara med {first}.",
    "picker_range": "Jag väljer inte åt dig — svara med {first}–{last}.",
    "picker_showing": (
        "Visar {shown} av {total} ({omitted} till). Hela listan finns i frågedetaljerna."
    ),
    "intro_guided": "Titlar i {suite} som passar din fråga. Vilken menade du?",
    "intro_several": 'Jag hittade flera träffar i {suite} för "{searched}". Vilken menade du?',
    "areas_head": "**Eller avgränsa efter område:**",
    "areas_foot": "Svara med en bokstav, eller beskriv arbetet du tänker på.",
    "bound": "Vald: {title}.",
    "narrow_fit": 'De här passar "{text}". Vilken menade du?',
    "narrow_area": '{names}: titlar som matchar "{query}". Vilken menade du?',
    "narrow_none": (
        "Ingen av titlarna i listan passar det. Beskriv det på ett annat sätt, eller ange en titel."
    ),
    "map_record": "Post i kartan: **{title}** ({kind}{confidence})",
    "map_confidence": ", säkerhet {percent}",
    "map_next": (
        "Du kan fråga efter nödvändiga kompetenser, valfria kompetenser, "
        "eller välja ett annat nummer."
    ),
    "miss": "Ingen nod för den frasen med dagens sökning. Det är en miss, inte ett kanske.",
    "no_published_facts": (
        "{suite} publicerar inga siffror om lön, framtidsutsikter eller utbildning "
        "för det här yrket."
    ),
    "no_subject": (
        "Vilket yrke eller vilken kompetens menar du? Till exempel: "
        "”Vilka kompetenser behöver en sjuksköterska?” eller ”Var finns data scientist?”"
    ),
    "skill": "kompetens",
    "skills": "kompetenser",
    "tool": "verktyg",
    "tools": "verktyg",
    "shared_skill": "gemensam kompetens",
    "shared_skills_noun": "gemensamma kompetenser",
    "shared_tool": "gemensamt verktyg",
    "shared_tools_noun": "gemensamma verktyg",
    "connect_head": "**{title}** — {counts} på kartan:",
    "transitions_head": "**{title}** — {count} yrken att byta till ({suite}):",
    "related_head": "**{title}** — {count} relaterade yrken ({suite}):",
    "long_input": (
        "Ditt meddelande är långt, så jag läste början och slutet ({limit} tecken totalt)."
    ),
    "filtered_count": "{count} {kind} (av {total} totalt)",
    "none_marked_count": "inga markerade {kind}; totalt {counts}",
    "transitions_foot": (
        "Procentsatsen anger hur stor del av erfarenheten taxonomin säger kan överföras. "
        "Det här är kartan, inte karriärråd."
    ),
    "no_transitions": "{suite} publicerar inga yrkesbyten för **{title}**.",
    "connect_foot_moves": (
        "Relaterade yrken så som taxonomin listar dem. Det här är kartan, inte karriärråd."
    ),
    "connect_more": "{count} till finns i frågedetaljerna.",
    "connect_foot": "Det här är grannar i grafen, inte en studieplan.",
    "compare_head": "**{first}** mot **{second}** — {shared}.",
    "compare_shared": "Gemensamma",
    "compare_only": "Bara {title}",
    "compare_none": "inga",
    "compare_more": "(+{count} till)",
    "compare_foot": "Det här är grannar i grafen, inte en rekommendation.",
    "pathfind_head": "**Från {start} till {target}:**",
    "pathfind_essential": "Nödvändiga kärnkompetenser",
    "pathfind_optional": "Valfria kompetenser som behövs",
    "pathfind_software": "Programvara/verktyg som behövs",
    "pathfind_gap_none": "inga",
    "pathfind_more": "(+{count} till)",
    "pathfind_full_retrain": (
        "Den här kompetensklyftan är stor nog för att vara en fullständig "
        "omskolning, inte en snabb övergång."
    ),
    "pathfind_route": "Rutt {number}: {chain}",
    "pathfind_more_routes": "(+{count} rutter till)",
    "pathfind_no_route": (
        "Ingen rutt hittades inom {steps} steg. Kompetensklyftan ovan gäller fortfarande."
    ),
    "pathfind_no_facts": "Ingen publicerad lön-, utsikts- eller utbildningsdata för detta yrke.",
    "profile_empty": (
        "Jag vet inte mycket om dig än. Berätta ditt nuvarande yrke (”jag är …”) "
        "eller ett yrke du siktar på (”mitt mål är …”) så noterar jag det."
    ),
    "profile_head": "Det här har du berättat för mig:",
    "profile_foot": (
        "Det är allt jag sparar om dig. Berätta ditt mål eller nuvarande yrke för att ändra det."
    ),
    "profile_goal": "Mål: {title}",
    "profile_current": "Nuvarande yrke ({suite}): {title}",
    "profile_rejected": "Inte ditt yrke (enligt dig): {title}",
    "noted_not_job": "Noterat: {titles} är inte längre sparat som ditt nuvarande yrke.",
    "compare_needs_pick": (
        "För att jämföra {first} och {second} behöver jag först veta vilken {subject} du menar."
    ),
    "comparing": "Jämför {first} med {second}:",
    "still_waiting": "{suite} behöver också ett val för att jämföra där:",
    "bound_many": "Valda: {titles}.",
    "one_per_suite": "Välj högst en titel per taxonomi, t.ex. {example}.",
    "pick_per_suite": (
        "Du kan välja en titel från varje lista i ett och samma svar, t.ex. {example}."
    ),
}

_TABLES: dict[str, dict[str, str]] = {"en": _EN, "es": _ES, "sv": _SV}


def normalize_language(code: object) -> str | None:
    """The two-letter language code of a message ("es-MX" -> "es"), or None.

    Any language is kept, so the model is told the right one; ``t`` falls back
    to English sentences for a language without a table.
    """
    if not isinstance(code, str):
        return None
    match = _LANGUAGE_CODE.fullmatch(code.strip().casefold())
    return match.group(1) if match else None


def set_language(code: object) -> None:
    """The language of this turn's reply (English if unknown)."""
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
