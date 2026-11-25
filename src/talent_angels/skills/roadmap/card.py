"""The learning order as a compact warning on a Pathfind result, for the chat path card.

``AgentResult`` is a typed contract with no field for it, so it rides in ``warnings`` the way
route scores do (``route_scores:``): a few labels per step, small enough to read at a glance.
"""

from __future__ import annotations

import json
from typing import Any

from talent_angels.skills.evaluate.policy import LEARN_ORDER_V1 as P
from talent_angels.skills.roadmap.roadmap import roadmap_suite

PREFIX = "learning_order:"
CARD_PER_STEP = 3


def card_warning(suite: Any, name: str, start_id: str, target_id: str) -> str | None:
    """``learning_order:[[step, [[label, stated], ...], more], ...]``, or None when empty."""
    try:
        plan = roadmap_suite(suite, name, start_id, target_id)
    except Exception:  # the card is an extra: never let it break the Pathfind answer
        return None
    steps = []
    for number in range(1, len(P.step_names) + 1):
        here = [i for i in plan.items if i.step == number]
        if here:
            shown = [[i.label, i.stated] for i in here[:CARD_PER_STEP]]
            steps.append([number, shown, max(0, len(here) - CARD_PER_STEP)])
    return PREFIX + json.dumps(steps, ensure_ascii=False, separators=(",", ":")) if steps else None
