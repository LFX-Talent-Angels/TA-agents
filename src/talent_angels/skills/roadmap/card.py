"""The learning order as a compact warning on a Pathfind result, for the chat path card.

``AgentResult`` is a typed contract with no field for it, so it rides in ``warnings`` the way
route scores do (``route_scores:``): a few labels per step, small enough to read at a glance.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from talent_angels.skills.evaluate.policy import LEARN_ORDER_V1 as P
from talent_angels.skills.roadmap.roadmap import roadmap_suite

PREFIX = "learning_order:"
FAILED = "learning_order_failed:"
_log = logging.getLogger(__name__)
CARD_PER_STEP = 3


def card_warning(suite: Any, name: str, start_id: str, target_id: str) -> str | None:
    """``learning_order:[[step, [[label, stated], ...], more], ...]``, None when empty, or
    ``learning_order_failed:<error type>`` when the roadmap could not be built."""
    try:
        plan = roadmap_suite(suite, name, start_id, target_id)
    except Exception as exc:  # the card is an extra: it must not break the Pathfind answer
        _log.warning("learning order for %s -> %s failed", start_id, target_id, exc_info=True)
        return FAILED + type(exc).__name__
    steps = []
    for number in range(1, len(P.step_names) + 1):
        here = [i for i in plan.items if i.step == number]
        if here:
            shown = [[i.label, i.stated] for i in here[:CARD_PER_STEP]]
            steps.append([number, shown, max(0, len(here) - CARD_PER_STEP)])
    return PREFIX + json.dumps(steps, ensure_ascii=False, separators=(",", ":")) if steps else None
