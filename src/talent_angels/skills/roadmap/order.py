"""The order of roadmap items under learn-order-v1: stated links first, then steps, then keys."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from talent_angels.skills.evaluate.policy import LEARN_ORDER_V1

RAISED = "step:raised-by-stated-prerequisite"


@dataclass
class Item:
    id: str
    label: str
    kind: str
    cite: str
    step: int
    reasons: list[str] = field(default_factory=list)
    stated: bool = False  # a taxonomy states this item as a prerequisite of another item
    levels: dict[str, float | None] | None = None  # O*NET: from, to, jump, importance
    jump: float = 0.0
    importance: float = 0.0

    @property
    def key(self) -> tuple[int, float, float, str, str]:
        return (self.step, -self.jump, -self.importance, self.label.casefold(), self.id)

    def to_dict(self, suite: str) -> dict[str, object]:
        out: dict[str, object] = {
            "suite": suite,
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "cite": self.cite,
            "step": self.step,
            "reasons": list(self.reasons),
            "stated_prerequisite": self.stated,
        }
        if self.levels is not None:
            out["levels"] = dict(self.levels)
        return out


def step_of_reuse(reuse: object) -> int:
    p = LEARN_ORDER_V1
    return p.reuse_steps[p.reuse_order.index(reuse)] if reuse in p.reuse_order else p.default_step


def order_items(
    items: list[Item], links: Iterable[tuple[str, str]]
) -> tuple[list[Item], list[str]]:
    """Order ``items``. ``links`` are (dependent, prerequisite) pairs a taxonomy states.

    A prerequisite never comes after its dependent: the dependent's step is raised to the
    prerequisite's, and inside the order a dependent waits for its prerequisites. A cycle is
    broken at the lowest node ID of the lowest step, and reported.
    """
    by_id = {i.id: i for i in items}
    pairs = sorted({(d, p) for d, p in links if d != p and d in by_id and p in by_id})
    needs: dict[str, set[str]] = {i.id: set() for i in items}
    for dep, pre in pairs:
        needs[dep].add(pre)
        by_id[pre].stated = True
        by_id[pre].reasons.append(f"esco:essential-for:{dep}")
        by_id[dep].reasons.append(f"esco:needs:{pre}")
    changed = True
    while changed:  # steps only go up and top out at 4, so this ends
        changed = False
        for dep, pre in pairs:
            if by_id[dep].step < by_id[pre].step:
                by_id[dep].step = by_id[pre].step
                if RAISED not in by_id[dep].reasons:
                    by_id[dep].reasons.append(RAISED)
                changed = True
    left = dict(by_id)
    ordered: list[Item] = []
    warnings: list[str] = []
    while left:
        low = min(i.step for i in left.values())  # steps stay grouped: the lowest goes first
        here = [i for i in left.values() if i.step == low]
        ready = [i for i in here if not needs[i.id] & left.keys()]
        if ready:
            pick = min(ready, key=lambda i: i.key)
        else:  # only a cycle is left in this step
            pick = min(here, key=lambda i: i.id)
            warnings.append(f"cycle_broken:{pick.id}")
        ordered.append(pick)
        del left[pick.id]
    return ordered, warnings
