"""List the OpenRouter models that cost nothing to call.

Comparing models means editing `.env` and restarting, which loses the session
and makes an A/B awkward enough that it does not happen. Free models are the
ones worth flipping between while developing, and which ones exist changes
often — several are promotional and disappear — so the list is fetched rather
than pinned in the repository where it would rot.

Network failure is not an error here: the catalogue is a convenience, and a
switch by explicit slug must keep working on a plane.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

MODELS_URL = "https://openrouter.ai/api/v1/models"
FETCH_TIMEOUT_SECONDS = 15.0
_FREE_PRICES = {"0", "0.0", "0.00"}


@dataclass(frozen=True)
class FreeModel:
    slug: str  # as LiteLLM wants it, e.g. openrouter/poolside/laguna-s-2.1:free
    name: str
    context: int

    @property
    def id(self) -> str:
        """The OpenRouter id, without the LiteLLM provider prefix."""
        return self.slug.removeprefix("openrouter/")


class CatalogUnavailable(RuntimeError):
    """The catalogue could not be fetched; switching by slug still works."""


def _is_free(model: dict[str, object]) -> bool:
    pricing = model.get("pricing")
    if not isinstance(pricing, dict):
        return False
    prompt = str(pricing.get("prompt", ""))
    completion = str(pricing.get("completion", ""))
    return prompt in _FREE_PRICES and completion in _FREE_PRICES


def fetch_free_models(*, url: str = MODELS_URL) -> list[FreeModel]:
    """Free OpenRouter models, cheapest-to-reason-about first (by name)."""
    try:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise CatalogUnavailable(str(exc)) from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise CatalogUnavailable("unexpected catalogue shape")

    models: list[FreeModel] = []
    for entry in data:
        if not isinstance(entry, dict) or not _is_free(entry):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        context = entry.get("context_length")
        models.append(
            FreeModel(
                slug=f"openrouter/{model_id}",
                name=str(entry.get("name") or model_id),
                context=context if isinstance(context, int) else 0,
            )
        )
    return sorted(models, key=lambda m: m.name.casefold())


def render_catalogue(models: list[FreeModel], *, current: str) -> str:
    """A numbered list; the number is what the user types to switch."""
    if not models:
        return "No free models are listed right now."
    # The reply is rendered as Markdown, so angle brackets read as HTML and
    # "1." starts an ordered list that gets renumbered. Avoid both.
    lines = ["Free OpenRouter models. Type `/model N` to switch:", "", "```"]
    for index, model in enumerate(models, start=1):
        marker = " ←" if model.slug == current else ""
        context = f"  {model.context // 1000}k" if model.context else ""
        lines.append(f"{index:>2}  {model.id}{context}{marker}")
    lines.append("```")
    lines.append("`/model none` — deterministic stub, no network, no tokens.")
    return "\n".join(lines)
