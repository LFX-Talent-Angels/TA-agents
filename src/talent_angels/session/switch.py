"""Resolve `/model` input into a live client without restarting the session.

The provider is read from the environment once at startup, so trying a second
model meant editing `.env` and starting over, losing the conversation. That is
slow while comparing models and worse when a provider starts rate-limiting
mid-session: the only recovery was a restart.

`none` is always reachable. It is the deterministic stub — no network, no
tokens — so a session can always retreat to something that answers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from talent_angels.llm import LLMClient, get_llm_client
from talent_angels.session.catalog import CatalogUnavailable, FreeModel, fetch_free_models

STUB_WORDS = {"none", "stub", "off"}


class SwitchError(ValueError):
    """The requested model cannot be used, with the reason to show the user."""


@dataclass(frozen=True)
class Choice:
    provider: str
    model: str

    def label(self) -> str:
        """Short enough for a status line: the provider prefix is redundant
        next to a model id that already names its vendor."""
        if self.provider == "none":
            return "none (stub)"
        return self.model.removeprefix("openrouter/")


def current_choice() -> Choice:
    return Choice(
        provider=(os.environ.get("LLM_PROVIDER") or "none").strip().lower() or "none",
        model=(os.environ.get("LLM_MODEL") or "").strip(),
    )


def resolve(argument: str, catalogue: list[FreeModel]) -> Choice:
    """Read a stub word, a catalogue number, or an explicit model slug."""
    text = argument.strip()
    if text.lower() in STUB_WORDS:
        return Choice(provider="none", model="stub")

    if text.isdigit():
        index = int(text)
        if not catalogue:
            raise SwitchError("the catalogue is unavailable; pass a model slug instead")
        if not 1 <= index <= len(catalogue):
            raise SwitchError(f"pick a number between 1 and {len(catalogue)}")
        return Choice(provider="litellm", model=catalogue[index - 1].slug)

    # An explicit slug. Accept it with or without the LiteLLM provider prefix,
    # since the catalogue prints ids and the README writes prefixed slugs.
    slug = text if text.startswith("openrouter/") else f"openrouter/{text}"
    return Choice(provider="litellm", model=slug)


def apply(choice: Choice) -> LLMClient:
    """Commit the environment only once a client was built successfully.

    Order matters: a failed switch must leave the session on whatever was
    already answering rather than stranding it on something broken.
    """
    if choice.provider == "litellm" and not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        raise SwitchError("OPENROUTER_API_KEY is not set; /model none still works")

    previous = current_choice()
    os.environ["LLM_PROVIDER"] = choice.provider
    os.environ["LLM_MODEL"] = choice.model
    try:
        return get_llm_client()
    except Exception as exc:
        os.environ["LLM_PROVIDER"] = previous.provider
        os.environ["LLM_MODEL"] = previous.model
        raise SwitchError(str(exc)) from exc


def load_catalogue() -> tuple[list[FreeModel], str | None]:
    """Free models plus a note when the list could not be fetched."""
    try:
        return fetch_free_models(), None
    except CatalogUnavailable as exc:
        return [], f"Could not reach the OpenRouter catalogue ({exc}). Switch by slug instead."
