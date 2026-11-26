"""Interactive `/connect` and `ta-agent setup` question flow for bringing your own LLM."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rich.console import Console

from talent_angels.env import load_local_dotenv
from talent_angels.session.copy import NO_MODEL_NOTICE
from talent_angels.session.credentials import (
    PROVIDERS,
    CredentialError,
    looks_like_openrouter_key,
    save_settings,
    settings_for,
    verify_model,
)
from talent_angels.tui.picker_ui import is_interactive

_ORDER = tuple(PROVIDERS.keys())


def _pick_provider(console: Console) -> str | None:
    console.print(
        "Which LLM do you use? "
        "[dim](Enter = none: every capability still works, deterministically)[/]"
    )
    rows = [f"  {idx} {PROVIDERS[key].label}" for idx, key in enumerate(_ORDER, start=1)]
    console.print("\n".join(rows[:4]))
    console.print("\n".join(rows[4:]))
    raw = console.input("[bold cyan]›[/] ").strip()
    if not raw or raw.casefold() in ("none", "0"):
        return "none"
    if raw.isdigit() and 1 <= int(raw) <= len(_ORDER):
        return _ORDER[int(raw) - 1]
    lowered = raw.casefold()
    if lowered in PROVIDERS:
        return lowered
    return None


def ask_and_save(console: Console, *, first_run: bool = False) -> str:
    """Which LLM -> model -> key (hidden) -> verify -> save. Returns one line to print."""
    try:
        provider = _pick_provider(console)
        if provider is None:
            return "Unknown choice. Nothing was stored."
        if provider == "none":
            save_settings({"LLM_PROVIDER": "none"})
            return NO_MODEL_NOTICE if first_run else "Saved: no model (deterministic mode)."

        spec = PROVIDERS[provider]
        hint = f" [{spec.default_model}]" if spec.default_model else ""
        model_raw = console.input(f"Model{hint}: ").strip()
        model = model_raw or spec.default_model
        if not model or model == "azure/<your deployment>":
            return "No model entered. Nothing was stored."

        key: str | None = None
        if spec.key_env is not None:
            key = console.input("API key (input hidden): ", password=True).strip()
            if not key and provider != "litellm":
                return "Nothing entered. Nothing was stored."
            if provider == "openrouter" and key and not looks_like_openrouter_key(key):
                return (
                    "That does not look like an OpenRouter key (they start with sk-or-). "
                    "Nothing was stored."
                )

        extras: dict[str, str] = {}
        for env_name, default_val in spec.extras:
            extra_hint = f" [{default_val}]" if default_val else ""
            ans = console.input(f"{env_name}{extra_hint}: ").strip()
            extras[env_name] = ans or default_val

        settings = settings_for(provider, model, key, extras)
        console.print("[dim]Checking with one tiny test call…[/]")
        verify_model(provider, settings)
        path = save_settings(settings)
        return f"Saved: {settings['LLM_MODEL']}. Settings in {path} (owner-only)."
    except (EOFError, KeyboardInterrupt):
        console.print()
        return "Cancelled. Nothing was stored."
    except CredentialError as exc:
        return f"Not stored: {exc}."


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ta-agent setup",
        description=(
            "Configure an LLM provider and save settings to <TA home>/llm.env (owner-only)."
        ),
    )
    p.add_argument(
        "--provider",
        choices=(*_ORDER, "none"),
        default=None,
        help="Provider name (omit for interactive prompt).",
    )
    p.add_argument("--model", default="", help="Model name or deployment id.")
    p.add_argument("--api-base", default="", help="Base URL for Azure, Ollama, or LiteLLM.")
    p.add_argument("--api-version", default="", help="API version for Azure OpenAI.")
    return p


def setup_main(argv: Sequence[str] | None = None) -> int:
    """`ta-agent setup [--provider P --model M --api-base URL --api-version V]`."""
    load_local_dotenv()
    args = _build_parser().parse_args(list(argv or ()))
    console = Console()

    if args.provider is None:
        if not is_interactive():
            console.print("give --provider when stdin is not a terminal.")
            return 2
        msg = ask_and_save(console)
        console.print(msg)
        return 0 if msg.startswith(("Saved:", NO_MODEL_NOTICE)) else 2

    provider: str = args.provider
    if provider == "none":
        try:
            save_settings({"LLM_PROVIDER": "none"})
        except CredentialError as exc:
            console.print(f"Not stored: {exc}.")
            return 2
        console.print("Saved: no model (deterministic mode).")
        return 0

    spec = PROVIDERS[provider]
    key: str | None = None
    if spec.key_env is not None:
        import os

        key = os.environ.get(spec.key_env, "").strip() or None
        if key is None and not sys.stdin.isatty():
            key = sys.stdin.readline().rstrip("\r\n").strip() or None
        if key is None and is_interactive():
            try:
                key = console.input("API key (input hidden): ", password=True).strip() or None
            except (EOFError, KeyboardInterrupt):
                console.print("\nCancelled. Nothing was stored.")
                return 2

    extras: dict[str, str] = {}
    if args.api_base:
        for env_name, _ in spec.extras:
            if env_name.endswith("_API_BASE"):
                extras[env_name] = args.api_base
    if args.api_version and provider == "azure":
        extras["AZURE_API_VERSION"] = args.api_version

    try:
        settings = settings_for(provider, args.model, key, extras)
        verify_model(provider, settings)
        path = save_settings(settings)
    except CredentialError as exc:
        console.print(f"Not stored: {exc}.")
        return 2

    console.print(f"Saved: {settings['LLM_MODEL']}. Settings in {path} (owner-only).")
    return 0
