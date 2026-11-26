"""Store LLM provider settings (OpenAI, Anthropic, Gemini, OpenRouter, Azure, Ollama, LiteLLM).

Keys are read with echo disabled (or from env/stdin in ``ta-agent setup``), verified
with one tiny check before anything is written, and saved owner-only (0600) in
``<TA home>/llm.env`` so they never enter transcripts, logs, or saved calls.
"""

from __future__ import annotations

import os
import stat
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from talent_angels.llm.litellm_client import ensure_local_model_cost_map, quiet_stdio
from talent_angels.memory.paths import ensure_home, home

VERIFY_URL = "https://openrouter.ai/api/v1/key"
VERIFY_TIMEOUT_SECONDS = 15.0
ENV_NAME = "OPENROUTER_API_KEY"
SETTINGS_FILE = "llm.env"
# Owner read/write only.
_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


@dataclass(frozen=True)
class Provider:
    label: str
    prefix: str
    key_env: str | None
    default_model: str
    extras: tuple[tuple[str, str], ...] = ()


PROVIDERS: dict[str, Provider] = {
    "openai": Provider("OpenAI", "openai/", "OPENAI_API_KEY", "openai/gpt-4o-mini"),
    "anthropic": Provider(
        "Anthropic", "anthropic/", "ANTHROPIC_API_KEY", "anthropic/claude-haiku-4-5"
    ),
    "gemini": Provider("Google Gemini", "gemini/", "GEMINI_API_KEY", "gemini/gemini-2.5-flash"),
    "openrouter": Provider(
        "OpenRouter",
        "openrouter/",
        "OPENROUTER_API_KEY",
        "openrouter/poolside/laguna-s-2.1:free",
    ),
    "azure": Provider(
        "Azure OpenAI",
        "azure/",
        "AZURE_API_KEY",
        "azure/<your deployment>",
        (("AZURE_API_BASE", ""), ("AZURE_API_VERSION", "2024-10-21")),
    ),
    "ollama": Provider(
        "Ollama (local, no key)",
        "ollama/",
        None,
        "ollama/llama3.1",
        (("OLLAMA_API_BASE", "http://localhost:11434"),),
    ),
    "litellm": Provider(
        "Any LiteLLM model string",
        "",
        "TA_LLM_API_KEY",
        "",
        (("TA_LLM_API_BASE", ""),),
    ),
}


class CredentialError(RuntimeError):
    """The credentials were not accepted or could not be stored, with category reason only."""


def settings_path() -> Path:
    return home() / SETTINGS_FILE


def looks_like_openrouter_key(value: str) -> bool:
    """Catch an obvious mis-paste before spending a network round trip."""
    return value.startswith("sk-or-") and len(value) > 20


def verify(key: str, *, url: str = VERIFY_URL) -> str:
    """Ask OpenRouter whether the key works (free GET, no tokens). Returns a short summary."""
    import json

    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=VERIFY_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CredentialError("the provider rejected the key") from None
        raise CredentialError(f"OpenRouter returned {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise CredentialError(f"could not reach OpenRouter ({type(exc).__name__})") from None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return "key accepted"
    limit = data.get("limit")
    usage = data.get("usage")
    if limit is None:
        return "key accepted, no spend limit set"
    return f"key accepted, usage {usage} of limit {limit}"


def settings_for(
    provider: str,
    model: str = "",
    key: str | None = None,
    extras: Mapping[str, str] = (),  # type: ignore[assignment]
) -> dict[str, str]:
    """Build the env dict to save in `llm.env`. `provider='none'` disables the model."""
    if provider == "none":
        return {"LLM_PROVIDER": "none"}
    if provider not in PROVIDERS:
        raise CredentialError(f"unknown provider {provider!r}")

    spec = PROVIDERS[provider]
    chosen = model.strip() or spec.default_model
    if not chosen or chosen == "azure/<your deployment>":
        raise CredentialError("a model or deployment name is required")
    if spec.prefix and not chosen.startswith(spec.prefix):
        chosen = spec.prefix + chosen

    out: dict[str, str] = {"LLM_PROVIDER": "litellm", "LLM_MODEL": chosen}
    if spec.key_env and key and key.strip():
        out[spec.key_env] = key.strip()
    elif spec.key_env and provider != "litellm":
        raise CredentialError(f"an API key ({spec.key_env}) is required")

    extras_map = dict(extras) if extras else {}
    for env_key, default_val in spec.extras:
        val = extras_map.get(env_key, default_val).strip()
        if val:
            out[env_key] = val
    return out


def _kwargs(provider: str, settings: Mapping[str, str]) -> dict[str, Any]:
    spec = PROVIDERS[provider]
    kw: dict[str, Any] = {}
    if spec.key_env and settings.get(spec.key_env):
        kw["api_key"] = settings[spec.key_env]
    for base_key in ("AZURE_API_BASE", "OLLAMA_API_BASE", "TA_LLM_API_BASE"):
        if settings.get(base_key):
            kw["api_base"] = settings[base_key]
    if settings.get("AZURE_API_VERSION"):
        kw["api_version"] = settings["AZURE_API_VERSION"]
    return kw


def _reason(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "AuthenticationError":
        return "the provider rejected the key"
    if name == "NotFoundError":
        return "the provider doesn't know that model"
    if name == "RateLimitError":
        return "rate-limited or out of credit"
    if name in ("Timeout", "APIConnectionError", "TimeoutError"):
        return "could not reach the provider"
    if name == "BadRequestError":
        return "the provider refused the test request"
    return f"the check failed ({name})"


def verify_model(provider: str, settings: Mapping[str, str]) -> str:
    """Verify credentials with the cheapest possible check. Never leaks provider text."""
    if provider == "none":
        return "no model selected"
    if provider == "openrouter":
        return verify(settings.get("OPENROUTER_API_KEY", ""))

    ensure_local_model_cost_map()
    reason: str | None = None
    with quiet_stdio():
        from litellm import completion

        try:
            completion(
                model=settings["LLM_MODEL"],
                messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=8,
                temperature=0,
                drop_params=True,
                timeout=VERIFY_TIMEOUT_SECONDS,
                num_retries=0,
                **_kwargs(provider, settings),
            )
        except Exception as exc:  # noqa: BLE001 - category only, never chain raw message
            reason = _reason(exc)
    if reason is not None:
        raise CredentialError(reason)
    return "the model answered"


def _dotenv_line(name: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{name}="{escaped}"'


def save_settings(settings: Mapping[str, str], *, path: Path | None = None) -> Path:
    """Replace the entire `llm.env` file atomically (mode 0600) and update `os.environ`."""
    if path is None:
        ensure_home()
        target = settings_path()
    else:
        target = path
        target.parent.mkdir(parents=True, exist_ok=True)

    content = "".join(_dotenv_line(k, v) + "\n" for k, v in settings.items())
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), _FILE_MODE)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise CredentialError(f"could not write {target} ({type(exc).__name__})") from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)

    # Clear stale provider extras / keys from earlier setups in the running process.
    for spec in PROVIDERS.values():
        if spec.key_env and spec.key_env not in settings:
            os.environ.pop(spec.key_env, None)
        for extra_key, _ in spec.extras:
            if extra_key not in settings:
                os.environ.pop(extra_key, None)
    if settings.get("LLM_PROVIDER") == "none":
        os.environ.pop("LLM_MODEL", None)
    os.environ.update(settings)
    return target
