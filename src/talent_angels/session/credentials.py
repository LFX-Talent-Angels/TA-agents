"""Store an OpenRouter key without it ever reaching the model.

A session with no key starts on the deterministic stub and never says why, so
the fix — put a key in a file you have not opened yet — is invisible from
inside the product.

The key must not travel where the rest of a turn travels. Three places would
leak it: the transcript, which is sent to the model on every turn; the run log,
which records the question; and the terminal, which is often shared on a call.
So the value is never returned through the normal input path, never recorded,
and the caller is expected to read it with echo disabled.

It is verified before it is written. Storing an unverified key would move the
failure to the next question, where it reads as a broken assistant rather than
a bad paste.
"""

from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path

VERIFY_URL = "https://openrouter.ai/api/v1/key"
VERIFY_TIMEOUT_SECONDS = 15.0
ENV_NAME = "OPENROUTER_API_KEY"
# Owner read/write only. The file holds a credential and sits in a repo
# checkout that other tools walk.
_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


class CredentialError(RuntimeError):
    """The key was not accepted or could not be stored, with the reason."""


def looks_like_openrouter_key(value: str) -> bool:
    """Catch an obvious mis-paste before spending a network round trip."""
    return value.startswith("sk-or-") and len(value) > 20


def verify(key: str, *, url: str = VERIFY_URL) -> str:
    """Ask OpenRouter whether the key works. Returns a short human summary."""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=VERIFY_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CredentialError("OpenRouter rejected that key") from exc
        raise CredentialError(f"OpenRouter returned {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise CredentialError(f"could not reach OpenRouter ({exc})") from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return "key accepted"
    limit = data.get("limit")
    usage = data.get("usage")
    if limit is None:
        return "key accepted, no spend limit set"
    return f"key accepted, usage {usage} of limit {limit}"


def env_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / ".env"


def store(key: str, *, root: Path | None = None) -> Path:
    """Write the key to `.env`, replacing any previous value, and lock the file.

    `.env` is already gitignored, which is why it is the destination; the mode
    change is for everything that is not git.
    """
    path = env_path(root)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [line for line in lines if not line.startswith(f"{ENV_NAME}=")]
    kept.append(f"{ENV_NAME}={key}")
    try:
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        path.chmod(_FILE_MODE)
    except OSError as exc:
        raise CredentialError(f"could not write {path} ({exc})") from exc

    # The running process needs it too, or the switch only works next launch.
    os.environ[ENV_NAME] = key
    return path


def have_key() -> bool:
    return bool((os.environ.get(ENV_NAME) or "").strip())
