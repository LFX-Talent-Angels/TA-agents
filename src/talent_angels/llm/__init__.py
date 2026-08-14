"""Provider-agnostic LLM layer (MVP plan Sec 2.8).

`LLM_PROVIDER=none` (default) is a deterministic, zero-token stub — Locate
runs entirely on `search_nodes` and never needs this. `LLM_PROVIDER=anthropic`
is the original Gate A path. `LLM_PROVIDER=litellm` is the unified adapter
(OpenRouter first). New providers are new client modules behind the same
`LLMClient` protocol, chosen by `get_llm_client()`.
"""

from talent_angels.llm.factory import get_llm_client
from talent_angels.llm.protocol import LLMClient, LLMResult, LLMUsage, Message

__all__ = ["LLMClient", "LLMResult", "LLMUsage", "Message", "get_llm_client"]
