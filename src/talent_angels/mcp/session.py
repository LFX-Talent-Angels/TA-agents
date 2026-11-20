"""Suite lifecycle for a long-lived MCP process.

Claude Code starts this server when its own session starts, keeps it running
for hours, and may well start it while Neo4j is down. So suites are opened on
first use rather than at startup: a database that is not up yet becomes a
warning on one tool call that the next call can recover from, instead of a
server that refuses to launch. Each opened suite is then kept for the life of
the process — one driver and one connection pool per suite, as the FastAPI
edge does for one request cycle.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import TracebackType

from talent_angels.suites import SuiteRegistry, SuiteRuntime, UnknownSuiteError

__all__ = ["SuiteSession", "SuiteUnavailableError"]


class SuiteUnavailableError(RuntimeError):
    """A registered suite could not be opened — infrastructure, not input."""

    def __init__(self, suite: str, cause: BaseException) -> None:
        # Only the exception's type reaches the client. Driver messages can
        # quote the connection URI, and this text ends up in a chat transcript.
        self.suite = suite
        self.reason = type(cause).__name__
        super().__init__(f"suite {suite!r} unavailable: {self.reason}")


class SuiteSession:
    """Open suites lazily by name and close them all on exit."""

    def __init__(self, registry: SuiteRegistry) -> None:
        self._registry = registry
        self._stack = ExitStack()
        self._open: dict[str, SuiteRuntime] = {}

    @property
    def registry(self) -> SuiteRegistry:
        return self._registry

    def runtime(self, suite: str | None = None) -> SuiteRuntime:
        """Return an open runtime, opening it on first use.

        Raises ``UnknownSuiteError`` for a name that is not registered (a
        client mistake) and ``SuiteUnavailableError`` when the adapter cannot
        connect (an operator problem). The two are reported differently.
        """
        name = self._registry.default if suite is None else suite
        cached = self._open.get(name)
        if cached is not None:
            return cached

        manager = self._registry.open(name)  # UnknownSuiteError escapes here.
        try:
            runtime = self._stack.enter_context(manager)
        except UnknownSuiteError:
            raise
        except Exception as exc:  # noqa: BLE001 — any adapter failure is "unavailable".
            raise SuiteUnavailableError(name, exc) from exc

        self._open[name] = runtime
        return runtime

    def close(self) -> None:
        self._open.clear()
        self._stack.close()

    def __enter__(self) -> SuiteSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
