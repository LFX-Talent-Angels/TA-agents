"""The engine's front door as MCP tools: `capabilities`, `run`, `start_job`, `job_status`,
`cancel_job`.

Kept out of `server.py` on purpose: this module has no `from __future__ import annotations`, so
the SDK can resolve the real `Result` / `JobState` classes in the tool signatures, and it is
imported lazily by `build_server()` (the engine imports `mcp.models`, so a top-level import
would be an import cycle).
"""

from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from talent_angels.engine import Result, manifest, run
from talent_angels.engine.jobs import (
    JobBusy,
    JobRejected,
    JobState,
    cancel_job,
    job_status,
    start_job,
)
from talent_angels.mcp.session import SuiteSession

ToolContext = Context[SuiteSession, Any]


def add_engine_tools(server: MCPServer[SuiteSession]) -> None:
    @server.tool(
        name="capabilities",
        title="What this engine can do",
        description=(
            "The capability manifest: for each capability its name, the question it answers, "
            "input and output JSON Schema, cost, determinism, whether it uses a model, and "
            "examples. Pass `name` (a capability name or tool name) for one entry."
        ),
    )
    def capabilities_tool(
        name: Annotated[str | None, Field(description="One capability; omit for all.")] = None,
    ) -> dict[str, Any]:
        return manifest(only=name)

    @server.tool(
        name="run",
        title="Run one capability",
        description=(
            "Run any capability by name with its input (see `capabilities`). Returns the status "
            "(ok, ambiguous, no_result, unavailable, invalid_input, unknown_suite, "
            "unknown_capability, use_job, error), the output, warnings, candidates and the "
            "resolved call. Deterministic; no model call. Unknown names and fields are refused. "
            "`tag` is long-running: use `start_job`."
        ),
    )
    def run_tool(
        ctx: ToolContext,
        capability: Annotated[str, Field(description="A capability name from `capabilities`.")],
        input: Annotated[  # noqa: A002 - the manifest's own word for it
            dict[str, Any] | None, Field(description="The capability's input object.")
        ] = None,
    ) -> Result:
        return run(capability, input or {}, registry=ctx.request_context.lifespan_context.registry)

    @server.tool(
        name="start_job",
        title="Start a long capability as a job",
        description=(
            "Start a long-running capability (`tag`) in the background and return its job state "
            "at once; poll `job_status`. `tag` reads `path` under the server's tag root and "
            "searches without a model unless the server allows one (TA_JOBS_ALLOW_MODEL=1)."
        ),
    )
    def start_job_tool(
        ctx: ToolContext,
        capability: Annotated[str, Field(description="A capability name, e.g. `tag`.")],
        input: Annotated[  # noqa: A002
            dict[str, Any], Field(description="The capability's input object.")
        ],
    ) -> JobState:
        try:
            return start_job(
                capability, input, registry=ctx.request_context.lifespan_context.registry
            )
        except JobRejected as exc:
            return JobState(id="", capability=capability[:64], status="failed", result=exc.result)
        except JobBusy:
            busy = Result(status="error", error="a job is already running in this process")
            return JobState(id="", capability=capability[:64], status="failed", result=busy)

    @server.tool(
        name="job_status",
        title="State and result of a job",
        description=(
            "A job's status (running, done, failed, cancelled, interrupted, not_found), its "
            "progress lines and step, and, once finished, the `run` result."
        ),
    )
    def job_status_tool(job_id: Annotated[str, Field(description="From `start_job`.")]) -> JobState:
        return job_status(job_id)

    @server.tool(
        name="cancel_job",
        title="Cancel a job",
        description=(
            "Ask a running job to stop at its next progress tick. Harmless twice or when the job "
            "has finished. Work done so far stays on disk, and a rerun resumes from it."
        ),
    )
    def cancel_job_tool(job_id: Annotated[str, Field(description="From `start_job`.")]) -> JobState:
        return cancel_job(job_id)
