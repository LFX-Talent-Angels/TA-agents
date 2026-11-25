"""The engine: one declared set of capabilities that every interface plugs into.

`CAPABILITIES` is the single declaration (name, question, input and output models, determinism,
cost, aliases per interface). `run()` is the one entry point, `manifest()` the machine-readable
description, and `engine.jobs` runs long capabilities (`tag`) as jobs.
"""

from talent_angels.engine.registry import (
    CAPABILITIES,
    ENGINE_VERSION,
    STATUSES,
    Call,
    CapabilitySpec,
    Result,
    RunContext,
    get_capability,
    manifest,
    run,
)

__all__ = [
    "CAPABILITIES",
    "ENGINE_VERSION",
    "STATUSES",
    "Call",
    "CapabilitySpec",
    "Result",
    "RunContext",
    "get_capability",
    "manifest",
    "run",
]
