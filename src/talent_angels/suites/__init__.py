"""Taxonomy-suite selection boundary for the assistant runtime."""

from talent_angels.suites.registry import (
    SuiteFactory,
    SuiteRegistry,
    SuiteRuntime,
    UnknownSuiteError,
    default_suite_registry,
)

__all__ = [
    "SuiteFactory",
    "SuiteRegistry",
    "SuiteRuntime",
    "UnknownSuiteError",
    "default_suite_registry",
]
