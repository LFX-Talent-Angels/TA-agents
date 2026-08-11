"""Typed contracts (Pydantic v2) that cross every component boundary.

AgentResult: capability, suite, nodes, edges, evidence, confidence, warnings.
Node IDs are suite-scoped; every node carries source + source_id; evidence is
a pointer, not a payload. Prose never crosses boundaries.
"""

from talent_angels.contracts.models import (
    AgentResult,
    EdgeRef,
    EvidencePointer,
    NodeRef,
)

__all__ = ["AgentResult", "EdgeRef", "EvidencePointer", "NodeRef"]
