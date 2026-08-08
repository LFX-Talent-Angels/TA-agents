"""Locate (Resolve): free text -> suite node candidates + confidence.

Resolution order (owned by the suite tool, not this skill): exact match,
alias/label match, case-insensitive/contains fallback. Confidence attaches to
every result and crosses all later steps.
"""

from talent_angels.skills.locate.esco import ESCO_SUITE_NAME, open_esco_suite
from talent_angels.skills.locate.resolve import locate

__all__ = ["ESCO_SUITE_NAME", "locate", "open_esco_suite"]
