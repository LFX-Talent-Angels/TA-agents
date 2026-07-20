"""Locate (Resolve): free text -> suite node candidates + confidence.

Resolution order: exact match, alias/label match, vector fallback.
Confidence attaches to every result and crosses all later steps.
Skeleton.
"""
