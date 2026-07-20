"""The main assistant: the only piece that talks to the user.

Owns intent, the plan (L / L+C / L+C+P / ...+E), suite selection, merging
across suites, honesty rules, and the final answer. Implemented as a LangGraph
stateful loop. Skeleton — see ARCHITECTURE.md.
"""
