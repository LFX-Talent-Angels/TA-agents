"""Map-work skills: passive procedures + the tool calls they drive.

Skills are loaded on demand by the assistant's plan. They do not own the
user's goal and never talk to the user. Determinism is pushed down into
tools/code; see ARCHITECTURE.md ("Skills").
"""
