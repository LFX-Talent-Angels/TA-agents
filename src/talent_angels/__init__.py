"""Talent Angels — a suite of AI Graph Agents over skill/task/occupation taxonomies.

Core agents:
    - locator    : pinpoint a node from natural language
    - connector  : neighbors of a node
    - pathfinder : routes between two nodes

Supporting layers:
    - graph      : knowledge graph model + Graph-RAG retrieval
    - taxonomies : load & normalize ESCO, O*NET, SFIA, BLS, Lightcast
"""

__version__ = "0.0.1"
