"""Connect (Reveal): neighbors and hierarchy around a resolved node.

Single-hop traversal over relationship types the suite marks traversable.
Deterministic — one graph hop via the taxonomy suite contract.
"""

from talent_angels.skills.connect.models import ConnectRequest
from talent_angels.skills.connect.reveal import ConnectableSuite, connect

__all__ = ["ConnectRequest", "ConnectableSuite", "connect"]
