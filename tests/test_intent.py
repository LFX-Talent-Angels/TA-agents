"""Keyword intent routing (the no-LLM planner and the tool-loop fallback)."""


def test_keyword_inside_a_word_does_not_route_to_pathfind() -> None:
    """Live repro: "Singapore" contains "gap" and was refused as a Pathfind question."""
    from talent_angels.assistant.intent import classify_capability

    assert classify_capability("software developer in Singapore") == "locate"
    assert classify_capability("skill gaps from nurse to doctor") == "pathfind"
    assert classify_capability("list the essential skills of a nurse") == "connect"
