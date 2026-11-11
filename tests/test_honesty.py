from talent_angels.assistant.honesty import honesty_warnings
from talent_angels.contracts import AgentResult, NodeRef


def test_honesty_flags_crossed_suite_ids() -> None:
    bad = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[
            NodeRef(
                id="onet:occupation:1",
                suite="onet",
                source="onet",
                source_id="1",
                kind="Occupation",
                pref_label="x",
            )
        ],
    )
    warnings = honesty_warnings((bad,))
    assert any(item.startswith("honesty:") for item in warnings)


def test_honesty_accepts_suite_scoped_ids() -> None:
    ok = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[
            NodeRef(
                id="esco:occupation:1",
                suite="esco",
                source="esco",
                source_id="1",
                kind="Occupation",
                pref_label="x",
            )
        ],
    )
    assert honesty_warnings((ok,)) == []
