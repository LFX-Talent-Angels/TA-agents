"""Tests for talent_angels.memory.profile — write_standing keyed by suite."""

from unittest.mock import patch


def _make_node(suite: str, source_id: str, pref_label: str):
    from talent_angels.contracts.models import NodeRef

    return NodeRef(
        id=f"{suite}:occupation:{source_id}",
        suite=suite,
        source=suite,
        source_id=source_id,
        kind="Occupation",
        pref_label=pref_label,
    )


def test_write_standing_keyed_by_suite(tmp_path):
    from talent_angels.memory import profile as prof

    with patch.object(prof, "USER_MD", tmp_path / "USER.md"):
        onet_node = _make_node("onet", "15-1252.00", "Software Developers")
        esco_node = _make_node(
            "esco",
            "http://data.europa.eu/esco/occupation/123",
            "Software developer",
        )

        prof.write_standing(onet_node)
        prof.write_standing(esco_node)

        content = (tmp_path / "USER.md").read_text()
        assert "STANDING[onet]:" in content
        assert "STANDING[esco]:" in content
        lines = [line for line in content.splitlines() if "STANDING" in line]
        assert len(lines) == 2


def test_write_standing_updates_same_suite(tmp_path):
    from talent_angels.memory import profile as prof

    with patch.object(prof, "USER_MD", tmp_path / "USER.md"):
        node_v1 = _make_node("onet", "15-1252.00", "Software Developers")
        node_v2 = _make_node("onet", "15-1253.00", "Web Developers")

        prof.write_standing(node_v1)
        prof.write_standing(node_v2)

        content = (tmp_path / "USER.md").read_text()
        lines = [line for line in content.splitlines() if "STANDING" in line]
        assert len(lines) == 1
        assert "Web Developers" in lines[0]
