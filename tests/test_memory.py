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


def test_node_id_str_strips_doubled_suite_prefix():
    from talent_angels.contracts.models import NodeRef
    from talent_angels.memory import profile as prof

    # graph-id-shaped source_id (rec["id"] fallback in the suite tool)
    node = NodeRef(
        id="onet:occupation:51-4121.00",
        suite="onet",
        source="onet",
        source_id="onet:occupation:51-4121.00",
        kind="Occupation",
        pref_label="Welders",
    )
    assert prof._node_id_str(node) == "onet:51-4121.00"


def test_node_id_str_leaf_http_uri():
    from talent_angels.contracts.models import NodeRef
    from talent_angels.memory import profile as prof

    node = NodeRef(
        id="esco:occupation:abc",
        suite="esco",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/abc",
        kind="Occupation",
        pref_label="Nurse",
    )
    assert prof._node_id_str(node) == "esco:abc"


def test_write_standing_skips_blank_label(tmp_path):
    from talent_angels.memory import profile as prof

    with patch.object(prof, "USER_MD", tmp_path / "USER.md"):
        good = _make_node("onet", "15-1252.00", "Software Developers")
        prof.write_standing(good)
        blank = _make_node("onet", "51-4121.00", "")
        prof.write_standing(blank)

        content = (tmp_path / "USER.md").read_text()
        assert "Software Developers" in content
        assert "51-4121" not in content


def test_user_profile_capped_and_consolidated(tmp_path):
    from talent_angels.memory import profile as prof

    with (
        patch.object(prof, "USER_MD", tmp_path / "USER.md"),
        patch.object(prof, "USER_MAX_CHARS", 300),
    ):
        for i in range(20):
            prof.write_rejected(_make_node("esco", f"id-{i}", f"Rejected option {i}"))
        content = (tmp_path / "USER.md").read_text()
        assert len(content) <= 300

        # oldest rejected entries evicted, newest kept
        assert "Rejected option 0" not in content
        assert "Rejected option 19" in content


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
