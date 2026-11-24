"""Official crosswalk rules only: a link needs a code; no code, no link."""

from __future__ import annotations

from talent_angels.skills.connect.crosswalk import crosswalk_links, links_from

URI = "http://data.europa.eu/esco/occupation/18e14e61-495b-44cc-a7c6-df4c625934ba"
ESCO = "esco:occupation:18e14e61-495b-44cc-a7c6-df4c625934ba"


def test_bls_and_onet_link_by_soc_code_from_either_side() -> None:
    onet = {"onet:occupation:15-1252.00": {}, "bls:occupation:15-1252": {}}
    [link] = crosswalk_links(onet)
    assert (link.from_id, link.to_id, link.strength) == (
        "bls:occupation:15-1252",
        "onet:occupation:15-1252.00",
        "soc",
    )
    assert link.to_dict()["source"] == "crosswalk:bls-onet-soc"


def test_bls_links_only_the_same_soc_dot_00_and_ignores_borrowed_codes() -> None:
    # onet_soc_codes lists O*NET jobs BLS borrowed skill scores from, not equivalents
    # (live: Engineers, All Other -> Aerospace Engineers). Only the .00 code is "same".
    props = {"onet_soc_codes": ["17-2011.00", "17-2199.01"]}
    assert [x.to_id for x in links_from("bls:occupation:17-2199", props)] == [
        "onet:occupation:17-2199.00"
    ]
    nodes = {"bls:occupation:17-2199": props, "onet:occupation:17-2011.00": {}}
    assert crosswalk_links(nodes) == []


def test_onet_sub_occupations_are_not_the_same_job() -> None:
    nodes = {"bls:occupation:11-1011": {}, "onet:occupation:11-1011.03": {}}
    assert crosswalk_links(nodes) == []
    assert len(crosswalk_links({**nodes, "onet:occupation:11-1011.00": {}})) == 1


def test_different_soc_codes_do_not_link() -> None:
    assert crosswalk_links({"bls:occupation:15-1252": {}, "onet:occupation:29-1141.00": {}}) == []


def test_jobtech_esco_exact_and_close_and_nothing_without_a_uri() -> None:
    props = {"jobtech:occupation:a": {"esco_exact": [URI]}, ESCO: {}}
    [link] = crosswalk_links(props)
    assert (link.to_id, link.strength) == (ESCO, "exact")
    close = {"jobtech:occupation:a": {"esco_close": [URI]}, ESCO: {}}
    assert crosswalk_links(close)[0].strength == "close"
    assert crosswalk_links({"jobtech:occupation:a": {"esco_exact": []}, ESCO: {}}) == []
    assert crosswalk_links({"jobtech:occupation:a": {}, ESCO: {}}) == []


def test_esco_and_onet_never_link_and_broad_narrow_are_ignored() -> None:
    assert crosswalk_links({ESCO: {}, "onet:occupation:15-1252.00": {}}) == []
    broad = {"jobtech:occupation:a": {"esco_broad": [URI]}, ESCO: {}}
    assert crosswalk_links(broad) == []
