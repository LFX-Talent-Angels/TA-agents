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


def test_bls_property_links_an_onet_code_that_is_in_the_map() -> None:
    props = {"bls:occupation:15-1252": {"onet_soc_codes": ["15-1252.00", "15-1252.01"]}}
    assert [
        x.to_id for x in links_from("bls:occupation:15-1252", props["bls:occupation:15-1252"])
    ] == [
        "onet:occupation:15-1252.00",
        "onet:occupation:15-1252.01",
    ]
    assert len(crosswalk_links({**props, "onet:occupation:15-1252.01": {}})) == 1


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
