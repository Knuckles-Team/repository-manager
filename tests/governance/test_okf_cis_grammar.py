"""OKF-CIS grammar + vocab tests (CONCEPT:MERISTEM-OS.governance.concept-2).

Covers id<->path<->IRI round-trip, the >=2-segment rule, strict rejection of
malformed ids, the shared-pillar federation invariant, the
closed domain vocab, and the 2-letter slug registry uniqueness.
"""

from __future__ import annotations

import pytest

from repository_manager.governance import concept_hierarchy as ch


@pytest.mark.parametrize(
    "cid,path",
    [
        ("MERISTEM-KG.ingest.entropy-dedup", "MERISTEM/KG/ingest/entropy-dedup"),
        ("MEDULLA-KG.storage.redb", "MEDULLA/KG/storage/redb"),
        ("DS-AHE.trainer.gpu-slot", "DS/AHE/trainer/gpu-slot"),
        (
            "MERISTEM-OS.governance.concept-allocator.v2",
            "MERISTEM/OS/governance/concept-allocator/v2",
        ),
    ],
)
def test_id_path_iri_roundtrip(cid: str, path: str) -> None:
    parsed = ch.parse_okf_id(cid)
    assert parsed.canonical == cid
    assert parsed.path == path
    assert ch.okf_id_to_path(cid) == path
    assert ch.path_to_okf_id(path) == cid
    assert ch.path_to_okf_id(path + ".md") == cid  # .md suffix tolerated
    assert parsed.iri == f"{ch.CONCEPT_IRI_BASE}/{path}"
    assert ch.concept_iri(cid) == parsed.iri


def test_shared_pillar_federates_across_repos() -> None:
    # Two repos, same pillar -> the SAME pillar IRI (no slug): cross-repo linking.
    au = ch.parse_okf_id("MERISTEM-KG.ingest.a")
    eg = ch.parse_okf_id("MEDULLA-KG.storage.b")
    assert au.pillar_iri == eg.pillar_iri == f"{ch.PILLAR_IRI_BASE}/KG"
    assert au.scheme_iri != eg.scheme_iri  # but distinct owning schemes


@pytest.mark.parametrize(
    "bad",
    [
        "MERISTEM-KG.ingest",  # only 1 segment -> a domain node, not a concept
        "KG-2.101",
        "A-KG.a.b",  # slug under 2 letters
        "MERISTEM-XX.a.b",  # pillar not in the closed set
        "MERISTEM-KG.A.b",  # uppercase segment
        "au-kg.a.b",  # lowercase slug/pillar
        "MERISTEM-KG..b",  # empty segment
    ],
)
def test_rejects_malformed(bad: str) -> None:
    assert not ch.is_okf_id(bad)
    with pytest.raises(ValueError):
        ch.parse_okf_id(bad)


def test_domain_vocab_closed_and_covers_all_pillars() -> None:
    vocab = ch.load_domain_vocab()
    assert set(vocab) == set(ch.PILLARS)
    assert ch.is_valid_domain("KG", "ingest")
    assert not ch.is_valid_domain("KG", "not-a-real-domain")
    # every listed domain is a valid concept-slug segment shape
    for pillar, domains in vocab.items():
        for dom in domains:
            assert ch.is_okf_id(f"MERISTEM-{pillar}.{dom}.x"), (pillar, dom)


def test_slug_registry_unique_and_uppercase() -> None:
    reg = ch.load_slug_registry()
    assert ch.slug_for_repo("scion-sdk") == "SCION"
    assert ch.slug_for_repo("epistemic-graph") == "MEDULLA"
    assert len(set(reg.values())) == len(reg)  # globally unique
    assert all(len(s) >= 2 and s.isupper() for s in reg.values())


def test_part_of_edges_chain_to_shared_pillar() -> None:
    parsed = [
        ch.parse_okf_id("MERISTEM-KG.ingest.dedup"),
        ch.parse_okf_id("MEDULLA-KG.storage.redb"),
    ]
    edges = dict(ch.okf_part_of_edges(parsed))
    # concept -> domain -> pillar; both repos' KG concepts reach one pillar node
    assert edges[parsed[0].iri] == parsed[0].domain_iri
    assert edges[parsed[0].domain_iri] == parsed[0].pillar_iri
    assert parsed[0].pillar_iri == parsed[1].pillar_iri


@pytest.mark.parametrize(
    "tail",
    [
        "/2.15/2.34",
        "_legacy_suffix",
        "UppercaseSuffix",
    ],
)
def test_marker_parser_preserves_adjacent_tail_semantics(tail: str) -> None:
    text = f"CONCEPT:MERISTEM-KG.ingest.entropy-dedup{tail}) — Durable description"

    marker = next(ch.iter_okf_markers(text))

    assert marker.id == "MERISTEM-KG.ingest.entropy-dedup"
    assert marker.tail == tail
    assert marker.raw == f"CONCEPT:{marker.id}{tail}"
    assert text[marker.end :] == ") — Durable description"


def test_marker_regex_findall_keeps_the_id_only_contract() -> None:
    text = "CONCEPT:MERISTEM-KG.ingest.entropy-dedup/2.15"

    assert ch.OKF_MARKER_RE.findall(text) == ["MERISTEM-KG.ingest.entropy-dedup"]
