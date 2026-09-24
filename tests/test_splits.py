"""Testes da particao dos conjuntos experimentais.

As invariantes testadas aqui sao as que sustentam a validade do experimento: uma
empresa em um unico conjunto, um relatorio por empresa, e particao congelada e
reproduzivel.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from findsum_rag.data import Task
from findsum_rag.splits import (
    DocumentRef,
    SplitManifest,
    build_manifest,
    load_set,
    pick_one_per_company,
    scan_pool,
)


def ref(company: str, report: str, split: str = "train", row: int = 0) -> DocumentRef:
    return DocumentRef(
        split=split, row=row, doc_id=f"{company}-{report}", stock_name=company, report_id=report
    )


def test_year_extracted_from_accession_number():
    assert ref("GNK", "0001558370-17-002200").year == 2017
    assert ref("AAA", "0000000000-04-000001").year == 2004
    # Anos de 1900: o FINDSum nao tem, mas o formato de 2 digitos permite.
    assert ref("AAA", "0000000000-98-000001").year == 1998


def test_year_is_none_for_malformed_accession():
    assert ref("AAA", "nao-e-accession").year is None


def test_pick_one_per_company_keeps_the_most_recent():
    refs = [
        ref("AAA", "0000000000-15-000001"),
        ref("AAA", "0000000000-21-000002"),
        ref("AAA", "0000000000-18-000003"),
        ref("BBB", "0000000000-09-000004"),
    ]
    chosen = pick_one_per_company(refs)
    assert len(chosen) == 2
    by_company = {c.stock_name: c for c in chosen}
    assert by_company["AAA"].year == 2021
    assert by_company["BBB"].year == 2009


def test_pick_one_per_company_is_deterministic():
    refs = [ref("AAA", f"0000000000-1{i}-00000{i}") for i in range(5)]
    assert pick_one_per_company(refs) == pick_one_per_company(list(reversed(refs)))


def test_manifest_rejects_company_in_two_sets():
    with pytest.raises(ValidationError, match="em eval e em"):
        SplitManifest(
            task=Task.LIQUIDITY,
            seed=1,
            created_at="2026-01-01T00:00:00Z",
            min_summary_words=1,
            sets={
                "examples": [ref("AAA", "0000000000-15-000001")],
                "eval": [ref("AAA", "0000000000-21-000002")],
            },
        )


def test_manifest_rejects_duplicate_document():
    duplicate = ref("AAA", "0000000000-15-000001")
    with pytest.raises(ValidationError, match="repetido"):
        SplitManifest(
            task=Task.LIQUIDITY,
            seed=1,
            created_at="2026-01-01T00:00:00Z",
            min_summary_words=1,
            sets={"eval": [duplicate, duplicate]},
        )


def test_scan_pool_finds_documents(fake_root: Path):
    refs = scan_pool(fake_root, Task.LIQUIDITY, min_summary_words=1)
    # 2 documentos x 3 splits sinteticos.
    assert len(refs) == 6
    assert {r.stock_name for r in refs} == {"AAA", "BBB"}
    assert all(r.split in ("train", "val", "test") for r in refs)


def test_scan_pool_drops_short_summaries(fake_root: Path):
    # Os resumos sinteticos tem ~10 palavras; um piso alto descarta tudo.
    assert scan_pool(fake_root, Task.LIQUIDITY, min_summary_words=1000) == []


def test_scan_pool_requires_downloaded_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="fetch_findsum"):
        scan_pool(tmp_path, Task.LIQUIDITY)


def test_build_manifest_partitions_without_overlap(fake_root: Path):
    manifest = build_manifest(
        fake_root, Task.LIQUIDITY, sizes={"examples": 1, "eval": 1}, min_summary_words=1
    )
    assert manifest.sizes() == {"examples": 1, "eval": 1}
    companies = [r.stock_name for refs in manifest.sets.values() for r in refs]
    assert sorted(companies) == ["AAA", "BBB"]


def test_build_manifest_is_reproducible_for_a_seed(fake_root: Path):
    def ids(seed: int) -> dict:
        m = build_manifest(
            fake_root,
            Task.LIQUIDITY,
            sizes={"examples": 1, "eval": 1},
            seed=seed,
            min_summary_words=1,
        )
        return {k: [r.doc_id for r in v] for k, v in m.sets.items()}

    assert ids(7) == ids(7)


def test_build_manifest_rejects_impossible_sizes(fake_root: Path):
    with pytest.raises(ValueError, match="pede 99 documentos"):
        build_manifest(
            fake_root, Task.LIQUIDITY, sizes={"eval": 99}, min_summary_words=1
        )


def test_manifest_roundtrip(fake_root: Path, tmp_path: Path):
    original = build_manifest(
        fake_root, Task.LIQUIDITY, sizes={"examples": 1, "eval": 1}, min_summary_words=1
    )
    path = tmp_path / "m.json"
    original.save(path)
    loaded = SplitManifest.load(path)
    assert loaded.task == original.task
    assert loaded.sizes() == original.sizes()
    assert loaded.sets["eval"][0].doc_id == original.sets["eval"][0].doc_id


def test_load_set_returns_documents_in_manifest_order(fake_root: Path, fake_manifest):
    manifest, _ = fake_manifest
    documents = load_set(fake_root, manifest, "eval")
    assert [d.doc_id for d in documents] == [r.doc_id for r in manifest.sets["eval"]]
    assert all(d.summary.strip() for d in documents)


def test_load_set_respects_limit(fake_root: Path, fake_manifest):
    manifest, _ = fake_manifest
    assert len(load_set(fake_root, manifest, "eval", limit=1)) == 1


def test_load_set_rejects_unknown_set(fake_root: Path, fake_manifest):
    manifest, _ = fake_manifest
    with pytest.raises(KeyError, match="inexistente"):
        load_set(fake_root, manifest, "inexistente")
