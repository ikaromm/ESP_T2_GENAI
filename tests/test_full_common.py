"""Amostra reservada, rastreabilidade e bloqueio de coorte incompleta; sem rede."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("full_common")


def test_random_selection_is_reproducible_and_preserves_full_membership(common):
    records = [{"doc_id": f"d{i:04}"} for i in range(1000)]
    selected = common.random_records(records, 1000, seed=42, split="eval")
    assert selected == common.random_records(records[::-1], 1000, seed=42, split="eval")
    assert {r["doc_id"] for r in selected} == {r["doc_id"] for r in records}
    assert selected != common.random_records(records, 1000, seed=43, split="eval")
    assert records[0]["doc_id"] == "d0000"


def test_saved_selection_verifies_seed_ids_and_manifest(common, tmp_path):
    records = [{"doc_id": f"d{i}"} for i in range(12)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sets": {"dev": records}}))
    selected = common.random_records(records, 5, seed=42, split="dev")
    payload = {
        "split": "dev",
        "manifest_path": str(manifest),
        "manifest_sha256": common.sha(manifest),
        "documents": selected,
        "sampling": {"method": "random_without_replacement", "seed": 42},
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload))
    assert common.selected_records(tmp_path) == ("dev", selected)
    payload["documents"] = selected[::-1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifesto reservado"):
        common.selected_records(tmp_path)


def test_full_eligibility_does_not_silently_reduce_1000(common):
    ids = [f"d{i}" for i in range(1000)]
    rows = [
        {"doc_id": d, "arm": a, "prompt_tokens": 100, "context_tokens": 20, "context": "x"}
        for d in ids
        for a in common.ARMS
    ]
    by_model = {m: [dict(r) for r in rows] for m in common.FULL_MODELS}
    limits = {m: common.Limits() for m in common.FULL_MODELS}
    assert common.eligibility(ids, by_model, limits)["ready"] is True
    by_model["qwen37"][-1]["prompt_tokens"] = 999999
    report = common.eligibility(ids, by_model, limits)
    assert report["ready"] is False
    assert report["eligible_count"] == 999
    assert report["excluded"][0]["doc_id"] == ids[-1]
    with pytest.raises(ValueError, match="esperados 1000"):
        common.eligibility(ids[:-1], by_model, limits)


def test_input_that_fits_alone_but_overflows_with_output_is_rejected(common):
    limits = common.Limits(max_input=100, max_output=30, context_length=120, margin=5)
    limits.check(85)
    with pytest.raises(ValueError, match="janela"):
        limits.check(86)
