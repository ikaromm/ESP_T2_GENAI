"""Lotes do full sem chamadas reais."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from findsum_rag.cli import app


@pytest.fixture
def batcher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_ling_batches")


def test_ten_batches_cover_same_thousand_once(batcher):
    records = [{"doc_id": str(i)} for i in range(1000)]
    rows = [{"doc_id": r["doc_id"], "arm": a} for r in records for a in batcher.ARMS]
    groups = batcher.batch_indices(records, rows)
    assert [len(g) for g in groups] == [600] * 10
    assert [index for group in groups for index in group] == list(range(6000))
    assert batcher.next_batch(groups, set(range(599))) == 1
    assert batcher.next_batch(groups, set(range(600))) == 2
    assert batcher.next_batch(groups, set(range(6000))) is None
    rows.reverse()
    with pytest.raises(ValueError, match="ordem"):
        batcher.batch_indices(records, rows)


def test_cli_does_not_execute_by_default(monkeypatch):
    seen = []
    monkeypatch.setattr("findsum_rag.cli._script", lambda name, args: seen.append((name, args)))
    assert CliRunner().invoke(app, ["ling-batch"]).exit_code == 0
    assert seen[0][0] == "run_ling_batches.py"
    assert "--execute" not in seen[0][1]
    assert CliRunner().invoke(app, ["ling-batch", "--batch", "11"]).exit_code != 0


def test_batch_resume_keeps_global_indices_and_no_duplicates(batcher, tmp_path):
    core = importlib.import_module("run_prepared_paid")
    rows = [
        {"doc_id": str(i // 6), "arm": batcher.ARMS[i % 6], "messages": [], "prompt_tokens": 100}
        for i in range(12)
    ]
    calls = []

    def request(*args):
        calls.append(args)
        return {
            "model": core.TARGETS["ling-free"][0],
            "provider": "Novita",
            "choices": [{"message": {"content": "Summary"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "cost": 0},
        }

    transport = SimpleNamespace(_request=request)
    for indices in [set(range(6)), set(range(6)), set(range(6, 12))]:
        core.run(
            rows,
            core.TARGETS["ling-free"],
            tmp_path,
            core.money(0),
            {},
            transport,
            case_indices=indices,
            max_new_calls=6,
        )
    assert len(calls) == 12
    assert batcher.accepted_cases(tmp_path, rows, {}) == set(range(12))
    path = next(tmp_path.glob("*.request.json"))
    path.write_text("{}")
    with pytest.raises(ValueError, match="prompt congelado"):
        batcher.accepted_cases(tmp_path, rows, {})


def test_transient_retry_circuit_can_resume_after_cooldown(batcher, tmp_path, monkeypatch):
    core = importlib.import_module("run_prepared_paid")
    rows = [{"doc_id": "a", "arm": "C1", "messages": [], "prompt_tokens": 100}]
    attempts = [
        {"case": 0, "doc_id": "a", "arm": "C1", "at": 100, "status": "http429", "reserved_usd": "0"}
        for _ in range(6)
    ]
    (tmp_path / "ledger.json").write_text(json.dumps({"identity": {}, "attempts": attempts}))

    def request(*args):
        return {
            "model": core.TARGETS["ling-free"][0],
            "provider": "Novita",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1, "cost": 0},
        }

    client = SimpleNamespace(_request=request)
    monkeypatch.setattr(core.time, "time", lambda: 101)
    with pytest.raises(ValueError, match="circuito aberto"):
        core.run(
            rows,
            core.TARGETS["ling-free"],
            tmp_path,
            core.money(0),
            {},
            client,
            retry_reset_after=3600,
        )
    monkeypatch.setattr(core.time, "time", lambda: 3701)
    ledger = core.run(
        rows, core.TARGETS["ling-free"], tmp_path, core.money(0), {}, client, retry_reset_after=3600
    )
    assert len(ledger["attempts"]) == 7
    assert ledger["attempts"][-1]["response_file"] == "0000-6.response.json"
