"""Concorrencia HTTP sem corrida no ledger ou no limite monetario."""

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    return importlib.import_module("scripts.experiments.benchmark_paid_rates")


def state(tmp_path, benchmark, budget="1"):
    rows = [
        dict(
            doc_id="doc",
            arm=str(i),
            messages=[{"role": "user", "content": str(i)}],
            prompt_tokens=10,
            max_output_tokens=8192,
        )
        for i in range(6)
    ]
    return SimpleNamespace(
        folder=tmp_path / "qwen",
        identity={"fixed": True},
        rows=rows,
        model="qwen37",
        budget=benchmark.money(budget),
    )


class Client:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def _request(self, route, payload):
        self.calls += 1
        time.sleep(0.01)
        return {
            "model": "qwen/qwen3.7-flash",
            "provider": "Alibaba",
            "usage": {
                "cost": 0.0001,
                "prompt_tokens": 11 if self.fail else 10,
                "completion_tokens": 1,
            },
            "choices": [{"finish_reason": "stop", "message": {"content": "summary"}}],
        }


def test_parallel_ledger_no_duplicates_and_resume_skips(benchmark, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "dispatch_delay", lambda *args: 0)
    s = state(tmp_path, benchmark)
    client = Client()
    r = benchmark.stage(s, list(range(6)), 500, client, tmp_path, max_workers=3)
    assert r["accepted_cases"] == 6 and 2 <= r["peak_in_flight"] <= 3
    ledger = json.loads((s.folder / "ledger.json").read_text())
    assert len(ledger["attempts"]) == len({a["case"] for a in ledger["attempts"]}) == 6
    assert all(a["status"] == "accepted" for a in ledger["attempts"])
    benchmark.stage(s, list(range(6)), 500, client, tmp_path, max_workers=3)
    assert client.calls == 6


def test_inflight_reservations_cannot_overcommit_budget(benchmark, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "dispatch_delay", lambda *args: 0)
    s = state(tmp_path, benchmark, budget=".008192")
    client = Client()
    r = benchmark.stage(s, list(range(6)), 500, client, tmp_path, max_workers=3)
    assert client.calls == 1 and r["stop_reason"] == "budget_limit"
    assert r["accepted_cases"] == 1


def test_token_mismatch_stops_dispatch_and_requires_audit(benchmark, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "dispatch_delay", lambda *args: 0)
    s = state(tmp_path, benchmark)
    client = Client(fail=True)
    r = benchmark.stage(s, list(range(6)), 500, client, tmp_path, max_workers=1)
    assert client.calls == 1 and r["stop_reason"] == "audit_required"
    with pytest.raises(ValueError, match="auditoria"):
        benchmark.stage(s, list(range(6)), 500, client, tmp_path, max_workers=1)
    assert client.calls == 1


def test_spacing_and_rolling_window(benchmark):
    assert benchmark.dispatch_delay([10], 500, 10) == pytest.approx(0.12)
    assert benchmark.dispatch_delay([10] * 500, 500, 11) == 59
    assert benchmark.dispatch_delay([10], 500, 71) == 0


def test_lower_sweep_uses_only_new_cases_in_first_batch(benchmark, tmp_path, monkeypatch):
    s = state(tmp_path, benchmark)
    s.output = tmp_path / "run"
    s.records = []
    seen = set(range(43))
    s.accepted = lambda: seen.copy()
    batches = []

    def fake_stage(state, indices, rpm, client, folder):
        assert not seen.intersection(indices)
        seen.update(indices)
        batches.append((rpm, indices))
        return {"stop_reason": None, "accepted_cases": len(indices)}

    monkeypatch.setattr(benchmark, "stage", fake_stage)
    monkeypatch.setattr(benchmark, "export_progress", lambda *args: None)
    benchmark.model_sweep(s, None, tmp_path, lower_rates=True)
    assert [rpm for rpm, _ in batches] == [100, 200, 300, 400]
    assert seen == set(range(443))
    assert all(len(indices) == 100 for _, indices in batches)
