"""Painel incremental: cobertura emparelhada, cache e Bash sem API."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("update_progress_metrics")


def states_for(metrics, tmp_path, count=2):
    docs = [{"doc_id": str(i), "reference": "cash increased"} for i in range(count)]
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "documents.json").write_text(json.dumps(docs))
    states = {}
    for model in metrics.MODELS:
        output = tmp_path / model
        folder = output / model
        folder.mkdir(parents=True)
        entries = []
        for index in range(count * 6):
            response = f"{index}.json"
            (folder / response).write_text(
                json.dumps(
                    {
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": "cash increased"}}
                        ]
                    }
                )
            )
            entries.append(
                dict(
                    case=index,
                    doc_id=str(index // 6),
                    arm=metrics.ARMS[index % 6],
                    status="accepted",
                    response_file=response,
                )
            )
        (folder / "ledger.json").write_text(json.dumps({"attempts": entries}))
        states[model] = SimpleNamespace(
            prepared=prepared, output=output, folder=folder, accepted=lambda: set(range(count * 6))
        )
    return docs, states


def fake_scoring(metrics, monkeypatch):
    calls = []

    def score(texts, refs, **kwargs):
        calls.append(len(texts))
        return [SimpleNamespace(precision=0.7, recall=0.5, f1=0.6) for _ in texts]

    monkeypatch.setattr(metrics, "bertscore_components", score)
    monkeypatch.setattr(
        metrics,
        "bertscore_token_audit",
        lambda texts: [dict(full_tokens=3, evaluated_tokens=3, truncated=False) for _ in texts],
    )
    monkeypatch.setattr(metrics, "ensure_meteor_resources", lambda: None)
    monkeypatch.setattr(metrics, "meteor", lambda *_: 0.2)
    return calls


def test_missing_early_case_does_not_select_later_successes(metrics):
    records = [{"doc_id": str(i)} for i in range(3)]
    accepted = {m: set(range(18)) for m in metrics.MODELS}
    accepted["gemma26"].remove(7)
    assert metrics.common_prefix(records, accepted) == 1
    accepted["gemma26"].add(7)
    assert metrics.common_prefix(records, accepted) == 3


def test_cache_reuses_scores_and_invalidates_changed_text(metrics, tmp_path, monkeypatch):
    records, states = states_for(metrics, tmp_path)
    calls = fake_scoring(metrics, monkeypatch)
    out, cache = tmp_path / "result", tmp_path / "cache.json"
    report = metrics.update_metrics(records, states, out=out, cache=cache)
    assert report["documents_compared"] == 2 and calls == [36]
    before = (out / "scores.csv").read_bytes()
    metrics.update_metrics(records, states, out=out, cache=cache)
    assert calls == [36] and before == (out / "scores.csv").read_bytes()
    path = states["qwen37"].folder / "0.json"
    response = json.loads(path.read_text())
    response["choices"][0]["message"]["content"] = "cash decreased"
    path.write_text(json.dumps(response))
    metrics.update_metrics(records, states, out=out, cache=cache)
    assert calls == [36, 1]
    assert "prediction" not in (out / "scores.csv").read_text().splitlines()[0]


def test_empty_and_partial_cohort_exports_coverage_without_scoring(metrics, tmp_path, monkeypatch):
    records, states = states_for(metrics, tmp_path)
    calls = fake_scoring(metrics, monkeypatch)
    states["gemma26"].accepted = lambda: set()
    result = metrics.update_metrics(
        records, states, out=tmp_path / "out", cache=tmp_path / "cache.json"
    )
    assert result["documents_compared"] == 0 and calls == []
    assert result["progress"]["qwen37"]["accepted"] == 12
    assert "<svg " in (tmp_path / "out/dashboard.svg").read_text()


def test_cli_metrics_only_does_not_plan_or_generate(metrics, monkeypatch, tmp_path):
    rounds = importlib.import_module("run_full_round")
    monkeypatch.setattr(rounds, "ROOT", tmp_path)
    monkeypatch.setattr(rounds, "verify_full_lock", lambda _: {"cohort": []})
    monkeypatch.setattr(rounds, "inspect_models", lambda _: ({}, {}))
    calls = []
    monkeypatch.setattr(rounds, "update_metrics", lambda *args: calls.append(args))

    def forbidden(*args, **kwargs):
        raise AssertionError("generation or planning during metrics-only")

    monkeypatch.setattr(rounds, "execute_round", forbidden)
    monkeypatch.setattr(rounds, "round_plan", forbidden)
    monkeypatch.setattr("sys.argv", ["round", "--metrics-only"])
    rounds.main()
    assert len(calls) == 1


def test_cli_failed_round_updates_metrics_and_keeps_failure(metrics, monkeypatch, tmp_path):
    rounds = importlib.import_module("run_full_round")
    monkeypatch.setattr(rounds, "ROOT", tmp_path)
    monkeypatch.setattr(rounds, "verify_full_lock", lambda _: {"cohort": []})
    monkeypatch.setattr(rounds, "inspect_models", lambda _: ({"ling-free": None}, {}))
    monkeypatch.setattr(rounds, "audit_batch", lambda *args: None)
    monkeypatch.setattr(rounds, "round_plan", lambda *args, **kwargs: {"documents": ["a"]})
    monkeypatch.setattr(rounds, "remaining", lambda *args: {})
    monkeypatch.setattr(
        rounds,
        "execute_round",
        lambda *args: {"complete": False, "pending_generations": {"gemma26": 1}},
    )
    calls = []
    monkeypatch.setattr(rounds, "update_metrics", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["round", "100"])
    with pytest.raises(SystemExit) as e:
        rounds.main()
    assert e.value.code == 1 and len(calls) == 1
