"""Rodadas comuns e retomadas, sem geracoes reais."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def rounds(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_full_round")


def cohort():
    return [{"doc_id": str(i)} for i in range(1000)]


def test_paid_models_catch_up_with_ling_instead_of_advancing_it(rounds, tmp_path):
    accepted = {"ling-free": set(range(600)), "gemma26": set(), "qwen37": set()}
    plan = rounds.round_plan(tmp_path, cohort(), accepted, 100, persist=False)
    assert plan["documents"] == [str(i) for i in range(100)]
    assert rounds.remaining(cohort(), plan["documents"], accepted) == {
        "ling-free": 0,
        "gemma26": 600,
        "qwen37": 600,
    }
    assert not (tmp_path / "active-round.json").exists()


def test_resume_keeps_same_documents_and_rejects_changed_size_or_plan(rounds, tmp_path):
    accepted = {model: set() for model in rounds.MODELS}
    plan = rounds.round_plan(tmp_path, cohort(), accepted, 37, persist=True)
    accepted["ling-free"] = set(range(222))
    assert rounds.round_plan(tmp_path, cohort(), accepted, 37, persist=True) == plan
    with pytest.raises(ValueError, match="37 casos"):
        rounds.round_plan(tmp_path, cohort(), accepted, 100, persist=True)
    path = tmp_path / "active-round.json"
    plan["documents"][0] = "999"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="alterado"):
        rounds.round_plan(tmp_path, cohort(), accepted, 37, persist=False)


def test_arbitrary_round_size_and_partial_document_keep_all_arms(rounds):
    accepted = {model: set(range(30)) for model in rounds.MODELS}
    accepted["gemma26"].remove(29)
    docs = rounds.choose_documents(cohort(), accepted, 7)
    assert docs == [str(i) for i in range(4, 11)]
    indices = rounds.document_indices(cohort(), docs)
    assert indices == set(range(24, 66))
    with pytest.raises(ValueError, match="ordem"):
        rounds.document_indices(cohort(), docs[::-1])
    with pytest.raises(ValueError, match="duplicada"):
        rounds.document_indices(cohort(), ["1", "1"])


def test_failure_preserves_round_and_other_models_continue_without_duplicates(
    rounds, tmp_path, monkeypatch
):
    records = cohort()
    accepted = {"ling-free": set(range(600)), "gemma26": set(), "qwen37": set()}
    states = {m: SimpleNamespace(model=m, accepted=lambda m=m: accepted[m]) for m in rounds.MODELS}
    plan = rounds.round_plan(tmp_path, records, accepted, 100, persist=True)
    sent = {m: [] for m in rounds.MODELS}
    fail = [True]

    def execute(state, *, documents, execute, stop_event=None):
        assert execute
        indices = rounds.document_indices(records, documents)
        pending = sorted(indices - accepted[state.model])
        if state.model == "gemma26" and fail[0]:
            fail[0] = False
            sent[state.model].extend(pending[:-1])
            accepted[state.model].update(pending[:-1])
            raise ValueError("interrupcao transitoria")
        sent[state.model].extend(pending)
        accepted[state.model].update(pending)

    monkeypatch.setattr(rounds, "run_selection", execute)
    first = rounds.execute_round(plan, records, states, accepted, {}, tmp_path)
    assert not first["complete"] and first["pending_generations"]["gemma26"] == 1
    assert (tmp_path / "active-round.json").exists()
    assert len(sent["qwen37"]) == 600
    second = rounds.execute_round(plan, records, states, accepted, {}, tmp_path)
    assert second["complete"] and not (tmp_path / "active-round.json").exists()
    assert len(sent["ling-free"]) == 0
    assert len(sent["gemma26"]) == len(set(sent["gemma26"])) == 600
    assert len(sent["qwen37"]) == 600
    assert rounds.choose_documents(records, accepted, 100) == [str(i) for i in range(100, 200)]


def test_dry_run_never_prepares_or_executes_paid_work(rounds, tmp_path, monkeypatch):
    records = cohort()
    accepted = {"ling-free": set(range(600)), "gemma26": set(), "qwen37": set()}
    monkeypatch.setattr(rounds, "ROOT", tmp_path)
    monkeypatch.setattr(rounds, "verify_full_lock", lambda _: {"cohort": records})
    monkeypatch.setattr(rounds, "inspect_models", lambda _: ({"ling-free": None}, accepted))
    monkeypatch.setattr(rounds, "audit_batch", lambda *a: None)

    def forbidden(*a, **k):
        raise AssertionError("API ou calibracao em dry-run")

    monkeypatch.setattr(rounds, "execute_round", forbidden)
    monkeypatch.setattr(rounds, "prepare_missing", forbidden)
    monkeypatch.setattr("sys.argv", ["round", "100", "--dry-run"])
    rounds.main()
    assert not (tmp_path / "active-round.json").exists()
    report = json.loads((tmp_path / "preview.json").read_text())
    assert report["preparation_pending"] == ["gemma26", "qwen37"]
    assert report["pending_generations"]["ling-free"] == 0


@pytest.mark.parametrize("value", ["0", "-1", "1001", "no"])
def test_invalid_size_rejected_before_work(rounds, monkeypatch, value):
    monkeypatch.setattr("sys.argv", ["round", value])
    with pytest.raises(SystemExit):
        rounds.main()


def test_finished_ling_selection_never_opens_api(rounds, tmp_path, monkeypatch):
    batcher = importlib.import_module("run_ling_batches")
    records = cohort()
    rows = [{"doc_id": r["doc_id"], "arm": arm} for r in records for arm in batcher.ARMS]
    state = batcher.PreparedRun(
        "ling-free", tmp_path, tmp_path, records, rows, None, {}, batcher.money(0)
    )
    monkeypatch.setattr(batcher.PreparedRun, "accepted", lambda _: set(range(600)))

    def forbidden(*a, **k):
        raise AssertionError("API consultada para documentos ja concluidos")

    monkeypatch.setattr(batcher, "OpenRouterFreeClient", forbidden)
    report = batcher.run_selection(state, documents=[str(i) for i in range(100)], execute=True)
    assert report["accepted_generations"] == 600


def test_audit_checks_failed_requests_too(rounds, tmp_path):
    batcher = importlib.import_module("run_ling_batches")
    core = importlib.import_module("run_prepared_paid")
    records = cohort()
    rows = [
        {"doc_id": r["doc_id"], "arm": arm, "messages": [], "prompt_tokens": 100}
        for r in records
        for arm in batcher.ARMS
    ]
    state = batcher.PreparedRun(
        "ling-free",
        tmp_path,
        tmp_path,
        records,
        rows,
        None,
        {"full_lock_id": "test"},
        core.money(0),
    )
    state.folder.mkdir()
    entry = {
        "case": 0,
        "doc_id": "0",
        "arm": "C1",
        "status": "accepted",
        "response_file": "0000-0.response.json",
    }
    retry = {
        "case": 1,
        "doc_id": "0",
        "arm": "C1t",
        "status": "http429",
        "response_file": "0001-0.response.json",
    }
    (state.folder / "ledger.json").write_text(
        json.dumps(
            {
                "identity": state.identity,
                "attempts": [entry, retry],
            }
        )
    )
    for i, row in enumerate(rows[:2]):
        (state.folder / f"{i:04d}-0.request.json").write_text(
            json.dumps(core.request_payload(core.TARGETS[state.model], row))
        )
    (state.folder / entry["response_file"]).write_text(
        json.dumps(
            {
                "model": core.TARGETS[state.model][0],
                "provider": "Novita",
                "usage": {"cost": 0, "prompt_tokens": 100, "completion_tokens": 5},
                "choices": [{"message": {"content": "summary"}, "finish_reason": "stop"}],
            }
        )
    )
    result = rounds.audit_batch(state)
    assert result["attempts"] == 2 and result["accepted_generations"] == 1
    assert not result["complete"] and result["reported_cost_usd"] == "0"
    (state.folder / "0001-0.request.json").write_text("{}")
    with pytest.raises(ValueError, match="historico"):
        rounds.audit_batch(state)


def test_qwen_missing_prepares_locally_without_paid_probes(rounds, tmp_path, monkeypatch):
    prepared = tmp_path / "qwen"
    monkeypatch.setattr(rounds, "paths", lambda _: (prepared, tmp_path / "run"))
    calls = []
    monkeypatch.setattr(rounds, "child", lambda script, args: calls.append((script, args)))
    rounds.prepare_missing("qwen37", {})
    assert calls == [
        (
            "prepare_gemma_from_common.py",
            [
                "--model",
                "qwen37",
                "--source",
                "outputs/full-ling-prepared",
                "--output",
                str(prepared),
            ],
        )
    ]
    prepared.mkdir()
    with pytest.raises(ValueError, match="incompleta"):
        rounds.prepare_missing("qwen37", {})


def test_models_execute_simultaneously_with_same_documents(rounds, tmp_path, monkeypatch):
    from threading import Barrier

    records = cohort()
    accepted = {model: set() for model in rounds.MODELS}
    states = {m: SimpleNamespace(model=m, accepted=lambda m=m: accepted[m]) for m in rounds.MODELS}
    plan = rounds.round_plan(tmp_path, records, accepted, 100, persist=True)
    barrier = Barrier(3)

    def execute(state, *, documents, execute, stop_event=None):
        assert execute and not stop_event.is_set()
        barrier.wait(timeout=3)
        accepted[state.model].update(rounds.document_indices(records, documents))

    monkeypatch.setattr(rounds, "run_selection", execute)
    result = rounds.execute_round(plan, records, states, accepted, {}, tmp_path)
    assert result["complete"] and not result["errors"]


@pytest.mark.parametrize("model", ["qwen37", "gemma26"])
def test_paid_batch_uses_adaptive_executor(rounds, tmp_path, monkeypatch, model):
    batcher = importlib.import_module("run_ling_batches")
    adaptive = importlib.import_module("run_paid_concurrent")
    records = cohort()
    rows = [{"doc_id": r["doc_id"], "arm": arm} for r in records for arm in batcher.ARMS]
    state = batcher.PreparedRun(
        model, tmp_path, tmp_path, records, rows, None, {}, batcher.money(9)
    )
    accepted = set()
    monkeypatch.setattr(batcher.PreparedRun, "accepted", lambda _: accepted.copy())
    monkeypatch.setattr(batcher, "ensure_meteor_resources", lambda: None)
    monkeypatch.setattr(batcher, "load_key", lambda: "mock")
    monkeypatch.setattr(batcher, "OpenRouterFreeClient", lambda *a, **k: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("executor serial pago invocado")

    def concurrent(*args, case_indices, **kwargs):
        assert len(case_indices) == 600
        accepted.update(case_indices)

    monkeypatch.setattr(batcher, "run", forbidden)
    monkeypatch.setattr(adaptive, "run", concurrent)
    result = batcher.run_selection(state, documents=[str(i) for i in range(100)], execute=True)
    assert result["accepted_generations"] == 600
