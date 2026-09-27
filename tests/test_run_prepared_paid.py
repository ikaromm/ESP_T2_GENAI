"""Orcamento, contrato e idempotencia sem enviar chamadas pagas."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_prepared_paid")


def row(arm="C1"):
    return {"doc_id": "test", "arm": arm, "prompt_tokens": 100, "messages": []}


def response(target, **usage_changes):
    return {
        "model": target[0],
        "provider": target[2],
        "choices": [{"message": {"content": "cash was 100"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "cost": 0.00001} | usage_changes,
    }


def test_stop_before_budget_overdraft_and_resume_without_repetition(runner, tmp_path):
    target = runner.TARGETS["gemma26"]
    calls = []

    def request(*args):
        calls.append(args)
        return response(target)

    client = SimpleNamespace(_request=request)
    budget = runner.reservation(target)
    for _ in range(2):
        with pytest.raises(ValueError, match="teto de gasto"):
            runner.run([row(), row("C2")], target, tmp_path, budget, {}, client)
    assert len(calls) == 1
    assert calls[0][1]["provider"]["only"] == ["darkbloom"]
    assert calls[0][1]["provider"]["allow_fallbacks"] is False


def test_ambiguous_network_failure_is_reserved_and_not_retried(runner, tmp_path):
    target = runner.TARGETS["gemma26"]
    calls = []

    def request(*args):
        calls.append(args)
        raise TimeoutError()

    client = SimpleNamespace(_request=request)
    with pytest.raises(TimeoutError):
        runner.run([row()], target, tmp_path, runner.money(1), {}, client)
    with pytest.raises(ValueError, match="auditar"):
        runner.run([row()], target, tmp_path, runner.money(1), {}, client)
    assert len(calls) == 1


def test_validated_cost_releases_unused_reserve_without_overspending(runner, tmp_path):
    target = runner.TARGETS["gemma26"]
    calls = []

    def request(*args):
        calls.append(args)
        return response(target)

    budget = runner.reservation(target) + runner.money("0.00001")
    client = SimpleNamespace(_request=request)
    rows = [row(), row("C2"), row("C3")]
    for _ in range(2):
        with pytest.raises(ValueError, match="teto de gasto"):
            runner.run(rows, target, tmp_path, budget, {}, client)
    # Two cheap successes fit after releasing their unused reservations;
    # a third cannot reserve its maximum cost, including on resume.
    assert len(calls) == 2
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert [a["status"] for a in ledger["attempts"]] == ["accepted", "accepted"]


def test_six_429_attempts_do_not_silently_complete_on_resume(runner, tmp_path):
    calls, waits = [], []

    def request(*args):
        calls.append(args)
        raise runner.OpenRouterHTTPError({"http_status": 429, "headers": {}})

    client = SimpleNamespace(_request=request)
    target = runner.TARGETS["gemma26"]
    with pytest.raises(runner.OpenRouterHTTPError):
        runner.run([row()], target, tmp_path, runner.money(1), {}, client, sleep=waits.append)
    with pytest.raises(ValueError, match="seis tentativas"):
        runner.run([row()], target, tmp_path, runner.money(1), {}, client)
    assert len(calls) == 6
    assert waits == [1, 2, 4, 8, 16]


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 101},
        {"cost": "NaN"},
        {"cost": -1},
        {"cost": 1},
        {"completion_tokens_details": {"reasoning_tokens": 1}},
        {"completion_tokens": 0},
    ],
)
def test_bad_contract_rejected(runner, usage):
    target = runner.TARGETS["gemma26"]
    with pytest.raises(ValueError):
        runner.validate_response(response(target, **usage), target, row())


def test_changed_run_identity_rejected(runner, tmp_path):
    target = runner.TARGETS["gemma26"]
    client = SimpleNamespace(_request=lambda *args: response(target))
    runner.run([row()], target, tmp_path, runner.money(1), {"version": 1}, client)
    with pytest.raises(ValueError, match="identidade"):
        runner.run([row()], target, tmp_path, runner.money(1), {"version": 2}, client)


def test_same_output_cannot_spend_concurrently(runner, tmp_path):
    target = runner.TARGETS["gemma26"]

    def request(*args):
        with pytest.raises(ValueError, match="outra execucao"):
            runner.run([row()], target, tmp_path, runner.money(1), {}, client)
        return response(target)

    client = SimpleNamespace(_request=request)
    result = runner.run([row()], target, tmp_path, runner.money(1), {}, client)
    assert len(result["attempts"]) == 1


def test_contingency_keeps_entire_partial_document_block(runner, tmp_path):
    rows = [
        row(arm) | {"doc_id": doc}
        for doc in ("done", "partial", "pending")
        for arm in ("C1", "C1t", "C2", "C3", "C4", "C5")
    ]
    (tmp_path / "ling/C1").mkdir(parents=True)
    (tmp_path / "report.json").write_text(json.dumps({"models": {"ling": {"status": "blocked"}}}))
    (tmp_path / "ling/preflight.json").write_text(json.dumps(rows))
    accepted = [
        r
        | {
            "generation_metadata": {
                "model": "inclusionai/ling-3.0-flash-fin:free",
                "provider": "Novita",
                "finish_reason": "stop",
                "usage": {"cost": 0},
            }
        }
        for r in rows
        if r["doc_id"] == "done" or (r["doc_id"], r["arm"]) == ("partial", "C1")
    ]
    (tmp_path / "ling/C1/predictions.jsonl").write_text("\n".join(json.dumps(r) for r in accepted))
    result = runner.pending_document_blocks(rows, tmp_path)
    assert len(result) == 12
    assert {r["doc_id"] for r in result} == {"partial", "pending"}
    (tmp_path / "report.json").write_text(
        json.dumps({"models": {"ling": {"status": "generating"}}})
    )
    with pytest.raises(ValueError, match="interrompida"):
        runner.pending_document_blocks(rows, tmp_path)


def test_new_output_budget_and_length_are_explicit(runner):
    target = runner.TARGETS["gemma26"]
    item = row() | {"max_output_tokens": 8192, "allow_length": True}
    result = response(target, completion_tokens=8192)
    result["choices"][0]["finish_reason"] = "length"
    runner.validate_response(result, target, item)
    assert runner.request_payload(target, item)["max_tokens"] == 8192
    with pytest.raises(ValueError, match="incompleta"):
        runner.validate_response(result, target, item | {"allow_length": False})
    with pytest.raises(ValueError, match="saida invalido"):
        runner.request_payload(target, item | {"max_output_tokens": 1000000})


def test_retry_503_preserves_successful_case_on_resume(runner, tmp_path):
    from findsum_rag.openrouter import OpenRouterHTTPError

    target = runner.TARGETS["gemma26"]
    calls = []

    def request(*args):
        calls.append(args)
        if len(calls) == 1:
            raise OpenRouterHTTPError({"http_status": 503, "headers": {}, "message": "temporary"})
        return response(target)

    client = SimpleNamespace(_request=request)
    runner.run(
        [row()], target, tmp_path, runner.money("1"), {"case": "test"}, client, sleep=lambda _: None
    )
    runner.run(
        [row()], target, tmp_path, runner.money("1"), {"case": "test"}, client, sleep=lambda _: None
    )
    assert len(calls) == 2


def test_free_quota_stops_before_starting_incomplete_document(runner, tmp_path):
    from types import SimpleNamespace

    def forbidden(*args, **kwargs):
        raise AssertionError("nao iniciar bloco sem cota")

    rows = [
        {"doc_id": "d", "arm": a, "max_output_tokens": 8192}
        for a in ["C1", "C1t", "C2", "C3", "C4", "C5"]
    ]
    with pytest.raises(ValueError, match="cota diaria"):
        runner.run(
            rows,
            runner.TARGETS["ling-free"],
            tmp_path,
            runner.money(0),
            {},
            SimpleNamespace(_request=forbidden),
            max_new_calls=5,
        )
