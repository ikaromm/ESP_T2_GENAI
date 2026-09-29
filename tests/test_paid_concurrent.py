"""Retries adaptativos e orcamento concorrente, sem API real."""

import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def core(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    m = importlib.import_module("scripts.execution.run_paid_concurrent")
    monkeypatch.setattr(m, "dispatch_delay", lambda *args: 0)
    return m


class Clock:
    def __init__(self):
        self.now = 10000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def setup(core, tmp_path):
    clock = Clock()
    target = ("qwen/qwen3.7-flash", "alibaba", "Alibaba", ".1", ".4")
    rows = [
        dict(
            doc_id="d",
            arm="C1",
            prompt_tokens=10,
            max_output_tokens=8192,
            messages=[{"role": "user", "content": "a"}],
        )
    ]
    good = dict(
        model=target[0],
        provider=target[2],
        usage=dict(cost=0.0001, prompt_tokens=10, completion_tokens=1),
        choices=[dict(finish_reason="stop", message=dict(content="summary"))],
    )
    args = (rows, target, tmp_path, core.money(1), {"same": True})
    kwargs = dict(clock=clock, sleep=clock.sleep, jitter=lambda: 0)
    return clock, good, args, kwargs


def test_429_retry_after_requeues_same_case_and_persists_lower_rate(core, tmp_path):
    clock, good, args, kwargs = setup(core, tmp_path)
    times = []

    def request(*unused):
        times.append(clock())
        if len(times) == 1:
            raise core.OpenRouterHTTPError({"http_status": 429, "headers": {"Retry-After": "3"}})
        return good

    client = SimpleNamespace(_request=request)
    core.run(*args, client, **kwargs)
    assert len(times) == 2 and times[1] - times[0] >= 3
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert [a["status"] for a in ledger["attempts"]] == ["http429", "accepted"]
    assert json.loads((tmp_path / "adaptive-rate.json").read_text())["rpm"] == 250
    core.run(*args, client, **kwargs)
    assert len(times) == 2


def test_repeated_429_is_bounded_and_does_not_restart_cycle_immediately(core, tmp_path):
    clock, good, args, kwargs = setup(core, tmp_path)
    calls = []

    def request(*unused):
        calls.append(clock())
        raise core.OpenRouterHTTPError({"http_status": 429, "headers": {}})

    client = SimpleNamespace(_request=request)
    with pytest.raises(ValueError, match="seis tentativas"):
        core.run(*args, client, **kwargs)
    assert len(calls) == 6
    with pytest.raises(ValueError, match="circuito"):
        core.run(*args, client, **kwargs)
    assert len(calls) == 6
    clock.now += 3601
    core.run(*args, SimpleNamespace(_request=lambda *a: good), **kwargs)


def test_timeout_is_not_repeated_and_budget_remains_reserved(core, tmp_path):
    _clock, _good, args, kwargs = setup(core, tmp_path)
    calls = []

    def request(*unused):
        calls.append(1)
        raise TimeoutError()

    for _ in range(2):
        with pytest.raises(ValueError, match="auditoria"):
            core.run(*args, SimpleNamespace(_request=request), **kwargs)
    assert calls == [1]
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert core.held_cost(ledger) == core.reservation(args[1], args[0][0])


def test_budget_inflight_drain_then_stop_without_overcommit(core, tmp_path):
    _clock, good, args, kwargs = setup(core, tmp_path)
    rows, target, folder, _, identity = args
    budget = core.reservation(target, rows[0])
    calls = []

    def request(*unused):
        calls.append(1)
        return good

    with pytest.raises(ValueError, match="teto de gasto"):
        core.run(
            rows * 2, target, folder, budget, identity, SimpleNamespace(_request=request), **kwargs
        )
    assert calls == [1]


def test_cancellation_records_inflight_and_stops_new_dispatch(core, tmp_path):
    _clock, good, args, kwargs = setup(core, tmp_path)
    stop = threading.Event()
    calls = []

    def request(*unused):
        calls.append(1)
        stop.set()
        return good

    with pytest.raises(ValueError, match="interrompido"):
        core.run(
            args[0] * 3, *args[1:], SimpleNamespace(_request=request), stop_event=stop, **kwargs
        )
    assert calls == [1]
    assert json.loads((tmp_path / "ledger.json").read_text())["attempts"][0]["status"] == "accepted"


def test_only_inflight_402_is_retryable_and_rate_recovers_gradually(core):
    def error(source, reason):
        return core.OpenRouterHTTPError(
            {
                "http_status": 402,
                "metadata": {"limit_source": source, "reason": reason},
                "headers": {},
            }
        )

    assert core.transient(error("openrouter_in_flight_budget", "in_flight_budget_exhausted"))
    assert not core.transient(error("openrouter_credits", "weight_exceeds_budget"))
    rate = core.AdaptiveRate()
    rate.failure(core.OpenRouterHTTPError({"http_status": 429, "headers": {}}), 1, 100, 0)
    for _ in range(30):
        rate.success(110)
    assert rate.data["rpm"] == 250
    rate.success(131)
    assert rate.data["rpm"] == 300
    for _ in range(100):
        rate.success(132)
    assert rate.data["rpm"] == 300


def test_paid_ling_resumes_audited_free_404_as_new_attempt(core, tmp_path):
    clock, _good, _args, kwargs = setup(core, tmp_path)
    target = core.TARGETS["ling-paid-novita"]
    row = dict(
        doc_id="d", arm="C1", prompt_tokens=10, max_output_tokens=8192,
        messages=[{"role": "user", "content": "a"}],
    )
    rows = [row] * 961
    old = dict(
        case=960, doc_id="d", arm="C1", reserved_usd="0",
        status="http_retryable", at=clock() - 100,
        error={"http_status": 404, "message": "This model is unavailable for free."},
    )
    (tmp_path / "ledger.json").write_text(json.dumps(dict(identity={"same": True}, attempts=[old])))
    free_request = core.request_payload(core.TARGETS["ling-free"], row)
    (tmp_path / "0960-0.request.json").write_text(json.dumps(free_request))
    response = dict(
        model=target[0], provider=target[2],
        usage=dict(cost=0.0001, prompt_tokens=10, completion_tokens=1),
        choices=[dict(finish_reason="stop", message=dict(content="summary"))],
    )
    core.run(
        rows, target, tmp_path, core.money(1), {"same": True},
        SimpleNamespace(_request=lambda *unused: response),
        case_indices={960}, amendment_sha256="a" * 64, **kwargs,
    )
    attempts = json.loads((tmp_path / "ledger.json").read_text())["attempts"]
    assert [a["status"] for a in attempts] == ["http_retryable", "accepted"]
    assert attempts[-1]["retry_cycle_end"] == 7
    assert attempts[-1]["amendment_sha256"] == "a" * 64
    assert json.loads((tmp_path / "0960-1.request.json").read_text())["model"] == target[0]
