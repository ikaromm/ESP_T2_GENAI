"""Limites operacionais pagos/gratuitos sem API nem esperas reais."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def core(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    return importlib.import_module("scripts.execution.run_prepared_paid")


@pytest.mark.parametrize(
    "model,limit",
    [
        ("ling-free", 20),
        ("ling-paid", 100),
        ("qwen37", 100),
        ("gemma26", 100),
    ],
)
def test_executor_selects_limit_by_paid_or_free(core, tmp_path, monkeypatch, model, limit):
    seen = []
    real = core.RequestWindow

    def window(path, **kwargs):
        seen.append(kwargs["limit"])
        return real(path, **kwargs)

    monkeypatch.setattr(core, "RequestWindow", window)
    core.run([], core.TARGETS[model], tmp_path, core.money(0), {}, None)
    assert seen == [limit]


def test_paid_window_allows_100_and_blocks_101_after_resume(core, tmp_path, capsys):
    now = [100.0]
    delays = []

    def sleep(delay):
        delays.append(delay)
        now[0] += delay

    path = tmp_path / "request-window.json"
    limiter = core.RequestWindow(path, limit=100, clock=lambda: now[0], sleep=sleep)
    for _ in range(100):
        limiter.acquire()
    assert not delays
    resumed = core.RequestWindow(path, limit=100, clock=lambda: now[0], sleep=sleep)
    resumed.acquire()
    assert len(delays) == 1 and delays[0] >= 60
    assert len(resumed.sent) == 1
    assert "Limite de 100" in capsys.readouterr().out


def test_qwen_calibration_uses_paid_limit(core, tmp_path):
    remote = importlib.import_module("scripts.preparation.prepare_qwen_remote")
    counter = remote.RemoteCounter(tmp_path, core.money(18), None)
    assert counter.limiter.limit == 100
