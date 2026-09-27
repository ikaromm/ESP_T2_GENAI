"""Contagem remota auditavel: dublê sem rede e sem alegar tokenizer local."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def remote(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    return importlib.import_module("scripts.preparation.prepare_qwen_remote")


def test_remote_count_is_cached_and_budget_is_checked_first(remote, tmp_path):
    calls = []

    def request(route, payload):
        calls.append(payload)
        return {
            "model": remote.MODEL,
            "provider": "Alibaba",
            "usage": {"cost": 0.00001, "prompt_tokens": 10, "completion_tokens": 1},
        }

    transport = SimpleNamespace(_request=request)
    c = remote.RemoteCounter(tmp_path, remote.money("0.00004"), transport)
    messages = [{"role": "user", "content": "cash"}]
    assert c.count(messages) == c.count(messages) == 10
    assert len(calls) == 1 and calls[0]["max_tokens"] == 1
    assert calls[0]["provider"]["allow_fallbacks"] is False
    assert remote.RemoteCounter(tmp_path, remote.money("0.00004"), transport).count(messages) == 10
    with pytest.raises(ValueError, match="teto"):
        c.count([{"role": "user", "content": "another"}])
    assert len(calls) == 1


def test_remote_network_failure_cannot_be_repeated_silently(remote, tmp_path):
    def fail(*args):
        raise TimeoutError()

    c = remote.RemoteCounter(tmp_path, remote.money(1), SimpleNamespace(_request=fail))
    with pytest.raises(TimeoutError):
        c.count([{"role": "user", "content": "cash"}])
    with pytest.raises(ValueError, match="auditoria"):
        c.count([{"role": "user", "content": "cash"}])


def test_match_requires_both_remote_counts_and_preserves_prefixes(remote):
    source, retrieved = "cash flow " * 20, "liquidity " * 20
    a, b, n = remote.match_contexts(source, retrieved, 30, len, lambda t: len(t) + 10)
    assert source.startswith(a) and retrieved.startswith(b)
    assert len(a) == len(b) == n == 30
    with pytest.raises(ValueError, match="igualar"):
        remote.match_contexts(
            source, retrieved, 30, len, lambda t: len(t) + (1 if t.startswith("c") else 2)
        )


def test_remote_prefix_does_not_cut_table_cell(remote):
    text = "prose\n[tabela x]\nlong row label | usd | 25 (2020)"
    prefix = remote.safe_character_prefix(text, text.index("label"))
    assert prefix == "prose"


def test_unused_success_reservation_is_released_but_actual_cost_counts(remote, tmp_path):
    calls = []

    def request(route, payload):
        calls.append(payload)
        return {
            "model": remote.MODEL,
            "provider": "Alibaba",
            "usage": {"cost": 0.00001, "prompt_tokens": 10, "completion_tokens": 1},
        }

    counter = remote.RemoteCounter(
        tmp_path, remote.money("0.00005"), SimpleNamespace(_request=request)
    )
    counter.count([{"role": "user", "content": "one"}])
    counter.count([{"role": "user", "content": "two"}])
    with pytest.raises(ValueError, match="teto"):
        counter.count([{"role": "user", "content": "three"}])
    assert len(calls) == 2


def test_calibration_dry_run_never_instantiates_api_counter(remote, tmp_path, monkeypatch):
    import json

    from scripts.common import full_common

    from findsum_rag.config import ExperimentConfig

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    cfg = ExperimentConfig()
    cfg.data.n_eval_docs = 10
    cfg.to_yaml(prepared / "config.yaml")
    (prepared / "sha256.json").write_text(json.dumps({}))
    monkeypatch.setattr(full_common, "selected_records", lambda _: ("dev", [{}] * 10))

    def forbidden(*args, **kwargs):
        raise AssertionError("API acessada sem --execute")

    monkeypatch.setattr(remote, "RemoteCounter", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            "--prepared",
            str(prepared),
            "--output",
            str(tmp_path / "out"),
            "--budget-usd",
            "18",
        ],
    )
    remote.main()
    assert not (tmp_path / "out").exists()


def test_remote_prefix_keeps_prose_with_pipes(remote):
    prose = "Financial report | page 37. Operating cash flow was positive."
    assert remote.safe_character_prefix(prose, 40) == prose[:40]
    text = "[1] [tabela cash]\ncomplete | 10\npartial label | 20"
    assert remote.safe_character_prefix(text, text.index("label")) == (
        "[1] [tabela cash]\ncomplete | 10"
    )
