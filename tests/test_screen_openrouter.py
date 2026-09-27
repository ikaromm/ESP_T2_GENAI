"""Retentativas limitadas e retomada sem chamadas repetidas."""
import importlib.util
from pathlib import Path

import pytest

from findsum_rag.openrouter import OpenRouterHTTPError

spec = importlib.util.spec_from_file_location(
    'screen', Path(__file__).parents[1] / 'scripts/screen_openrouter.py'
)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def error(status=429, retry_after=None):
    return OpenRouterHTTPError({
        'http_status': status,
        'headers': {} if retry_after is None else {'Retry-After': retry_after},
    })


def test_backoff_recovers_without_extra_calls():
    calls, sleeps = [], []
    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise error()
        return 'ok'
    case = {}
    assert screen.with_retries(operation, case, lambda: None, sleep=sleeps.append) == 'ok'
    assert len(calls) == 3
    assert sleeps == [1, 2]
    assert len(case['attempt_errors']) == 2


def test_repeated_429_opens_circuit():
    calls, sleeps = [], []
    def operation():
        calls.append(1)
        raise error()
    with pytest.raises(OpenRouterHTTPError):
        screen.with_retries(operation, {}, lambda: None, sleep=sleeps.append)
    assert len(calls) == 6
    assert sleeps == [1, 2, 4, 8, 16]


@pytest.mark.parametrize('exc', [error(401), error(429, '120'), ValueError('bad contract')])
def test_no_retry_for_auth_contract_or_long_retry_after(exc):
    calls, sleeps = [], []
    def operation():
        calls.append(1)
        raise exc
    with pytest.raises(type(exc)):
        screen.with_retries(operation, {}, lambda: None, sleep=sleeps.append)
    assert len(calls) == 1
    assert not sleeps


def test_retry_after_is_respected():
    assert error(429, '45').retry_delay(10) == 45
    assert error(429, '2').retry_delay(10) == 10
    assert error(429, 'invalid').retry_delay(10) == 10


def test_resume_restores_results_and_rejects_changed_evidence(tmp_path):
    import json

    from findsum_rag.config import ExperimentConfig, GenerationConfig
    from findsum_rag.metrics import RougeScorer
    from findsum_rag.openrouter import MODEL

    cfg = ExperimentConfig(generation=GenerationConfig(backend='openrouter', model_name=MODEL))
    row = {
        'doc_id': 'doc1', 'arm': 'C1', 'context': 'cash was 100',
        'source': 'cash was 100', 'prediction': 'cash was 100', 'reference': 'cash was 100',
        'example_ids': [], 'context_tokens': 4, 'prompt_tokens': 10, 'truncated_prompt': False,
        'generation_metadata': {'model': MODEL, 'provider': 'ModelRun',
                                'finish_reason': 'stop', 'usage': {'cost': 0}},
    }
    path = tmp_path / 'C1' / 'predictions.jsonl'
    path.parent.mkdir()
    path.write_text(json.dumps(row) + '\n')
    expected = [dict(row)]
    results, completed = screen.restore_results(tmp_path, cfg, expected, RougeScorer())
    assert completed == {('doc1', 'C1')}
    assert results['C1'].predictions == [row]
    assert results['C1'].summary['n_docs'] == 1
    expected[0]['context'] = 'changed evidence'
    with pytest.raises(ValueError, match='preflight'):
        screen.restore_results(tmp_path, cfg, expected, RougeScorer())
    expected[0]['context'] = row['context']
    path.write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError, match='duplicado'):
        screen.restore_results(tmp_path, cfg, expected, RougeScorer())


def test_window_counts_retries_across_models_and_survives_resume(tmp_path):
    now, sleeps = [100.0], []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    path = tmp_path / 'window.json'
    limiter = screen.RequestWindow(path, clock=lambda: now[0], sleep=sleep)
    for _ in range(20):
        limiter.acquire()
    assert not sleeps
    resumed = screen.RequestWindow(path, clock=lambda: now[0], sleep=sleep)
    resumed.acquire()
    assert sleeps == [60.05]
    assert len(resumed.sent) == 1


def test_retries_each_reserve_a_rate_slot(tmp_path):
    now = [100.0]
    limiter = screen.RequestWindow(tmp_path / 'window.json', clock=lambda: now[0])
    calls = []
    def operation():
        calls.append(1)
        if len(calls) == 1:
            raise error()
        return 'ok'
    result = screen.with_retries(
        operation, {}, lambda: None, limiter=limiter, sleep=lambda s: now.__setitem__(0, now[0]+s)
    )
    assert result == 'ok'
    assert limiter.sent == [100, 101]


def test_daily_quota_allows_available_calls_and_stops_before_overdraft(tmp_path):
    window = screen.RequestWindow(tmp_path / 'window.json')
    limiter = screen.FreeQuotaLimiter(window, 2)
    calls = []
    def operation():
        calls.append(1)
        raise error()
    with pytest.raises(ValueError, match='cota gratuita diaria'):
        screen.with_retries(operation, {}, lambda: None, limiter=limiter, sleep=lambda _: None)
    assert len(calls) == 2
    assert len(window.sent) == 2
    assert limiter.remaining == 0
