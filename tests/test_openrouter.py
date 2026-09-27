"""Contrato da PoC gratuita, sem chamadas externas nos testes."""

import pytest

from findsum_rag.openrouter import MODEL, PROVIDER, OpenRouterFreeClient


def response(**changes):
    result = {
        "model": MODEL,
        "provider": "ModelRun",
        "choices": [{"message": {"content": "cash was 100"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0},
    }
    result.update(changes)
    return result


def test_payload_free_only_no_fallback_no_unsupported_seed():
    payload = OpenRouterFreeClient.payload([{"role": "user", "content": "test"}])
    assert payload["model"] == MODEL
    assert payload["provider"]["only"] == [PROVIDER]
    assert payload["provider"]["allow_fallbacks"] is False
    assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0}
    assert payload["reasoning"]["enabled"] is False
    assert "seed" not in payload


def test_success_preserves_actual_usage_and_finish_reason(monkeypatch):
    client = OpenRouterFreeClient("test-key")
    monkeypatch.setattr(client, "_request", lambda *args: response())
    result = client.complete([])
    assert result["text"] == "cash was 100"
    assert result["usage"]["cost"] == 0
    assert result["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "body",
    [
        response(choices=[]),
        response(error={"code": 429}),
        response(usage={"cost": 0.01}),
        response(choices=[{"message": {"content": ""}}]),
    ],
)
def test_invalid_responses_fail_without_retry(monkeypatch, body):
    client = OpenRouterFreeClient("test-key")
    calls = []

    def request(*args):
        calls.append(args)
        return body

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RuntimeError):
        client.complete([])
    assert len(calls) == 1


def test_http_error_does_not_expose_credentials(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    key = "private-test-value"

    def reject(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429,
            "limited",
            {},
            io.BytesIO(key.encode()),
        )

    monkeypatch.setattr(urllib.request, "urlopen", reject)
    with pytest.raises(RuntimeError, match="HTTP 429") as error:
        OpenRouterFreeClient(key).complete([])
    assert key not in str(error.value)


def test_http_error_keeps_limit_diagnostics_but_redacts_key(monkeypatch):
    import io
    import json
    import urllib.error
    import urllib.request

    key = "private-test-key"
    body = {
        "error": {
            "message": "Provider returned error",
            "metadata": {
                "provider_name": "ModelRun",
                "limit_source": "upstream_provider_shared_pool",
                "raw": f"temporarily limited {key}",
                "unrelated": "do not log",
            },
        }
    }

    def reject(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429,
            "limited",
            {"Retry-After": "30", "Authorization": "never-log"},
            io.BytesIO(json.dumps(body).encode()),
        )

    monkeypatch.setattr(urllib.request, "urlopen", reject)
    with pytest.raises(RuntimeError) as error:
        OpenRouterFreeClient(key).complete([])
    details = json.loads(str(error.value).split(": ", 1)[1])
    assert details["metadata"]["limit_source"] == "upstream_provider_shared_pool"
    assert details["headers"] == {"Retry-After": "30"}
    assert key not in str(error.value)
    assert "never-log" not in str(error.value)
    assert "unrelated" not in details["metadata"]


@pytest.mark.parametrize('change', [
    {'model_name': 'paid/model'}, {'temperature': 0.5},
    {'max_input_tokens': 262144},
])
def test_remote_config_rejects_incompatible_settings(change):
    from findsum_rag.config import GenerationConfig

    with pytest.raises(ValueError):
        GenerationConfig(**({'backend': 'openrouter', 'model_name': MODEL} | change))


@pytest.mark.parametrize('bad_usage,finish', [(True, 'stop'), (False, 'length')])
def test_remote_rejects_count_mismatch_and_incomplete_output(
    tmp_path, monkeypatch, bad_usage, finish,
):
    import json

    from findsum_rag.config import GenerationConfig
    from findsum_rag.openrouter import OpenRouterSummarizer
    from findsum_rag.prompts import Prompt

    monkeypatch.setenv('OPEN_ROUTER_KEY', 'test-key')
    gen = OpenRouterSummarizer(
        GenerationConfig(backend='openrouter', model_name=MODEL), audit_dir=tmp_path,
    )
    monkeypatch.setattr(gen, 'load_tokenizer', lambda: None)
    monkeypatch.setattr(gen, 'count_tokens', lambda _: 10)
    result = {'text': 'cash was 100', 'finish_reason': finish,
              'model': MODEL, 'provider': 'ModelRun',
              'usage': {'prompt_tokens': 11 if bad_usage else 10, 'completion_tokens': 5}}
    monkeypatch.setattr(gen.client, 'complete', lambda *a, **k: result)
    with pytest.raises(ValueError):
        gen.generate(Prompt('system', 'user', 0, 1, 1))
    assert json.loads((tmp_path / '00001.response.json').read_text()) == result
    assert 'test-key' not in (tmp_path / '00001.request.json').read_text()


@pytest.mark.parametrize('model,provider', [
    ('inclusionai/ling-3.0-flash-fin:free', 'novita'),
    ('google/gemma-4-31b-it:free', 'google-ai-studio'),
])
def test_screening_models_keep_pinned_free_routing(monkeypatch, model, provider):
    from findsum_rag.config import GenerationConfig

    GenerationConfig(backend='openrouter', model_name=model)
    calls = []
    client = OpenRouterFreeClient('test-key', model=model)

    def request(route, payload):
        calls.append(payload)
        return response(model=model)

    monkeypatch.setattr(client, '_request', request)
    client.complete([{'role': 'user', 'content': 'cash was 10'}])
    assert calls[0]['model'] == model
    assert calls[0]['provider']['only'] == [provider]
    assert calls[0]['provider']['max_price'] == {'prompt': 0, 'completion': 0}
    assert calls[0]['provider']['allow_fallbacks'] is False
    assert calls[0]['reasoning'] == {'enabled': False}
    assert 'seed' not in calls[0]
