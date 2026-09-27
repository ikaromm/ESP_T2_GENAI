"""Validacao paralela conserva ordem, rejeita adulteracao e nao acessa a API."""

import importlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("model", ["ling-free", "qwen37"])
def test_parallel_preflight_matches_serial_and_rejects_drift(tmp_path, monkeypatch, model):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    core = importlib.import_module("scripts.execution.run_prepared_paid")
    common = importlib.import_module("scripts.common.full_common")
    from findsum_rag.full_lock import digest

    monkeypatch.chdir(tmp_path)
    manifest = Path("data/interim/splits-liquidity.json")
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    folder = tmp_path / "prepared"
    (folder / model / "tokenizer").mkdir(parents=True)
    (folder / "selection.json").write_text(
        json.dumps(
            {
                "manifest_sha256": digest(manifest),
                "documents": [{"doc_id": "a"}],
            }
        )
    )
    (folder / "sha256.json").write_text("{}")
    rows = [
        {
            "doc_id": "a",
            "arm": arm,
            "context": "same",
            "context_tokens": 4,
            "messages": [{"role": "user", "content": "same"}],
            "prompt_tokens": 4,
        }
        for arm in common.ARMS
    ]
    preflight = folder / model / "preflight.json"
    preflight.write_text(json.dumps(rows))
    monkeypatch.setattr(common, "selected_records", lambda _: ("eval", [{"doc_id": "a"}]))
    threads = set()

    def encode(text, **kwargs):
        threads.add(threading.get_ident())
        time.sleep(0.01)
        return list(text)

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: messages[0]["content"],
        encode=encode,
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tokenizer)
    local = importlib.import_module("scripts.preparation.qwen_local_tokenizer")
    monkeypatch.setattr(local, "load_qwen_tokenizer", lambda _: tokenizer)
    assert core.load_prepared(folder, model, 1, validation_workers=1) == rows
    threads.clear()
    assert core.load_prepared(folder, model, 1, validation_workers=4) == rows
    assert len(threads) > 1
    rows[-1]["prompt_tokens"] = 5
    preflight.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="contagem local alterada"):
        core.load_prepared(folder, model, 1, validation_workers=4)


def test_heartbeat_exits_on_error(capsys):
    from findsum_rag.progress import activity

    with pytest.raises(ValueError), activity("Validacao", interval=0.005):
        time.sleep(0.02)
        raise ValueError("private payload")
    output = capsys.readouterr().out
    assert "em andamento" in output and "interrompido" in output
    assert "private payload" not in output
