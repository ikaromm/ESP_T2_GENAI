import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findsum_rag.cli import app
from findsum_rag.full_lock import digest, lock_digest, verify_full_lock


def test_full_lock_rejects_changed_source_before_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("prompt.txt")
    source.write_text("original")
    lock = {
        "files": {"prompt.txt": digest(source)},
        "dataset_files": {},
        "versions": {},
        "hf_assets": {},
    }
    lock["lock_id"] = lock_digest(lock)
    Path("lock.json").write_text(json.dumps(lock))
    assert verify_full_lock("lock.json")["lock_id"] == lock["lock_id"]
    source.write_text("changed")
    with pytest.raises(ValueError, match="congelado"):
        verify_full_lock("lock.json")


def test_full_lock_rejects_edited_manifest(tmp_path):
    data = {"files": {}, "dataset_files": {}, "versions": {}, "hf_assets": {}}
    data["lock_id"] = lock_digest(data)
    data["versions"]["torch"] = "not-the-frozen-version"
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identificador"):
        verify_full_lock(path)


def test_cli_run_is_non_executing_by_default(monkeypatch):
    seen = []
    monkeypatch.setattr("findsum_rag.cli._script", lambda name, args: seen.append((name, args)))
    result = CliRunner().invoke(
        app, ["run", "--prepared", "a", "--qwen-prepared", "b", "--output", "c", "--allow-eval"]
    )
    assert result.exit_code == 0
    assert seen[0][0] == "run_experiment_openrouter.py"
    assert "--execute" not in seen[0][1]
    assert "--config" not in seen[0][1]


def test_operational_upgrade_preserves_scientific_lock_only():
    from copy import deepcopy

    from findsum_rag.full_lock import compatible_preparation

    previous = {
        "files": {"scripts/run_ling_batches.py": "old", "src/findsum_rag/prompts.py": "fixed"},
        "cohort": ["same"],
        "budgets_usd": {"qwen_generation": 18},
    }
    previous["lock_id"] = lock_digest(previous)
    current = deepcopy(previous)
    current["files"]["scripts/run_ling_batches.py"] = "new"
    current["files"]["src/findsum_rag/progress.py"] = "new"
    assert compatible_preparation(previous, current)
    current["files"]["src/findsum_rag/prompts.py"] = "different"
    assert not compatible_preparation(previous, current)
    current = deepcopy(previous)
    current["cohort"] = ["other"]
    assert not compatible_preparation(previous, current)
    previous["files"]["src/findsum_rag/prompts.py"] = "tampered"
    assert not compatible_preparation(previous, previous)
