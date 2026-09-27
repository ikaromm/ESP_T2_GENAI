"""Realocacoes revisadas nao liberam mudancas cientificas ou comandos quebrados."""

import json
from copy import deepcopy
from pathlib import Path

from findsum_rag.full_lock import compatible_preparation, lock_digest


def example_locks():
    layout = json.loads(Path("configs/repository-layout.json").read_text())
    move = layout["moves"]["scripts/full_common.py"]
    previous = {"files": {"scripts/full_common.py": move["before_sha256"]}, "cohort": ["same"]}
    previous["lock_id"] = lock_digest(previous)
    current = {
        "files": {move["path"]: move["after_sha256"], **layout["package_initializers"]},
        "cohort": ["same"],
    }
    return previous, current, move


def test_exact_layout_migration_preserves_preparation():
    previous, current, _ = example_locks()
    assert compatible_preparation(previous, current)


def test_layout_rejects_unknown_source_hash_and_modified_scientific_script():
    previous, current, move = example_locks()
    changed = deepcopy(previous)
    changed["files"]["scripts/full_common.py"] = "unknown"
    changed["lock_id"] = lock_digest(changed)
    assert not compatible_preparation(changed, current)
    current["files"][move["path"]] = "different"
    assert not compatible_preparation(previous, current)


def test_new_initializers_are_not_blanket_exemptions():
    previous, current, _ = example_locks()
    current["files"]["scripts/__init__.py"] = "unexpected-code"
    assert not compatible_preparation(previous, current)


def test_relocated_cli_executes_module_from_checkout(monkeypatch):
    from types import SimpleNamespace

    from findsum_rag.cli import _script

    calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **kw: calls.append((command, kw)) or SimpleNamespace(returncode=0),
    )
    _script("scripts.execution.run_full_round", ["--metrics-only"])
    command, options = calls[0]
    assert command[1:] == ["-u", "-m", "scripts.execution.run_full_round", "--metrics-only"]
    assert options["cwd"] == Path(__file__).resolve().parents[1]


def test_current_documentation_has_no_broken_relative_links():
    import re

    paths = [Path("README.md"), Path("scripts/README.md"), *Path("docs").glob("*.md")]
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path, target)
