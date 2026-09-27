"""Contrato congelado do full: verificacao offline antes de qualquer chamada."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_LOCK = Path("configs/full_openrouter.lock.json")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lock_digest(data):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in data.items() if k != "lock_id"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


# Somente mudancas operacionais desta revisao. Prompts, contexto, configuracao,
# dados, metricas e seletores continuam exigindo hashes identicos.
OPERATIONAL_COMPATIBILITY = {
    "src/findsum_rag/cli.py",
    "src/findsum_rag/full_lock.py",
    "src/findsum_rag/progress.py",
    "scripts/run_ling_batches.py",
    "scripts/run_prepared_paid.py",
    "scripts/freeze_full_openrouter.py",
    "scripts/prepare_gemma_from_common.py",
    "scripts/prepare_qwen_remote.py",
}


def compatible_preparation(previous, current):
    if previous.get("lock_id") != lock_digest(previous):
        return False
    scientific = (
        "dataset_files",
        "versions",
        "hf_assets",
        "wordnet_sha256",
        "cohort",
        "analysis_plan",
        "supplementary_metrics",
        "budgets_usd",
    )
    if any(previous.get(k) != current.get(k) for k in scientific):
        return False
    changed = {
        name
        for name in previous["files"].keys() | current["files"].keys()
        if previous["files"].get(name) != current["files"].get(name)
    }
    return changed <= OPERATIONAL_COMPATIBILITY


def verify_hashes_parallel(files, *, workers=4, label="arquivos"):
    """Hashes em paralelo, leitura em blocos; nunca carrega pesos na memoria."""
    from .progress import log

    def check(item):
        name, expected = item
        if digest(name) != expected:
            raise ValueError(f"arquivo difere do congelado: {name}")

    log(f"Conferindo {len(files)} {label} com {workers} trabalhadores")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(check, files.items()), 1):
            if i % 100 == 0 or i == len(files):
                log(f"Integridade {label}: {i}/{len(files)}")


def verify_full_lock(path=DEFAULT_LOCK, *, prepared=None, check_assets=True):
    from huggingface_hub.constants import HF_HUB_CACHE

    path = Path(path)
    if not path.is_file():
        raise ValueError("full exige configuracao congelada antes de qualquer chamada")
    data = json.loads(path.read_text())
    if data.get("lock_id") != lock_digest(data):
        raise ValueError("identificador do lock invalido")
    for section in ["files", "dataset_files"]:
        verify_hashes_parallel(data[section], label=section)
    for package, version in data["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"versao diferente da congelada: {package}")
    if check_assets:
        for model, spec in data["hf_assets"].items():
            base = Path(HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))
            if (base / "refs/main").read_text().strip() != spec["revision"]:
                raise ValueError(f"revisao local mudou: {model}")
            verify_hashes_parallel(
                {
                    base / "snapshots" / spec["revision"] / name: expected
                    for name, expected in spec["files"].items()
                },
                label=model,
            )
    if data.get("wordnet_sha256"):
        import nltk

        resource = nltk.data.find("corpora/wordnet.zip")
        if digest(resource.zipfile.filename) != data["wordnet_sha256"]:
            raise ValueError("recurso WordNet difere do congelado")
    if prepared is not None:
        prepared = Path(prepared)
        selection = json.loads((prepared / "selection.json").read_text())
        if selection["split"] != "eval" or selection["documents"] != data["cohort"]:
            raise ValueError("coorte difere dos 1000 casos congelados")
        from .config import ExperimentConfig

        expected = ExperimentConfig.from_yaml("configs/full_openrouter.yaml")
        actual = ExperimentConfig.from_yaml(prepared / "config.yaml")
        if actual != expected:
            raise ValueError("configuracao preparada difere do full congelado")
        prepared_lock = json.loads((prepared / "full-lock.json").read_text())
        if prepared_lock["lock_id"] != data["lock_id"]:
            report = json.loads((prepared / "report.json").read_text())
            if (
                prepared_lock["lock_id"] not in data.get("compatible_prepared_locks", [])
                or not compatible_preparation(prepared_lock, data)
                or set(report.get("models", {})) != {"ling-free"}
            ):
                raise ValueError("preparacao pertence a outro lock")
    # Evita atualizacao automatica de main nos modelos avaliadores/tokenizers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_OFFLINE = True
    return data
