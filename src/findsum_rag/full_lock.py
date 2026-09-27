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
    "scripts/run_full_round.py",
    "rodar_rodada.sh",
    "scripts/run_prepared_paid.py",
    "scripts/screen_openrouter.py",
    "scripts/freeze_full_openrouter.py",
    "scripts/prepare_gemma_from_common.py",
    "scripts/prepare_qwen_remote.py",
    "scripts/qwen_local_tokenizer.py",
    "scripts/benchmark_paid_rates.py",
    "scripts/run_paid_concurrent.py",
    "scripts/update_progress_metrics.py",
    "configs/repository-layout.json",
}


def layout_files(files):
    """Normaliza apenas as realocacoes e hashes explicitamente revisados.

    Um hash antigo desconhecido nao e convertido. Assim, mudar a logica de
    preparacao ou os insumos cientificos nao se torna compativel por renomear.
    """
    path = Path("configs/repository-layout.json")
    if not path.exists():
        return dict(files)
    layout = json.loads(path.read_text())
    moves = layout["moves"]
    legacy = any(name in moves and moves[name]["path"] != name for name in files)
    result = {}
    for name, sha in files.items():
        move = moves.get(name)
        if move:
            name = move["path"]
            if sha == move["before_sha256"]:
                sha = move["after_sha256"]
        if name in result:
            raise ValueError("lock contem caminhos antigos e novos para o mesmo modulo")
        result[name] = sha
    if legacy:
        result.update(layout["package_initializers"])
    return result


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
    before = layout_files(previous["files"])
    after = layout_files(current["files"])
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    operational = set(layout_files(dict.fromkeys(OPERATIONAL_COMPATIBILITY, "operational")))
    # Novos __init__ sao permitidos somente com o hash revisado acima, nao por
    # pertencerem a lista de scripts operacionais.
    operational = {p for p in operational if not p.endswith("/__init__.py")}
    return changed <= operational


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
                or not set(report.get("models", {}))
                or not set(report["models"])
                <= set(
                    data.get("compatible_prepared_models", {}).get(
                        prepared_lock["lock_id"], ["ling-free"]
                    )
                )
            ):
                raise ValueError("preparacao pertence a outro lock")
    # Evita atualizacao automatica de main nos modelos avaliadores/tokenizers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_OFFLINE = True
    return data
