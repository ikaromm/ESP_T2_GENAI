"""Congela configuracao, codigo, dados e modelos locais. Nao executa a pipeline."""

import argparse
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from findsum_rag.full_lock import (
    DEFAULT_LOCK,
    compatible_preparation,
    digest,
    lock_digest,
    verify_full_lock,
)
from full_common import random_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--compatible-prepared-lock", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.verify:
        result = verify_full_lock()
        print("Lock verificado:", result["lock_id"], "; nenhuma chamada API")
        return
    if DEFAULT_LOCK.exists():
        raise ValueError("Lock existente: nao sobrescrever silenciosamente")
    from findsum_rag.analyze import ANALYSIS_PLAN
    from findsum_rag.config import ExperimentConfig
    from findsum_rag.metrics import SUPPLEMENTARY_METRICS

    cfg = ExperimentConfig.from_yaml("configs/full_openrouter.yaml")
    plan = json.loads(Path("configs/openrouter_full.json").read_text())
    manifest = json.loads(cfg.data.manifest.read_text())
    cohort = random_records(manifest["sets"]["eval"], 1000, seed=42, split="eval")
    assert len(cohort) == len({r["stock_name"] for r in cohort}) == 1000
    import csv

    import nltk

    with Path("configs/full-eval-cohort.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order", *cohort[0]])
        writer.writeheader()
        writer.writerows(dict(order=i, **r) for i, r in enumerate(cohort, 1))
    files = list(Path("src").rglob("*.py")) + list(Path("scripts").glob("*.py"))
    files += [Path("rodar_rodada.sh")]
    files += [
        Path(x)
        for x in [
            "configs/full_openrouter.yaml",
            "configs/openrouter_full.json",
            "configs/full-eval-cohort.csv",
            "pyproject.toml",
            "uv.lock",
            str(cfg.data.manifest),
        ]
    ]
    assets = {}
    for model in [
        cfg.retrieval.embedding_model,
        "xlnet-base-cased",
        "inclusionAI/Ling-3.0-flash-Fin",
        "google/gemma-4-26B-A4B-it",
    ]:
        base = Path(HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))
        revision = (base / "refs/main").read_text().strip()
        snapshot = base / "snapshots" / revision
        assets[model] = {
            "revision": revision,
            "files": {
                str(p.relative_to(snapshot)): digest(p)
                for p in sorted(snapshot.rglob("*"))
                if p.is_file()
            },
        }
    dataset = sorted(
        p
        for p in cfg.data.root.rglob("*")
        if p.is_file() and ("FINDSum-Liquidity" in str(p)) and p.suffix in {".csv", ".txt"}
    )
    if not dataset:
        raise ValueError("Nenhum arquivo FINDSum Liquidity encontrado")
    if any(not spec["files"] for spec in assets.values()):
        raise ValueError("Snapshot local vazio")
    result = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "state": "frozen_not_executed",
        "generation_backend": "openrouter_only",
        "files": {str(p): digest(p) for p in sorted(files)},
        "dataset_files": {str(p): digest(p) for p in dataset},
        "hf_assets": assets,
        "wordnet_sha256": digest(nltk.data.find("corpora/wordnet.zip").zipfile.filename),
        "cohort": cohort,
        "analysis_plan": ANALYSIS_PLAN,
        "supplementary_metrics": SUPPLEMENTARY_METRICS,
        "budgets_usd": plan["budgets_usd"],
        "versions": {
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "transformers",
                "sentence-transformers",
                "bert-score",
                "nltk",
                "rouge-score",
                "numpy",
                "scipy",
                "faiss-cpu",
                "pydantic",
                "tokenizers",
            ]
        },
    }
    result = json.loads(json.dumps(result))
    for prepared_lock in args.compatible_prepared_lock:
        previous = json.loads(prepared_lock.read_text())
        if not compatible_preparation(previous, result):
            raise ValueError("mudanca cientifica impede reaproveitar preparacao anterior")
        report = json.loads((prepared_lock.parent / "report.json").read_text())
        models = set(report.get("models", {}))
        if not models or not models <= {"ling-free", "gemma26", "qwen37"}:
            raise ValueError("preparacao local sem modelos reconhecidos")
        if "qwen37" in models:
            from qwen_local_tokenizer import load_qwen_tokenizer

            load_qwen_tokenizer(prepared_lock.parent / "qwen37" / "tokenizer")
        ids = result.setdefault("compatible_prepared_locks", [])
        if previous["lock_id"] not in ids:
            ids.append(previous["lock_id"])
        allowed = result.setdefault("compatible_prepared_models", {})
        allowed[previous["lock_id"]] = sorted(set(allowed.get(previous["lock_id"], [])) | models)
    result["lock_id"] = lock_digest(result)
    DEFAULT_LOCK.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("Congelado:", result["lock_id"], len(cohort), "casos; 0 chamadas API")


if __name__ == "__main__":
    main()
