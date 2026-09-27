"""Gera guia HTML offline a partir do codigo e dos artefatos de dev existentes.

Nao carrega documentos eval, nao consulta API, nao le credenciais.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from findsum_rag.data import Task
from findsum_rag.prompts import SYSTEM_PROMPT, TASK_INSTRUCTIONS

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    manifest = read("data/interim/splits-liquidity.json")
    common = read("outputs/dev50-evidence-prepared-20260926/documents.json")
    chosen = [d for d in common if d["doc_id"].split("-")[0] in {"ABEO", "ACET", "ADEX"}]
    ids = {d["doc_id"] for d in chosen}
    model_specs = {
        "ling": (
            "Ling 3.0 Flash Fin",
            "Novita · gratuito",
            "dev50-evidence-prepared-20260926/ling-free",
            "dev50-ling-evidence-20260926/ling",
        ),
        "gemma26": (
            "Gemma 4 26B A4B",
            "Darkbloom · pago",
            "dev50-evidence-prepared-20260926/gemma26",
            "dev50-gemma26-evidence-20260926/results",
        ),
        "qwen37": (
            "Qwen3.7 Flash",
            "Alibaba · pago",
            "dev50-qwen37-evidence-prepared-20260926/qwen37",
            "dev50-qwen37-evidence-20260926/results",
        ),
    }
    models = {}
    for model, (label, provider, prepared, result) in model_specs.items():
        rows = read(f"outputs/{prepared}/preflight.json")
        outputs = {}
        for arm in ("C1", "C1t", "C2", "C3", "C4", "C5"):
            for line in (
                (ROOT / f"outputs/{result}/{arm}/predictions.jsonl").read_text().splitlines()
            ):
                prediction = json.loads(line)
                if prediction["doc_id"] in ids:
                    outputs[prediction["doc_id"] + "/" + arm] = prediction["prediction"]
        models[model] = {
            "label": label,
            "provider": provider,
            "rows": [r for r in rows if r["doc_id"] in ids],
            "outputs": outputs,
            "max_input_observed": max(r["prompt_tokens"] for r in rows),
            "preflight_path": f"outputs/{prepared}/preflight.json",
        }
    audit = read("outputs/dev50-ling-evidence-20260926/table-audit.json")
    cells = [c for d in audit for t in d["tables"] for c in t["cells"]]
    reasons = dict(Counter(reason for cell in cells for reason in cell["reasons"]))
    paths = [
        "README.md",
        "docs/metricas-automaticas.md",
        "configs/dev_openrouter_ling_evidence.yaml",
        "src/findsum_rag/data.py",
        "src/findsum_rag/splits.py",
        "src/findsum_rag/chunking.py",
        "src/findsum_rag/retrieval.py",
        "src/findsum_rag/examples.py",
        "src/findsum_rag/prompts.py",
        "src/findsum_rag/context.py",
        "src/findsum_rag/pipeline.py",
        "src/findsum_rag/metrics.py",
        "src/findsum_rag/analyze.py",
        "scripts/execution/run_prepared_paid.py",
        "scripts/preparation/prepare_qwen_remote.py",
        "scripts/preparation/prepare_full_openrouter.py",
        "scripts/common/full_common.py",
        "configs/full_openrouter.yaml",
        "configs/openrouter_full.json",
        "data/interim/splits-liquidity.json",
        "outputs/dev50-evidence-comparison-20260926/validation.json",
    ]
    data = {
        "snapshot": "26/09/2026",
        "system": SYSTEM_PROMPT,
        "task": TASK_INSTRUCTIONS[Task.LIQUIDITY],
        "manifest": {k: v for k, v in manifest.items() if k != "sets"},
        "splits": {
            k: {
                "documents": len(v),
                "companies": len({x["stock_name"] for x in v}),
                "original_splits": dict(Counter(x["split"] for x in v)),
            }
            for k, v in manifest["sets"].items()
        },
        "documents": chosen,
        "models": models,
        "table_audit": {
            "total": len(cells),
            "eligible": sum(c["eligible"] for c in cells),
            "reasons": reasons,
            "empty_columns": sum(not str(c["cell"][1]).strip() for c in cells),
            "examples": [*cells[:2], next(c for c in cells if "&" not in str(c["cell"][2]))],
        },
        "results": read("outputs/dev50-evidence-comparison-20260926/validation.json"),
        "full_plan": read("configs/openrouter_full.json"),
        "provenance": [
            {"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
            for p in paths
        ],
    }
    # Conteudo do corpus nao pode encerrar o elemento script nem criar HTML executavel.
    payload = (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = (ROOT / "scripts/reports/templates/pipeline.html").read_text()
    assert template.count("__PIPELINE_DATA__") == 1
    output = ROOT / "outputs/reports/pipeline.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__PIPELINE_DATA__", payload))
    print(f"{output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
