"""Duas chamadas reais no dev: fonte inteira, sem/com quatro exemplos fixos.

PoC de integracao, nao rodada de ablacao nem avaliacao das hipoteses.
Nao carrega pesos locais e nunca acessa o conjunto eval.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from findsum_rag.chunking import Chunk, chunk_document
from findsum_rag.config import ExperimentConfig
from findsum_rag.data import clean_text
from findsum_rag.openrouter import MODEL, PROVIDER, OpenRouterFreeClient
from findsum_rag.pipeline import build_example_store
from findsum_rag.prompts import build_prompt
from findsum_rag.splits import SplitManifest, load_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="repete apenas o ultimo caso que falhou, usando o prompt salvo",
    )
    args = parser.parse_args()
    key = os.environ.get("OPEN_ROUTER_KEY")
    if not key and args.env_file:
        for line in args.env_file.read_text().splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and name.strip() == "OPEN_ROUTER_KEY":
                key = value.strip().strip("\"'")
    client = OpenRouterFreeClient(key or "")
    if args.retry_failed:
        path = args.output / "report.json"
        report = json.loads(path.read_text())
        failure = report.get("error")
        if not failure or failure["case"] not in {"full_zero_shot", "full_four_fixed_examples"}:
            raise RuntimeError("nenhum caso pendente para repetir")
        name = failure["case"]
        if any(case["case"] == name for case in report["cases"]):
            raise RuntimeError("caso ja concluido; nao repetir")
        request = json.loads((args.output / f"{name}.request.json").read_text())
        if request != client.payload(request["messages"], request["max_tokens"]):
            raise RuntimeError("payload salvo difere da configuracao gratuita da PoC")
        report["quota_before_retry"] = client.quota()
        print(json.dumps(report["quota_before_retry"]), flush=True)
        try:
            result = client.complete(request["messages"], max_tokens=request["max_tokens"])
        except Exception as exc:
            report.setdefault("retry_errors", []).append(
                {
                    "case": name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "at": datetime.now(UTC).isoformat(),
                }
            )
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            raise
        result["case"] = name
        report["cases"].append(result)
        report.setdefault("previous_errors", []).append(report.pop("error"))
        report["quota_after"] = client.quota()
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (args.output / f"{name}.txt").write_text(result["text"])
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ["case", "provider", "finish_reason", "usage", "elapsed_seconds"]
                }
            ),
            flush=True,
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = ExperimentConfig.from_yaml("configs/test_50.yaml")
    manifest = SplitManifest.load(cfg.data.manifest)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "provider_requested": PROVIDER,
        "scope": "integration_poc_dev_only",
        "quota_before": client.quota(),
        "cases": [],
        "limitations": [
            "nao executa RAG/C1t/C5",
            "nao testa hipoteses",
            "seed nao suportada pelo endpoint; nao enviada",
        ],
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    save()
    quota = report["quota_before"].get("free_model_daily_requests") or {}
    if quota.get("remaining", 0) < 2:
        raise RuntimeError("cota insuficiente para duas chamadas; nenhuma geracao iniciada")
    doc = load_set(cfg.data.root, manifest, "dev", limit=1)[0]
    chunks = chunk_document(
        doc,
        chunk_size=cfg.retrieval.chunk_size,
        overlap=cfg.retrieval.chunk_overlap,
        include_tables=True,
    )
    source = "\n\n".join(
        [clean_text(doc.document)] + [c.text for c in chunks if c.source == "table"]
    )
    store = build_example_store(cfg.data.root, manifest, "examples")
    examples = sorted(store.examples, key=lambda example: example.doc_id)[:4]
    if len(examples) != 4:
        raise RuntimeError("base insuficiente para quatro exemplos")
    for name, selected in [("full_zero_shot", []), ("full_four_fixed_examples", examples)]:
        prompt = build_prompt(
            task=doc.task,
            arm=cfg.arm("C1"),
            context_chunks=[Chunk(doc.doc_id, 0, source)],
            context_text=source,
            examples=selected,
            example_max_words=cfg.data.example_max_words,
        )
        request = client.payload(prompt.as_messages())
        (args.output / f"{name}.request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2)
        )
        print(f"Chamando {name}: documento dev={doc.doc_id}, exemplos={len(selected)}", flush=True)
        try:
            result = client.complete(prompt.as_messages())
        except Exception as exc:
            report["error"] = {"case": name, "type": type(exc).__name__, "message": str(exc)}
            save()
            raise
        result.update(
            {
                "case": name,
                "doc_id": doc.doc_id,
                "example_ids": [example.doc_id for example in selected],
                "reference": clean_text(doc.summary),
            }
        )
        report["cases"].append(result)
        (args.output / f"{name}.txt").write_text(result["text"])
        save()
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ["case", "provider", "finish_reason", "usage", "elapsed_seconds"]
                }
            ),
            flush=True,
        )
    report["quota_after"] = client.quota()
    save()
    print(f"Relatorio: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
