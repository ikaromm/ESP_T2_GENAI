"""Triagem gratuita Qwen/Ling/Gemma: Documentos dev x 6 bracos, sem tocar eval.

Salva cada caso aceito imediatamente. Um 429 ou erro de contrato interrompe
somente aquele modelo; saidas cortadas sao registradas e a triagem continua.
Retoma casos salvos com --resume; retries de 429 limitados. Sem testes de hipoteses.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from findsum_rag.config import ExperimentConfig, GenerationConfig
from findsum_rag.context import token_count
from findsum_rag.examples import make_selector
from findsum_rag.metrics import RougeScorer, aggregate, score_document
from findsum_rag.openrouter import OpenRouterHTTPError, OpenRouterSummarizer
from findsum_rag.pipeline import (
    ArmResult,
    _write_arm,
    arm_prompt,
    build_example_store,
    prepare_contexts,
    prepare_documents,
    run_arm,
)
from findsum_rag.remote_models import FREE_ENDPOINTS
from findsum_rag.retrieval import SentenceTransformerEncoder
from findsum_rag.splits import SplitManifest, load_set

MODELS = {
    "ling": "inclusionai/ling-3.0-flash-fin:free",
    "qwen": "qwen/qwen3.8-27b:free",
    "gemma": "google/gemma-4-31b-it:free",
}


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class RequestWindow:
    """Janela compartilhada entre modelos/retries deste processo sequencial."""

    def __init__(self, path, *, limit=20, period=60, clock=time.time, sleep=time.sleep):
        self.path, self.limit, self.period = path, limit, period
        self.clock, self.sleep = clock, sleep
        self.sent = json.loads(path.read_text()) if path.exists() else []

    def acquire(self):
        while True:
            now = self.clock()
            self.sent = [t for t in self.sent if t > now - self.period]
            if len(self.sent) < self.limit:
                self.sent.append(now)
                save(self.path, self.sent)
                return
            delay = max(0, min(self.sent) + self.period - now) + 0.05
            print(f"Limite de {self.limit} chamadas/{self.period}s: aguardando {delay:.2f}s",
                  flush=True)
            self.sleep(delay)


class FreeQuotaLimiter:
    """Conta tentativas nesta sessao; nao exige seis vezes o lote na cota."""

    def __init__(self, window, remaining):
        self.window = window
        self.remaining = remaining

    def acquire(self):
        if self.remaining <= 0:
            raise ValueError("cota gratuita diaria esgotada; retome apenas as pendencias")
        self.window.acquire()
        self.remaining -= 1


def with_retries(
    operation, case, checkpoint, *, max_attempts=6, sleep=time.sleep, limiter=None,
):
    """Seis tentativas, com esperas de 1, 2, 4, 8 e 16s; respeita Retry-After."""
    for attempt in range(max_attempts):
        if limiter is not None:
            limiter.acquire()
        try:
            return operation()
        except OpenRouterHTTPError as exc:
            delay = exc.retry_delay(2**attempt)
            case.setdefault("attempt_errors", []).append({
                "at": datetime.now(UTC).isoformat(), "error": str(exc),
                "retry_delay_seconds": delay,
            })
            checkpoint()
            if exc.status != 429 or attempt == max_attempts - 1 or delay > 60:
                raise
            print(f"HTTP 429: nova tentativa em {delay:g}s", flush=True)
            sleep(delay)


def restore_results(out, cfg, preflight, rouge):
    """So reaproveita resultados com contexto/exemplos/contagens do preflight atual."""
    expected = {(p["doc_id"], p["arm"]): p for p in preflight}
    results, completed = {}, set()
    for arm in cfg.arms:
        path = out / arm.id / "predictions.jsonl"
        if not path.exists():
            continue
        result = ArmResult(arm, [], [], {})
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["doc_id"], row["arm"])
            if key in completed or key not in expected or row["arm"] != arm.id:
                raise ValueError("caso salvo duplicado ou fora da triagem")
            for field in ("context", "source", "example_ids", "context_tokens", "prompt_tokens"):
                if row[field] != expected[key][field]:
                    raise ValueError(f"caso salvo diverge do preflight: {key}/{field}")
            meta = row["generation_metadata"]
            endpoint = FREE_ENDPOINTS[cfg.generation.model_name]
            if (meta["model"] != cfg.generation.model_name
                    or meta["provider"] != endpoint.response_provider
                    or meta["finish_reason"] != "stop" or row["truncated_prompt"]
                    or meta["usage"].get("cost") != 0):
                raise ValueError("contrato da geracao salva difere da configuracao")
            score = score_document(
                row["doc_id"], row["prediction"], row["reference"], row["source"], rouge=rouge
            )
            result.predictions.append(row)
            result.scores.append(score)
            completed.add(key)
        result.summary = aggregate(result.scores)
        results[arm.id] = result
    return results, completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/poc_openrouter.yaml"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--n-docs", type=int, default=3, choices=range(1, 51),
                        metavar="1..50", help="Numero de documentos dev (padrao: 3)")
    args = parser.parse_args()
    if args.resume:
        report = json.loads((args.output / "report.json").read_text())
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        report = None
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_eval_docs = args.n_docs
    if cfg.data.eval_set != "dev" or cfg.data.example_set != "examples":
        raise ValueError("triagem limitada a dev e examples")
    manifest = SplitManifest.load(cfg.data.manifest)
    if manifest.task != cfg.data.task:
        raise ValueError("tarefa difere do manifesto")
    documents = load_set(cfg.data.root, manifest, "dev", limit=args.n_docs)
    table_audit = [
        {"doc_id": d.doc_id, "tables": [
            {"section": t.section, "section_index": t.section_index,
             "cells": t.evidence_audit()} for t in d.task_tables()
        ]} for d in documents
    ]
    audit_path = args.output / "table-audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text()) != table_audit:
        raise ValueError("evidencia tabular mudou; use outra rodada")
    save(audit_path, table_audit)
    encoder = SentenceTransformerEncoder(cfg.retrieval.embedding_model)
    print(f"Preparando {args.n_docs} documentos dev e indice local de exemplos", flush=True)
    prepared = prepare_documents(
        documents, encoder, chunk_size=cfg.retrieval.chunk_size,
        chunk_overlap=cfg.retrieval.chunk_overlap, include_tables=cfg.retrieval.include_tables,
    )
    store = build_example_store(
        cfg.data.root, manifest, "examples", limit=cfg.data.n_example_docs,
        max_summary_words=cfg.data.example_max_summary_words,
    )
    store.build_index(encoder)
    new_report = {
        "created_at": datetime.now(UTC).isoformat(), "scope": "screening_dev_only",
        "documents": [p.doc_id for p in prepared], "example_pool_size": len(store),
        "bertscore_computed": False, "evaluation_scope": "reference_similarity_only", "models": {},
    }
    if report is None:
        report = new_report
    elif any(report[k] != new_report[k] for k in ("scope", "documents", "example_pool_size")):
        raise ValueError("escopo da retomada difere da rodada salva")
    save(args.output / "report.json", report)
    rouge = RougeScorer()
    window_path = args.output / "request-window.json"
    if not window_path.exists():
        # Migra tentativas da versao anterior ainda dentro da janela atual.
        save(window_path, [
            p.stat().st_mtime for p in args.output.glob('*/api/*.request.json')
            if p.stat().st_mtime > time.time() - 60
        ])
    limiter = RequestWindow(window_path)
    for name in args.models:
        model = MODELS[name]
        out = args.output / name
        out.mkdir(exist_ok=args.resume)
        cfg.name = name
        cfg.output_dir = args.output
        cfg.generation = GenerationConfig(
            **(cfg.generation.model_dump() | {"model_name": model})
        )
        if (out / "config.yaml").exists():
            saved = ExperimentConfig.from_yaml(out / "config.yaml")
            if saved.model_dump() != cfg.model_dump():
                raise ValueError("configuracao mudou; use outra rodada")
        else:
            cfg.to_yaml(out / "config.yaml")
        status = report["models"].setdefault(
            name, {"model": model, "cases": [], "status": "preparing"}
        )
        summarizer = OpenRouterSummarizer(cfg.generation, audit_dir=out / "api")
        results = {}
        try:
            preflight = []
            for item in prepared:
                prepare_contexts(item, cfg, summarizer)
                for arm in cfg.arms:
                    selector = make_selector(
                        arm.example_strategy, store=store,
                        n_examples=arm.n_examples, seed=cfg.seed,
                    )
                    prompt, examples, context = arm_prompt(item, arm, cfg, summarizer, selector)
                    if len(examples) != arm.n_examples:
                        raise ValueError("numero de exemplos difere do protocolo")
                    preflight.append({
                        "arm": arm.id, "doc_id": item.doc_id,
                        "example_ids": [e.doc_id for e in examples],
                        "context": context, "source": item.source_text,
                        "context_tokens": token_count(summarizer.tokenizer, context),
                        "prompt_tokens": summarizer.count_tokens(prompt),
                        "messages": prompt.as_messages(),
                    })
            if (out / "preflight.json").exists():
                if json.loads((out / "preflight.json").read_text()) != preflight:
                    raise ValueError("prompts da retomada divergem dos prompts salvos")
            else:
                save(out / "preflight.json", preflight)
            results, completed = restore_results(out, cfg, preflight, rouge)
            for doc_id, arm_id in sorted(completed):
                old_case = next((c for c in status["cases"]
                                 if (c["doc_id"], c["arm"]) == (doc_id, arm_id)), None)
                if old_case is None:
                    status["cases"].append({
                        "doc_id": doc_id, "arm": arm_id, "status": "accepted",
                    })
                else:
                    old_case["status"] = "accepted"
            pending = len(preflight) - len(completed)
            if not pending:
                status["status"] = "complete"
                print(f"{name}: {len(completed)} casos reaproveitados, nenhuma chamada", flush=True)
                continue
            status.pop("error", None)
            status["quota_before_resume"] = summarizer.client.quota()
            remaining = (status["quota_before_resume"].get("free_model_daily_requests") or {}).get(
                "remaining", 0
            )
            quota_limiter = FreeQuotaLimiter(limiter, remaining)
            summarizer._calls = max(
                (int(p.name.split('.')[0]) for p in (out / "api").glob('*.request.json')),
                default=0,
            )
            status["status"] = "generating"
            save(args.output / "report.json", report)
            for item in prepared:
                for arm in cfg.arms:
                    if (item.doc_id, arm.id) in completed:
                        continue
                    case = next((c for c in status["cases"]
                                 if c["doc_id"] == item.doc_id and c["arm"] == arm.id), None)
                    if case is None:
                        case = {"doc_id": item.doc_id, "arm": arm.id}
                        status["cases"].append(case)
                    if "error" in case:
                        case.setdefault("previous_errors", []).append(case.pop("error"))
                    case["status"] = "running"
                    save(args.output / "report.json", report)
                    print(f"{name} {item.doc_id} {arm.id}", flush=True)
                    try:
                        result = with_retries(
                            partial(run_arm,
                                arm, [item], config=cfg, summarizer=summarizer,
                                store=store, rouge=rouge,
                            ), case, lambda: save(args.output / "report.json", report),
                            limiter=quota_limiter,
                        )
                    except Exception as exc:
                        case.update(status="rejected", error=str(exc))
                        save(args.output / "report.json", report)
                        if "geracao incompleta" not in str(exc):
                            raise
                    else:
                        case["status"] = "accepted"
                        previous = results.setdefault(arm.id, ArmResult(arm, [], [], {}))
                        previous.scores.extend(result.scores)
                        previous.predictions.extend(result.predictions)
                        previous.summary = aggregate(previous.scores)
                        _write_arm(out, previous)
                        save(args.output / "report.json", report)
                    time.sleep(4)  # abaixo de 20 chamadas/minuto, mesmo com respostas rapidas
            status["status"] = (
                "complete" if all(c["status"] == "accepted" for c in status["cases"])
                else "incomplete"
            )
        except Exception as exc:
            status.update(status="blocked", error=str(exc))
            print(f"{name}: {exc}", flush=True)
        finally:
            save(out / "summary.json", {
                "scope": "screening_dev_only", "status": status["status"],
                "arms": {k: v.summary for k, v in results.items()},
            })
            save(args.output / "report.json", report)
    print(f"Relatorio: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
