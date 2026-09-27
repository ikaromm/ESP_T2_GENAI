"""Um caso dev sorteado, seis bracos Ling free e rastro completo; nunca eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from findsum_rag.analyze import ANALYSIS_PLAN
from findsum_rag.config import ExperimentConfig
from findsum_rag.context import token_count
from findsum_rag.data import clean_text
from findsum_rag.examples import make_selector
from findsum_rag.metrics import RougeScorer, aggregate, bertscore
from findsum_rag.openrouter import OpenRouterSummarizer
from findsum_rag.pipeline import (
    _write_arm,
    arm_prompt,
    build_example_store,
    prepare_contexts,
    prepare_documents,
    run_arm,
)
from findsum_rag.prompts import TASK_INSTRUCTIONS
from findsum_rag.retrieval import SentenceTransformerEncoder, VectorIndex
from findsum_rag.splits import DocumentRef, SplitManifest, load_set
from scripts.common.full_common import Limits, random_records
from scripts.experiments.screen_openrouter import (
    FreeQuotaLimiter,
    RequestWindow,
    save,
    with_retries,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    # Ler somente a variavel necessaria, sem imprimir ou copiar a credencial.
    if not os.environ.get("OPEN_ROUTER_KEY"):
        for line in Path("../.env").read_text().splitlines():
            if line.startswith("OPEN_ROUTER_KEY="):
                os.environ["OPEN_ROUTER_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
                break
    cfg = ExperimentConfig.from_yaml("configs/dev_openrouter_ling_evidence.yaml")
    cfg.data.n_eval_docs = 1
    cfg.generation.max_new_tokens = 8192
    cfg.generation.max_input_tokens = 49152
    cfg.to_yaml(out / "config.yaml")
    manifest = SplitManifest.load(cfg.data.manifest)
    selected = random_records(
        [r.model_dump() for r in manifest.sets["dev"]], 1, seed=cfg.seed, split="dev"
    )
    selection = {
        "split": "dev",
        "documents": selected,
        "seed": cfg.seed,
        "method": "random_without_replacement",
        "manifest_sha256": hashlib.sha256(cfg.data.manifest.read_bytes()).hexdigest(),
    }
    save(out / "selection.json", selection)
    copied = manifest.model_copy(deep=True)
    copied.sets["dev"] = [DocumentRef(**selected[0])]
    doc = load_set(cfg.data.root, copied, "dev")[0]
    print("Documento sorteado:", doc.doc_id, flush=True)
    encoder = SentenceTransformerEncoder(cfg.retrieval.embedding_model)
    item = prepare_documents(
        [doc],
        encoder,
        chunk_size=cfg.retrieval.chunk_size,
        chunk_overlap=cfg.retrieval.chunk_overlap,
        include_tables=True,
    )[0]
    print("Indexando os 1000 exemplos localmente", flush=True)
    store = build_example_store(cfg.data.root, manifest, "examples", limit=None)
    store.build_index(encoder)
    llm = OpenRouterSummarizer(cfg.generation, audit_dir=out / "api")
    prepare_contexts(item, cfg, llm)
    index = VectorIndex(item.chunk_vectors.shape[1])
    index.add(item.chunk_vectors, list(item.chunks))
    hits = index.search(item.retrieval_vector, cfg.retrieval.top_k)
    trace = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "preparing",
        "scope": "one_dev_document_six_arms_no_hypothesis_test",
        "selection": selection,
        "config": json.loads(cfg.model_dump_json()),
        "analysis_plan": ANALYSIS_PLAN,
        "raw_document": asdict(doc),
        "source": item.source_text,
        "reference": clean_text(doc.summary),
        "retrieval_query": TASK_INSTRUCTIONS[doc.task],
        "chunks": [asdict(c) for c in item.chunks],
        "retrieval": [
            {"rank": i, "score": float(h.score), "chunk": asdict(h.payload)}
            for i, h in enumerate(hits, 1)
        ],
        "similar_examples": [
            {"rank": i, "score": float(h.score), "doc_id": h.payload.doc_id}
            for i, h in enumerate(store.index.search(item.query_vector, 8), 1)
        ],
        "arms": [],
        "metrics_status": "pending",
    }
    for arm in cfg.arms:
        selector = make_selector(
            arm.example_strategy, store=store, n_examples=arm.n_examples, seed=cfg.seed
        )
        prompt, examples, context = arm_prompt(item, arm, cfg, llm, selector)
        count = llm.count_tokens(prompt)
        Limits(
            cfg.generation.max_input_tokens,
            cfg.generation.max_new_tokens,
            llm.endpoint.context_length,
        ).check(count)
        trace["arms"].append(
            {
                "arm": arm.id,
                "label": arm.label,
                "context": context,
                "context_tokens": token_count(llm.tokenizer, context),
                "prompt_tokens": count,
                "messages": prompt.as_messages(),
                "examples": [asdict(e) for e in examples],
                "status": "pending",
            }
        )
    assert trace["arms"][1]["context_tokens"] == trace["arms"][2]["context_tokens"]
    assert trace["arms"][1]["prompt_tokens"] == trace["arms"][2]["prompt_tokens"]
    assert len({a["context"] for a in trace["arms"][2:]}) == 1
    save(out / "trace.json", trace)
    save(out / "analysis_plan.json", ANALYSIS_PLAN)
    print("Seis prompts validados. Consultando cota gratuita.", flush=True)
    quota = llm.client.quota()
    trace["quota_before"] = quota
    remaining = (quota.get("free_model_daily_requests") or {}).get("remaining", 0)
    limiter = FreeQuotaLimiter(RequestWindow(out / "request-window.json"), remaining)
    results = []
    for arm, row in zip(cfg.arms, trace["arms"], strict=True):
        print("Gerando", arm.id, "entrada:", row["prompt_tokens"], flush=True)
        row["status"] = "running"
        save(out / "trace.json", trace)
        try:
            result = with_retries(
                partial(
                    run_arm,
                    arm,
                    [item],
                    config=cfg,
                    summarizer=llm,
                    store=store,
                    rouge=RougeScorer(),
                ),
                row,
                lambda: save(out / "trace.json", trace),
                limiter=limiter,
            )
            row.update(
                status="accepted", prediction=result.predictions[0], scores=result.scores[0].flat()
            )
            _write_arm(out, result)
            results.append(result)
            print(arm.id, "concluido", flush=True)
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            print(arm.id, "falhou:", str(exc), flush=True)
        save(out / "trace.json", trace)
    if results:
        print("Calculando BERTScore local (roberta-large)", flush=True)
        try:
            f1s = bertscore(
                [r.predictions[0]["prediction"] for r in results],
                [trace["reference"]] * len(results),
                batch_size=1,
            )
            for result, f1 in zip(results, f1s, strict=True):
                result.scores[0].bertscore = f1
                result.summary = aggregate(result.scores)
                _write_arm(out, result)
                next(a for a in trace["arms"] if a["arm"] == result.arm.id)["scores"] = (
                    result.scores[0].flat()
                )
            trace["metrics_status"] = "complete"
        except Exception as exc:
            trace.update(metrics_status="failed", metrics_error=str(exc))
    trace["status"] = (
        "complete" if len(results) == 6 and trace["metrics_status"] == "complete" else "incomplete"
    )
    trace["finished_at"] = datetime.now(UTC).isoformat()
    save(out / "trace.json", trace)
    print("Estado final:", trace["status"], flush=True)


if __name__ == "__main__":
    main()
