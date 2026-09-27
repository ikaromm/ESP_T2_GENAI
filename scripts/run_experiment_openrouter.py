"""Executa uma coorte preparada completa nos tres modelos; retomada e metricas integrais.

Prepara separadamente com prepare_full_openrouter e prepare_qwen_remote.
Sem --execute, somente valida e produz o relatorio de elegibilidade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from findsum_rag.analyze import ANALYSIS_PLAN, compare_hypotheses
from findsum_rag.config import ExperimentConfig
from findsum_rag.data import clean_text
from findsum_rag.generate import clean_generation
from findsum_rag.metrics import (
    BERTSCORE_MODEL,
    SUPPLEMENTARY_METRICS,
    aggregate,
    bertscore_components,
    bertscore_token_audit,
    ensure_meteor_resources,
    meteor,
    score_document,
)
from findsum_rag.openrouter import OpenRouterFreeClient
from findsum_rag.pipeline import ArmResult, _write_arm
from full_common import FULL_MODELS, Limits, eligibility, selected_records
from run_prepared_paid import TARGETS, load_prepared, money, run
from screen_openrouter import save


def load_key():
    if not os.environ.get("OPEN_ROUTER_KEY"):
        for line in Path("../.env").read_text().splitlines():
            if line.startswith("OPEN_ROUTER_KEY="):
                os.environ["OPEN_ROUTER_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
                break
    return os.environ["OPEN_ROUTER_KEY"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--qwen-prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-eval", action="store_true")
    parser.add_argument("--qwen-budget", type=money, default=money("18"))
    parser.add_argument("--gemma-budget", type=money, default=money("9"))
    args = parser.parse_args()
    split, records = selected_records(args.prepared)
    if split == "eval" and not args.allow_eval:
        raise ValueError("eval exige --allow-eval explicito")
    frozen = None
    if split == "eval":
        from findsum_rag.full_lock import verify_full_lock

        frozen = verify_full_lock(prepared=args.prepared)
        verify_full_lock(prepared=args.qwen_prepared, check_assets=False)
        for supplied, name in [
            (args.qwen_budget, "qwen_generation"),
            (args.gemma_budget, "gemma_generation"),
        ]:
            if supplied > money(frozen["budgets_usd"][name]):
                raise ValueError("orcamento excede o teto congelado")
    if selected_records(args.qwen_prepared) != (split, records):
        raise ValueError("coortes dos modelos divergem")
    for filename in ("config.yaml", "documents.json", "examples.json"):
        if (args.prepared / filename).read_bytes() != (args.qwen_prepared / filename).read_bytes():
            raise ValueError(f"insumos dos modelos divergem: {filename}")
    cfg = ExperimentConfig.from_yaml(args.prepared / "config.yaml")
    folders = {m: args.qwen_prepared if m == "qwen37" else args.prepared for m in FULL_MODELS}
    rows = {m: load_prepared(folders[m], m, len(records)) for m in FULL_MODELS}
    plan = json.loads(Path("configs/openrouter_full.json").read_text())
    limits = {
        m: Limits(
            cfg.generation.max_input_tokens,
            cfg.generation.max_new_tokens,
            plan["models"][m]["context_length"],
        )
        for m in FULL_MODELS
    }
    eligible = eligibility(
        [r["doc_id"] for r in records],
        rows,
        limits,
        expected_count=1000 if split == "eval" else len(records),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "eligibility.json", eligible)
    if not eligible["ready"]:
        raise ValueError("coorte incompleta; nao iniciar geracoes")
    if not args.execute:
        print("Preflight validado; nenhuma chamada de geracao.")
        return
    if frozen:
        for model in FULL_MODELS:
            target = TARGETS[model]
            contract = json.loads(Path("configs/openrouter_full.json").read_text())["models"][model]
            if (target[0], target[1]) != (contract["model"], contract["provider"]):
                raise ValueError("endpoint difere do congelado")
    ensure_meteor_resources()
    key = load_key()
    transport = OpenRouterFreeClient(key, timeout=240)
    quota = transport.quota()
    save(args.output / "quota-before.json", quota)
    budgets = {"ling-free": money(0), "qwen37": args.qwen_budget, "gemma26": args.gemma_budget}
    root_identity = {
        "split": split,
        "documents": [r["doc_id"] for r in records],
        "config": json.loads(cfg.model_dump_json()),
        "analysis_plan": ANALYSIS_PLAN,
    }
    if frozen:
        root_identity["full_lock_id"] = frozen["lock_id"]
    identity_file = args.output / "identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != json.loads(
        json.dumps(root_identity)
    ):
        raise ValueError("identidade de retomada alterada")
    save(identity_file, root_identity)

    def execute(model):
        identity = root_identity | {
            "model": model,
            "preflight_sha256": hashlib.sha256(
                json.dumps(rows[model], sort_keys=True).encode()
            ).hexdigest(),
        }
        free_remaining = (quota.get("free_model_daily_requests") or {}).get("remaining")
        if model == "ling-free" and frozen and not isinstance(free_remaining, int):
            raise ValueError("cota gratuita nao informada; nao iniciar bloco full")
        return run(
            rows[model],
            TARGETS[model],
            args.output / model,
            budgets[model],
            json.loads(json.dumps(identity)),
            transport,
            max_new_calls=free_remaining if model == "ling-free" else None,
        )

    errors = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(execute, model): model for model in FULL_MODELS}
        for future in as_completed(futures):
            model = futures[future]
            try:
                future.result()
                print("Geracao concluida:", model, flush=True)
            except Exception as exc:
                errors[model] = str(exc)
                print("Falha:", model, str(exc), flush=True)
    save(args.output / "generation-status.json", {"errors": errors, "complete": not errors})
    score_results(rows, records, cfg, args.prepared, args.output, errors, split)


def score_results(rows, records, cfg, prepared, output, errors, split):
    """Pontua artefatos persistidos; nenhuma chamada de geracao."""
    # Exporta tambem resultados parciais para inspecao, sem inferencia incompleta.
    documents = json.loads((prepared / "documents.json").read_text())
    by_id = {d["doc_id"]: d for d in documents}
    results = {m: {a.id: ArmResult(a, [], [], {}) for a in cfg.arms} for m in rows}
    ordered = []
    for model in rows:
        folder = output / model
        if not (folder / "ledger.json").exists():
            continue
        ledger = json.loads((folder / "ledger.json").read_text())
        accepted = {a["case"]: a for a in ledger["attempts"] if a["status"] == "accepted"}
        for i, row in enumerate(rows[model]):
            if i not in accepted:
                continue
            attempt = accepted[i]
            response = json.loads((folder / attempt["response_file"]).read_text())
            raw = response["choices"][0]["message"]["content"]
            text = clean_generation(raw)
            reference = clean_text(by_id[row["doc_id"]]["reference"])
            score = score_document(row["doc_id"], text, reference, by_id[row["doc_id"]]["source"])
            score.extra["meteor"] = meteor(text, reference)
            prediction = {
                "doc_id": row["doc_id"],
                "arm": row["arm"],
                "prediction": text,
                "reference": reference,
                "raw_output": raw,
                "context": row["context"],
                "source": by_id[row["doc_id"]]["source"],
                "example_ids": row["example_ids"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "truncated_output": response["choices"][0]["finish_reason"] == "length",
                "truncated_prompt": False,
                "usage": response["usage"],
                "provider": response["provider"],
                "model": response["model"],
                "elapsed_seconds": attempt.get("elapsed_seconds"),
            }
            result = results[model][row["arm"]]
            result.predictions.append(prediction)
            result.scores.append(score)
            ordered.append((score, prediction))
    print("Calculando BERTScore integral:", len(ordered), "pares", flush=True)
    texts = [p["prediction"] for _, p in ordered]
    refs = [p["reference"] for _, p in ordered]
    audits = bertscore_token_audit(texts + refs)
    components = bertscore_components(texts, refs, batch_size=1)
    for i, ((score, prediction), component) in enumerate(zip(ordered, components, strict=True)):
        score.bertscore = component.f1
        score.extra.update(
            bertscore_precision=component.precision, bertscore_recall=component.recall
        )
        prediction["metric_token_audit"] = {
            "model": BERTSCORE_MODEL,
            "prediction": audits[i],
            "reference": audits[len(ordered) + i],
        }
    report = {
        "split": split,
        "documents": len(records),
        "models": {},
        "hypotheses_confirmed": False,
        "analysis_plan": ANALYSIS_PLAN,
        "supplementary_metrics": SUPPLEMENTARY_METRICS,
        "scope": "dev_descriptive" if split == "dev" else "eval",
        "errors": errors,
    }
    for model, arms in results.items():
        folder = output / model / "results"
        folder.mkdir(exist_ok=True)
        save(folder / "analysis_plan.json", ANALYSIS_PLAN)
        for result in arms.values():
            result.summary = aggregate(result.scores)
            _write_arm(folder, result)
        scores = {a: {s.doc_id: s.flat() for s in result.scores} for a, result in arms.items()}
        complete = all(len(r.scores) == len(records) for r in arms.values())
        if complete and len(records) > 1:
            from dataclasses import asdict

            save(folder / "comparisons.json", [asdict(c) for c in compare_hypotheses(scores)])
        report["models"][model] = {
            "complete": complete,
            "arms": {a: r.summary for a, r in arms.items()},
        }
    save(output / "report.json", report)
    print("Resultados e metricas gravados em", output, flush=True)


if __name__ == "__main__":
    main()
