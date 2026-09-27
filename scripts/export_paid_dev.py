"""Materializa uma rodada paga concluida para metricas locais de similaridade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findsum_rag.config import ExperimentConfig
from findsum_rag.data import clean_text
from findsum_rag.generate import clean_generation
from findsum_rag.metrics import RougeScorer, aggregate, score_document
from findsum_rag.pipeline import ArmResult, _write_arm
from run_prepared_paid import TARGETS, load_prepared, validate_response
from screen_openrouter import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=TARGETS, required=True)
    args = parser.parse_args()
    rows = load_prepared(args.prepared, args.model, 50)
    ledger = json.loads((args.run / "ledger.json").read_text())
    accepted = {a["case"]: a for a in ledger["attempts"] if a["status"] == "accepted"}
    if set(accepted) != set(range(len(rows))):
        raise ValueError("rodada paga incompleta; nao apresentar comparacao completa")
    cfg = ExperimentConfig.from_yaml(args.prepared / "config.yaml")
    common = {d["doc_id"]: d for d in json.loads((args.prepared / "documents.json").read_text())}
    refs = {
        d["doc_id"]: d for d in json.loads((args.prepared / "selection.json").read_text())["dev"]
    }
    results = {arm.id: ArmResult(arm, [], [], {}) for arm in cfg.arms}
    rouge = RougeScorer()
    for index, row in enumerate(rows):
        response = json.loads((args.run / accepted[index]["response_file"]).read_text())
        validate_response(response, TARGETS[args.model], row)
        text = clean_generation(response["choices"][0]["message"]["content"])
        doc, ref = common[row["doc_id"]], refs[row["doc_id"]]
        prediction = {
            "arm": row["arm"],
            "doc_id": row["doc_id"],
            "stock_name": ref["stock_name"],
            "report_id": ref["report_id"],
            "prediction": text,
            "reference": clean_text(doc["reference"]),
            "example_ids": row["example_ids"],
            "context": row["context"],
            "source": doc["source"],
            "context_words": len(row["context"].split()),
            "context_tokens": row["context_tokens"],
            "context_count_method": row.get("context_count_method", "local_native_tokenizer"),
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "truncated_prompt": False,
            "generation_metadata": {
                "id": response["id"],
                "model": response["model"],
                "provider": response["provider"],
                "usage": response["usage"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "reasoning_enabled": False,
                "seed_sent": False,
                "prompt_count_method": (
                    "api_probe" if args.model == "qwen37" else "local_tokenizer"
                ),
            },
        }
        result = results[row["arm"]]
        score = score_document(
            row["doc_id"], text, prediction["reference"], doc["source"], rouge=rouge
        )
        result.scores.append(score)
        result.predictions.append(prediction)
    destination = args.run / "results"
    destination.mkdir(exist_ok=False)
    for result in results.values():
        result.summary = aggregate(result.scores)
        _write_arm(destination, result)
    save(
        destination / "summary.json",
        {
            "scope": "dev_only",
            "status": "complete",
            "documents": 50,
            "generations": len(rows),
            "attempts": len(ledger["attempts"]),
            "reported_cost_usd": sum(a.get("reported_cost_usd", 0) for a in ledger["attempts"]),
            "arms": {key: r.summary for key, r in results.items()},
            "evaluation_scope": "reference_similarity_only",
            "bertscore_computed": False,
            "hypotheses_confirmed": False,
            "final_eval_executed": False,
        },
    )
    print(destination)


if __name__ == "__main__":
    main()
