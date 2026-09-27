"""Pagina offline para comparar os casos dev, entradas, prompts e referencias."""

import argparse
import hashlib
import json
from pathlib import Path

from findsum_rag.data import Task, clean_text
from findsum_rag.prompts import TASK_INSTRUCTIONS
from scripts.common.full_common import FULL_MODELS


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    prepared = root / "prepared"
    run = root / "run"
    metrics = root / "metrics-extended"
    if not (metrics / "report.json").exists():
        metrics = run
    documents = read(prepared / "documents.json")
    for d in documents:
        d["raw_reference"] = d["reference"]
        d["reference"] = clean_text(d["reference"])
    selected = {i for d in documents for ids in d["example_ids"].values() for i in ids}
    models = {}
    for model in FULL_MODELS:
        folder = root / "qwen-prepared" if model == "qwen37" else prepared
        rows = read(folder / model / "preflight.json")
        attempts = read(run / model / "ledger.json")["attempts"]
        predictions = []
        scores = []
        for p in sorted((run / model / "results").glob("*/predictions.jsonl")):
            predictions.extend(json.loads(line) for line in p.read_text().splitlines())
        import csv

        for p in sorted((metrics / model / "results").glob("*/scores.csv")):
            with p.open() as handle:
                scores.extend(dict(r, arm=p.parent.name) for r in csv.DictReader(handle))
        for attempt in attempts:
            if attempt["status"] == "accepted":
                req = read(
                    run / model / attempt["response_file"].replace(".response.", ".request.")
                )
                assert req["messages"] == rows[attempt["case"]]["messages"]
        models[model] = {
            "rows": rows,
            "predictions": predictions,
            "scores": scores,
            "attempts": attempts,
            "cost_usd": sum(a.get("reported_cost_usd") or 0 for a in attempts),
        }
    data = {
        "documents": documents,
        "examples": [e for e in read(prepared / "examples.json") if e["doc_id"] in selected],
        "selection": read(prepared / "selection.json"),
        "report": read(metrics / "report.json"),
        "models": models,
        "calibration": read(root / "qwen-prepared" / "report.json"),
        "eligibility": read(run / "eligibility.json"),
        "validation": read(root / "validation.json"),
        "retrieval_query": TASK_INSTRUCTIONS[Task.LIQUIDITY],
        "artifact_root": str(root),
    }
    if metrics != run:
        data["supplementary_validation"] = read(metrics / "validation.json")
    data["total_cost_usd"] = (
        sum(m["cost_usd"] for m in models.values())
        + data["calibration"]["calibration_reported_cost_usd"]
    )
    payload = (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    template = Path("scripts/reports/templates/dev-review.html").read_text()
    output = Path("outputs/reports/dev10.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REVIEW_DATA__", payload))
    (root / "review-data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(
        output,
        output.stat().st_size,
        "bytes",
        "SHA256",
        hashlib.sha256(output.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
