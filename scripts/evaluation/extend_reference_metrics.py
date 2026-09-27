"""Adiciona metricas descritivas a respostas salvas, sem importar cliente de API."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

from findsum_rag.metrics import (
    SUPPLEMENTARY_METRICS,
    bertscore_components,
    bertscore_token_audit,
    ensure_meteor_resources,
    meteor,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    out = root / "metrics-extended"
    if out.exists():
        raise ValueError("Use uma pasta nova; resultados anteriores sao preservados")
    ensure_meteor_resources()
    report = json.loads((root / "run/report.json").read_text())
    pending = []
    hashes = {}
    for model, info in report["models"].items():
        for arm in info["arms"]:
            folder = root / "run" / model / "results" / arm
            predictions = [
                json.loads(x) for x in (folder / "predictions.jsonl").read_text().splitlines()
            ]
            with (folder / "scores.csv").open() as f:
                scores = list(csv.DictReader(f))
            by_id = {s["doc_id"]: s for s in scores}
            assert len(predictions) == len(by_id) == report["documents"]
            assert {p["doc_id"] for p in predictions} == set(by_id)
            for p in predictions:
                pending.append((model, arm, p, by_id[p["doc_id"]]))
            for name in ["predictions.jsonl", "scores.csv"]:
                path = folder / name
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    texts = [p["prediction"] for _, _, p, _ in pending]
    refs = [p["reference"] for _, _, p, _ in pending]
    audits = bertscore_token_audit(texts + refs)
    components = bertscore_components(texts, refs, batch_size=1)
    grouped = {}
    max_f1_diff = 0.0
    flat = []
    for (model, arm, p, old), c in zip(pending, components, strict=True):
        diff = abs(float(old["bertscore"]) - c.f1)
        max_f1_diff = max(diff, max_f1_diff)
        if diff > 1e-6:
            raise ValueError("F1 recalculado diverge; investigar antes de exportar")
        row = {k: float(v) if k != "doc_id" and v != "" else v for k, v in old.items()}
        row.update(
            bertscore_precision=c.precision,
            bertscore_recall=c.recall,
            meteor=meteor(p["prediction"], p["reference"]),
        )
        grouped.setdefault((model, arm), []).append(row)
        flat.append(dict(model=model, arm=arm, **row))
    out.mkdir()
    for (model, arm), rows in grouped.items():
        folder = out / model / "results" / arm
        folder.mkdir(parents=True)
        with (folder / "scores.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"n_docs": len(rows)}
        for key in rows[0]:
            values = [r[key] for r in rows if isinstance(r[key], (int, float))]
            if values:
                summary[key + "_mean"] = statistics.mean(values)
                summary[key + "_median"] = statistics.median(values)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
        report["models"][model]["arms"][arm] = summary
    report["supplementary_metrics"] = SUPPLEMENTARY_METRICS
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with (out / "scores.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    validation = {
        "pairs": len(pending),
        "source_sha256": hashes,
        "max_original_f1_difference": max_f1_diff,
        "metric_token_audit": audits,
        "method": SUPPLEMENTARY_METRICS,
        "versions": {
            n: importlib.metadata.version(n) for n in ["nltk", "bert-score", "transformers"]
        },
        "api_calls": 0,
        "new_api_cost_usd": 0,
    }
    from nltk.corpus import wordnet

    validation["wordnet_version"] = wordnet.get_version()
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    print("Pares:", len(pending), "maior diferenca F1:", max_f1_diff, "destino:", out)
    for model, v in report["models"].items():
        print(model)
        for arm, s in v["arms"].items():
            print(
                arm,
                *(
                    f"{s[k + '_mean']:.4f}"
                    for k in [
                        "bertscore_precision",
                        "bertscore_recall",
                        "bertscore",
                        "meteor",
                        "rougeL",
                    ]
                ),
            )


if __name__ == "__main__":
    main()
