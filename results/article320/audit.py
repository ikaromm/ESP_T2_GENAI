"""Auditoria independente dos 320 primeiros documentos; não envia chamadas à API.

Execute da raiz do repositório: uv run --locked python results/article320/audit.py
As respostas brutas permanecem em outputs/; publica-se apenas o manifesto de hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from findsum_rag.data import clean_text
from findsum_rag.generate import clean_generation

ROOT = Path.cwd()
OUT = ROOT / "results/article320"
ARMS = ("C1", "C1t", "C2", "C3", "C4", "C5")
MODELS = {
    "ling-free": ("ling", "novita", "Novita"),
    "qwen37": ("qwen", "alibaba", "Alibaba"),
    "gemma26": ("gemma", "darkbloom", "Darkbloom"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = list(csv.DictReader((ROOT / "configs/full-eval-cohort.csv").open()))
    ids = [r["doc_id"] for r in cohort]
    require(len(ids) == len(set(ids)) == 1000, "coorte invalida")
    require(len({r["stock_name"] for r in cohort}) == 1000,
            "mais de um documento por empresa na coorte")
    manifest = json.loads((ROOT / "data/interim/splits-liquidity.json").read_text())
    sets = manifest["sets"]
    company_sets = [{r["stock_name"] for r in sets[s]} for s in ("examples", "dev", "eval")]
    require(all(not a & b for i, a in enumerate(company_sets) for b in company_sets[i + 1 :]),
            "empresa presente em mais de um split")
    require(set(ids) == {r["doc_id"] for r in sets["eval"]}, "coorte diferente do eval")
    summary = json.loads((ROOT / "results/progress/summary.json").read_text())
    require(summary["document_ids"][:320] == ids[:320],
            "prefixo atual diferente da coorte congelada")
    scores = list(csv.DictReader((OUT / "scores-320.csv").open()))
    score_map = {(r["model"], r["doc_id"], r["arm"]): r for r in scores}
    require(len(scores) == len(score_map) == 320 * 6 * 3, "scores faltantes/duplicados")
    current_scores = list(csv.DictReader((ROOT / "results/progress/scores.csv").open()))
    current_map = {(r["model"], r["doc_id"], r["arm"]): r for r in current_scores}
    require(all(current_map.get(key) == row for key, row in score_map.items()),
            "snapshot de scores diverge do painel atual")
    for model in MODELS:
        for doc in ids[:320]:
            for arm in ARMS:
                require((model, doc, arm) in score_map, f"score ausente: {model}/{doc}/{arm}")
    for r in scores:
        for metric in ("bertscore", "bertscore_precision", "bertscore_recall", "rougeL", "meteor"):
            x = float(r[metric])
            require(math.isfinite(x) and 0 <= x <= 1, f"score invalido: {metric}")
    require(len(summary["token_audit"]) == len(current_scores),
            "auditoria BERTScore atual incompleta")
    require(all(not part["truncated"] and part["full_tokens"] == part["evaluated_tokens"]
                for pair in summary["token_audit"] for part in pair), "texto BERTScore truncado")
    documents = {d["doc_id"]: d for d in json.loads(
        (ROOT / "outputs/full-ling-prepared/documents.json").read_text())}
    require(set(documents) == set(ids), "documentos preparados diferentes da coorte")
    cache = json.loads((ROOT / "outputs/progress-metrics/cache.json").read_text())
    require(cache["method"] == summary["method_id"], "versao das metricas/cache mudou")
    lineage_checks = 0

    raw_rows: list[dict] = []
    model_report: dict[str, dict] = {}
    document_hashes = set()
    for model, (slug, provider, response_provider) in MODELS.items():
        prepared = ROOT / f"outputs/full-{slug}-prepared"
        folder = ROOT / f"outputs/full-{slug}-batches" / model
        doc_hash = digest(prepared / "documents.json")
        document_hashes.add(doc_hash)
        frozen_hashes = json.loads((prepared / "sha256.json").read_text())
        require(doc_hash == frozen_hashes["documents.json"], f"documentos alterados: {model}")
        preflight = prepared / model / "preflight.json"
        require(digest(preflight) == frozen_hashes[f"{model}/preflight.json"],
                f"prompts alterados: {model}")
        rows = json.loads(preflight.read_text())
        require(len(rows) == 6000, f"preflight incompleto: {model}")
        example_ids = {r["doc_id"] for r in sets["examples"]}
        fixed = None
        max_tokens = 0
        for i in range(1000):
            group = rows[6 * i : 6 * i + 6]
            require([(r["doc_id"], r["arm"]) for r in group] ==
                    [(ids[i], arm) for arm in ARMS], f"ordem dos bracos: {model}/{i}")
            require(group[1]["context_tokens"] == group[2]["context_tokens"] and
                    group[1]["prompt_tokens"] == group[2]["prompt_tokens"],
                    f"C1t/C2 sem paridade: {model}/{i}")
            require(len({r["context"] for r in group[2:]}) == 1,
                    f"RAG diferente entre C2-C5: {model}/{i}")
            source = documents[ids[i]]["source"]
            require(group[0]["context"] == source and
                    source.startswith(group[1]["context"]),
                    f"fonte integral/prefixo invalido: {model}/{i}")
            require(not group[0]["example_ids"] and not group[1]["example_ids"]
                    and not group[2]["example_ids"], f"exemplo em zero-shot: {model}/{i}")
            for r in group[3:]:
                chosen = r["example_ids"]
                require(len(chosen) == len(set(chosen)) == 4 and set(chosen) <= example_ids
                        and ids[i] not in chosen, f"exemplo invalido: {model}/{i}/{r['arm']}")
            if fixed is None:
                fixed = group[3]["example_ids"]
            require(group[3]["example_ids"] == fixed, f"C3 variou: {model}/{i}")
            for r in group:
                require(0 < r["prompt_tokens"] <= 49152 and r["max_output_tokens"] == 8192,
                        f"janela invalida: {model}/{i}/{r['arm']}")
                max_tokens = max(max_tokens, r["prompt_tokens"])

        ledger_path = folder / "ledger.json"
        ledger = json.loads(ledger_path.read_text())
        require(ledger["identity"]["documents"] == ids,
                f"identidade da coorte no ledger: {model}")
        accepted = [a for a in ledger["attempts"] if a["status"] == "accepted" and a["case"] < 1920]
        require(len(accepted) == len({a["case"] for a in accepted}) == 1920,
                f"respostas aceitas faltantes/duplicadas: {model}")
        cost, finishes, models_seen = 0.0, Counter(), Counter()
        for a in accepted:
            index = a["case"]
            row = rows[index]
            require((a["doc_id"], a["arm"]) == (row["doc_id"], row["arm"]),
                    f"ledger/preflight diferentes: {model}/{index}")
            response_path = folder / a["response_file"]
            request_path = folder / a["response_file"].replace(".response.", ".request.")
            req = json.loads(request_path.read_text())
            resp = json.loads(response_path.read_text())
            expected_model = (
                "inclusionai/ling-3.0-flash-fin:free" if model == "ling-free" and index < 960
                else "inclusionai/ling-3.0-flash-fin" if model == "ling-free"
                else "qwen/qwen3.7-flash" if model == "qwen37"
                else "google/gemma-4-26b-a4b-it"
            )
            require(req["model"] == resp["model"] == expected_model,
                    f"modelo inesperado: {model}/{index}")
            require(req["messages"] == row["messages"], f"prompt salvo diferente: {model}/{index}")
            require(req["provider"]["only"] == [provider] and
                    req["provider"]["allow_fallbacks"] is False and
                    req["provider"]["require_parameters"] is True and
                    resp["provider"] == response_provider, f"provedor diferente: {model}/{index}")
            require(req["temperature"] == 0 and req["top_p"] == 1 and
                    req["reasoning"]["enabled"] is False and req["max_tokens"] == 8192,
                    f"parametros diferentes: {model}/{index}")
            require(resp["usage"]["prompt_tokens"] == row["prompt_tokens"] and
                    0 < resp["usage"]["completion_tokens"] <= 8192 and
                    resp["choices"][0]["finish_reason"] in ("stop", "length") and
                    resp["choices"][0]["message"]["content"].strip(),
                    f"resposta/token invalido: {model}/{index}")
            if model == "ling-free" and index >= 960:
                require(a.get("amendment_sha256") ==
                        digest(ROOT / "configs/ling-paid-round-161-320.json"),
                        f"adendo Ling ausente: {index}")
            require(score_map[(model, row["doc_id"], row["arm"])]["finish_reason"] ==
                    resp["choices"][0]["finish_reason"],
                    f"finish_reason score/resposta: {model}/{index}")
            pair = dict(
                model=model, arm=row["arm"], doc_id=row["doc_id"],
                prediction=clean_generation(resp["choices"][0]["message"]["content"]),
                reference=clean_text(documents[row["doc_id"]]["reference"]),
                finish_reason=resp["choices"][0]["finish_reason"],
            )
            key = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
            require(key in cache["scores"], f"resposta/referencia fora do cache: {model}/{index}")
            cached_row = cache["scores"][key]["row"]
            score_row = score_map[(model, row["doc_id"], row["arm"])]
            require(set(cached_row) == set(score_row) and all(
                str(cached_row[k]) == score_row[k] for k in score_row),
                f"score CSV diverge da resposta/cache: {model}/{index}")
            lineage_checks += 1
            finishes[resp["choices"][0]["finish_reason"]] += 1
            models_seen[req["model"]] += 1
            cost += float(resp["usage"]["cost"])
            raw_rows.append({
                "model": model, "case": index, "doc_id": a["doc_id"], "arm": a["arm"],
                "request_path": str(request_path.relative_to(ROOT)),
                "request_sha256": digest(request_path),
                "response_path": str(response_path.relative_to(ROOT)),
                "response_sha256": digest(response_path),
            })
        accepted_digest = hashlib.sha256(json.dumps(accepted, sort_keys=True).encode()).hexdigest()
        model_report[model] = {
            "prepared_documents_sha256": doc_hash,
            "preflight_sha256": digest(preflight),
            "first_320_accepted_ledger_sha256": accepted_digest,
            "accepted": len(accepted), "finish_reasons": dict(finishes),
            "request_models": dict(models_seen), "reported_cost_usd": round(cost, 9),
            "maximum_preflight_prompt_tokens": max_tokens,
        }
        print(model, model_report[model], flush=True)
        del rows
    require(len(document_hashes) == 1, "fontes/referencias diferentes entre modelos")
    if summary["documents_compared"] == 320:
        for model in MODELS:
            for arm in ARMS:
                for metric in (
                    "bertscore", "bertscore_precision", "bertscore_recall", "rougeL", "meteor"
                ):
                    vals = [float(r[metric]) for r in scores
                            if r["model"] == model and r["arm"] == arm]
                    require(len(vals) == 320 and math.isclose(sum(vals) / 320,
                            summary["models"][model][arm][metric], abs_tol=1e-12),
                            f"agregado incorreto: {model}/{arm}/{metric}")
    report = {
        "scope": "first_320_eval_documents",
        "documents": 320, "accepted_responses": len(raw_rows),
        "scores": len(scores), "scores_snapshot_sha256": digest(OUT / "scores-320.csv"),
        "lineage_checks": lineage_checks,
        "truncated_bert_texts": 0,
        "company_split_overlap": 0, "model_checks": model_report,
        "raw_manifest_sha256": None,
        "limitations": [
            "Raw requests/responses and prepared artifacts are local in outputs and not in Git.",
            "BERTScore is not recomputed here; a full metric rerun is available via "
            "--metrics-only.",
            "Embeddings and FAISS retrieval are not recomputed here; frozen prepared "
            "artifacts, prompts, and code tests are checked.",
            "Factual correctness and numeric preservation are not evaluated by the current scores.",
        ],
    }
    manifest_path = OUT / "raw-artifact-manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(raw_rows, key=lambda r: (r["model"], r["case"])))
    report["raw_manifest_sha256"] = digest(manifest_path)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("audit complete", report["accepted_responses"], report["scores"], flush=True)


if __name__ == "__main__":
    main()
