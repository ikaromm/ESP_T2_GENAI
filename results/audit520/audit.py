"""Auditoria independente dos 520 primeiros documentos; não envia chamadas à API.

Execute da raiz do repositório: uv run --locked python results/audit520/audit.py
Confere coorte, prompts congelados, pedidos e respostas brutos (inclusive retries da
rodada 321-520), adendos do Ling pago, cache, scores e auditoria de tokens. As
respostas brutas permanecem em outputs/; publica-se apenas o manifesto de hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

from findsum_rag.data import clean_text
from findsum_rag.generate import clean_generation

ROOT = Path.cwd()
OUT = ROOT / "results/audit520"
N_DOCS = 520
ROUND_FIRST, ROUND_LAST = 1920, 3120
ARMS = ("C1", "C1t", "C2", "C3", "C4", "C5")
METRICS = ("bertscore", "bertscore_precision", "bertscore_recall", "rougeL", "meteor")
ROUNDS = {"1-100": (0, 600), "101-160": (600, 960), "161-320": (960, 1920),
          "321-520": (1920, 3120)}
FREE_LING = "inclusionai/ling-3.0-flash-fin:free"
PAID_LING = "inclusionai/ling-3.0-flash-fin"
AMENDMENTS = {
    "configs/ling-paid-round-161-320.json": (960, 1920),
    "configs/ling-paid-round-321-520.json": (1920, 3120),
}
MODELS = {
    # modelo: (pasta, provedor pedido, provedor na resposta, max_price entrada/saída)
    "ling-free": ("ling", "novita", "Novita", None),
    "qwen37": ("qwen", "alibaba", "Alibaba", ("0.1", "0.4")),
    "gemma26": ("gemma", "darkbloom", "Darkbloom", ("0.042", "0.22")),
}
LING_PRICES = {FREE_LING: ("0", "0"), PAID_LING: ("0.042", "0.1232")}
MODEL_NAMES = {"qwen37": "qwen/qwen3.7-flash", "gemma26": "google/gemma-4-26b-a4b-it"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def round_of(case: int) -> str:
    return next(name for name, (a, b) in ROUNDS.items() if a <= case < b)


def expected_request_model(model: str, case: int) -> str:
    if model == "ling-free":
        return FREE_LING if case < 960 else PAID_LING
    return MODEL_NAMES[model]


def check_request(req: dict, row: dict, model: str, case: int, provider: str, prices) -> None:
    name = expected_request_model(model, case)
    prices = LING_PRICES[name] if model == "ling-free" else prices
    require(req["model"] == name, f"modelo pedido: {model}/{case}")
    require(req["messages"] == row["messages"], f"prompt salvo diferente: {model}/{case}")
    p = req["provider"]
    require(p["only"] == [provider] and p["allow_fallbacks"] is False
            and p["require_parameters"] is True, f"provedor pedido: {model}/{case}")
    require(p["max_price"] == {"prompt": float(prices[0]), "completion": float(prices[1])},
            f"max_price: {model}/{case}")
    require(req["temperature"] == 0 and req["top_p"] == 1 and req["stream"] is False
            and req["reasoning"] == {"enabled": False} and req["max_tokens"] == 8192,
            f"parametros: {model}/{case}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = list(csv.DictReader((ROOT / "configs/full-eval-cohort.csv").open()))
    ids = [r["doc_id"] for r in cohort]
    require(len(ids) == len(set(ids)) == 1000, "coorte invalida")
    require(len({r["stock_name"] for r in cohort}) == 1000, "empresa repetida na coorte")
    lock = json.loads((ROOT / "configs/full_openrouter.lock.json").read_text())
    snapshot_lock = json.loads((ROOT / "configs/lock-history" /
                                "97ee50ac6e006be31e36420e7622de8d77e8ef6cd43275a2cc8e1afe16ddd8a1.json"
                                ).read_text())
    require([r["doc_id"] for r in lock["cohort"]] == ids, "coorte do lock difere do CSV")
    require(lock["cohort"] == snapshot_lock["cohort"]
            and lock["analysis_plan"] == snapshot_lock["analysis_plan"]
            and lock["dataset_files"] == snapshot_lock["dataset_files"],
            "lock atual alterou a coorte, o plano ou os dados do snapshot")
    manifest = json.loads((ROOT / "data/interim/splits-liquidity.json").read_text())
    sets = manifest["sets"]
    companies = [{r["stock_name"] for r in sets[s]} for s in ("examples", "dev", "eval")]
    require(all(not a & b for i, a in enumerate(companies) for b in companies[i + 1:]),
            "empresa presente em mais de um split")
    require(set(ids) == {r["doc_id"] for r in sets["eval"]}, "coorte diferente do eval")

    amendment_sha = {path: digest(ROOT / path) for path in AMENDMENTS}
    new_amendment = json.loads((ROOT / "configs/ling-paid-round-321-520.json").read_text())
    history = json.loads((ROOT / "outputs/full-rounds/history" /
                          f"{new_amendment['plan_id']}.json").read_text())
    require(history["plan"]["documents"] == ids[320:520] and history["result"]["complete"]
            and history["plan"]["plan_id"] == new_amendment["plan_id"],
            "plano 321-520 ausente, incompleto ou diferente do adendo")
    require(new_amendment["previous_amendments_sha256"]
            == [amendment_sha["configs/ling-paid-round-161-320.json"]],
            "adendo 321-520 nao referencia o adendo anterior")

    summary = json.loads((OUT / "summary-520.json").read_text())
    require(summary["documents_compared"] == N_DOCS and summary["document_ids"] == ids[:N_DOCS],
            "snapshot nao cobre exatamente os 520 primeiros documentos")
    require(summary["hypotheses_tested"] is False, "painel declara hipoteses testadas")
    scores = list(csv.DictReader((OUT / "scores-520.csv").open()))
    score_map = {(r["model"], r["doc_id"], r["arm"]): r for r in scores}
    expected_keys = {(m, d, a) for m in MODELS for d in ids[:N_DOCS] for a in ARMS}
    require(len(scores) == len(score_map) == N_DOCS * 18 and set(score_map) == expected_keys,
            "scores faltantes, extras ou duplicados")
    current_summary = json.loads((ROOT / "results/progress/summary.json").read_text())
    require(current_summary["documents_compared"] >= N_DOCS
            and current_summary["document_ids"][:N_DOCS] == ids[:N_DOCS],
            "prefixo atual do painel difere do snapshot")
    current_scores = list(csv.DictReader((ROOT / "results/progress/scores.csv").open()))
    current_map = {(r["model"], r["doc_id"], r["arm"]): r for r in current_scores}
    require(len(current_map) == len(current_scores)
            and all(current_map.get(key) == row for key, row in score_map.items()),
            "scores atuais alteraram o snapshot dos 520 primeiros documentos")
    for r in scores:
        for metric in METRICS:
            x = float(r[metric])
            require(math.isfinite(x) and 0 <= x <= 1, f"score invalido: {metric}")
    token_audit = summary["token_audit"]
    require(len(token_audit) == len(scores), "auditoria BERTScore incompleta")
    require(all(not part["truncated"] and part["full_tokens"] == part["evaluated_tokens"]
                for pair in token_audit for part in pair), "texto BERTScore truncado")
    position = {(r["model"], r["doc_id"], r["arm"]): i for i, r in enumerate(scores)}

    frozen_scores = list(csv.DictReader((ROOT / "results/article320/scores-320.csv").open()))
    require(len(frozen_scores) == 5760 and all(
        score_map.get((r["model"], r["doc_id"], r["arm"])) == r for r in frozen_scores),
        "scores dos 320 primeiros documentos mudaram")
    frozen_raw = {(r["model"], int(r["case"])): r for r in csv.DictReader(
        (ROOT / "results/article320/raw-artifact-manifest.csv").open())}
    require(len(frozen_raw) == 5760, "manifesto article320 incompleto")

    documents = {d["doc_id"]: d for d in json.loads(
        (ROOT / "outputs/full-ling-prepared/documents.json").read_text())}
    require(set(documents) == set(ids), "documentos preparados diferentes da coorte")
    cache = json.loads((ROOT / "outputs/progress-metrics/cache.json").read_text())
    require(cache["method"] == summary["method_id"], "cache de outro metodo")
    for name, sha in summary["method"].items():
        require(digest(ROOT / name) == sha, f"arquivo do metodo alterado: {name}")

    raw_rows, model_report, round_report = [], {}, {}
    document_hashes, lineage = set(), 0
    for model, (slug, provider, response_provider, prices) in MODELS.items():
        prepared = ROOT / f"outputs/full-{slug}-prepared"
        folder = ROOT / f"outputs/full-{slug}-batches" / model
        frozen_hashes = json.loads((prepared / "sha256.json").read_text())
        doc_hash = digest(prepared / "documents.json")
        document_hashes.add(doc_hash)
        require(doc_hash == frozen_hashes["documents.json"], f"documentos alterados: {model}")
        preflight = prepared / model / "preflight.json"
        require(digest(preflight) == frozen_hashes[f"{model}/preflight.json"],
                f"prompts alterados: {model}")
        rows = json.loads(preflight.read_text())
        require(len(rows) == 6000, f"preflight incompleto: {model}")
        example_ids = {r["doc_id"] for r in sets["examples"]}
        fixed, max_tokens = rows[3]["example_ids"], 0
        for i in range(N_DOCS):
            group = rows[6 * i: 6 * i + 6]
            require([(r["doc_id"], r["arm"]) for r in group] == [(ids[i], a) for a in ARMS],
                    f"ordem dos bracos: {model}/{i}")
            require(group[1]["context_tokens"] == group[2]["context_tokens"]
                    and group[1]["prompt_tokens"] == group[2]["prompt_tokens"],
                    f"C1t/C2 sem paridade: {model}/{i}")
            require(len({r["context"] for r in group[2:]}) == 1, f"RAG C2-C5: {model}/{i}")
            source = documents[ids[i]]["source"]
            require(group[0]["context"] == source and source.startswith(group[1]["context"]),
                    f"fonte integral/prefixo invalido: {model}/{i}")
            require(not any(r["example_ids"] for r in group[:3]), f"zero-shot: {model}/{i}")
            for r in group[3:]:
                chosen = r["example_ids"]
                require(len(chosen) == len(set(chosen)) == 4 and set(chosen) <= example_ids
                        and ids[i] not in chosen, f"exemplo invalido: {model}/{i}/{r['arm']}")
            require(group[3]["example_ids"] == fixed, f"C3 variou: {model}/{i}")
            for r in group:
                require(0 < r["prompt_tokens"] <= 49152 and r["max_output_tokens"] == 8192
                        and r["allow_length"] is True, f"janela: {model}/{i}/{r['arm']}")
                max_tokens = max(max_tokens, r["prompt_tokens"])

        ledger = json.loads((folder / "ledger.json").read_text())
        require(ledger["identity"]["documents"] == ids, f"identidade do ledger: {model}")
        before = json.loads((ROOT / "outputs/round-321-520-20260929/ledger-before" /
                             f"{model}.json").read_text())
        prefix_ok = (ledger["identity"] == before["identity"] and
                     ledger["attempts"][: len(before["attempts"])] == before["attempts"])
        require(prefix_ok, f"historico do ledger anterior alterado: {model}")
        require(all(a["case"] < ROUND_FIRST for a in before["attempts"]),
                f"backup contem casos da rodada: {model}")
        prefix_attempts = [a for a in ledger["attempts"] if a["case"] < ROUND_LAST]
        by_case: dict[int, list] = {}
        for a in prefix_attempts:
            require((a["doc_id"], a["arm"]) == (rows[a["case"]]["doc_id"], rows[a["case"]]["arm"]),
                    f"ledger/preflight diferentes: {model}/{a['case']}")
            by_case.setdefault(a["case"], []).append(a)
        require(sorted(by_case) == list(range(ROUND_LAST)), f"casos faltantes: {model}")
        # Rodada nova: todas as tentativas, inclusive retries, conferidas.
        statuses, retry_kinds, amendments = Counter(), Counter(), Counter()
        for case in range(ROUND_FIRST, ROUND_LAST):
            attempts = by_case[case]
            require([a["status"] for a in attempts].count("accepted") == 1
                    and attempts[-1]["status"] == "accepted",
                    f"caso sem exatamente uma aceita final: {model}/{case}")
            for n, a in enumerate(attempts):
                statuses[a["status"]] += 1
                require(a["response_file"] == f"{case:04d}-{n}.response.json",
                        f"arquivo de tentativa: {model}/{case}/{n}")
                if a["status"] != "accepted":
                    require(a["status"] in {"http429", "http_retryable"},
                            f"tentativa nao transitoria: {model}/{case}")
                    retry_kinds[a.get("retry_classification") or
                                f"http_{a.get('error', {}).get('http_status')}"] += 1
                req = json.loads((folder / a["response_file"].replace(
                    ".response.", ".request.")).read_text())
                check_request(req, rows[case], model, case, provider, prices)
                if model == "ling-free":
                    amendments[a.get("amendment_sha256")] += 1
                    require(a.get("amendment_sha256") ==
                            amendment_sha["configs/ling-paid-round-321-520.json"],
                            f"tentativa Ling paga sem hash do adendo: {case}")
                else:
                    require("amendment_sha256" not in a, f"adendo fora do Ling: {model}")

        accepted = [a for a in prefix_attempts if a["status"] == "accepted"]
        require(len(accepted) == len({a["case"] for a in accepted}) == ROUND_LAST,
                f"respostas aceitas faltantes/duplicadas: {model}")
        cost = {name: Decimal(0) for name in ROUNDS}
        finishes = {name: Counter() for name in ROUNDS}
        request_models, reserved = Counter(), Decimal(0)
        for a in accepted:
            case, row = a["case"], rows[a["case"]]
            request_path = folder / a["response_file"].replace(".response.", ".request.")
            response_path = folder / a["response_file"]
            req = json.loads(request_path.read_text())
            resp = json.loads(response_path.read_text())
            check_request(req, row, model, case, provider, prices)
            usage, choice = resp["usage"], resp["choices"][0]
            require(resp["model"] == req["model"] and resp["provider"] == response_provider,
                    f"modelo/provedor da resposta: {model}/{case}")
            require(usage["prompt_tokens"] == row["prompt_tokens"]
                    and 0 < usage["completion_tokens"] <= 8192
                    and not (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                    and choice["finish_reason"] in ("stop", "length")
                    and choice["message"]["content"].strip(),
                    f"resposta/tokens invalidos: {model}/{case}")
            value = Decimal(str(usage["cost"]))
            paid = req["model"] != FREE_LING
            require(value > 0 if paid else value == 0, f"custo inconsistente: {model}/{case}")
            require(Decimal(str(a["reported_cost_usd"])) == value, f"custo do ledger: {model}")
            if model == "ling-free" and 960 <= case < ROUND_LAST:
                span = next(p for p, (x, y) in AMENDMENTS.items() if x <= case < y)
                require(a.get("amendment_sha256") == amendment_sha[span],
                        f"adendo Ling ausente/errado: {case}")
            key_tuple = (model, row["doc_id"], row["arm"])
            require(score_map[key_tuple]["finish_reason"] == choice["finish_reason"],
                    f"finish_reason score/resposta: {model}/{case}")
            pair = dict(model=model, arm=row["arm"], doc_id=row["doc_id"],
                        prediction=clean_generation(choice["message"]["content"]),
                        reference=clean_text(documents[row["doc_id"]]["reference"]),
                        finish_reason=choice["finish_reason"])
            key = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
            require(key in cache["scores"], f"resposta fora do cache: {model}/{case}")
            cached = cache["scores"][key]
            require(set(cached["row"]) == set(score_map[key_tuple]) and all(
                str(cached["row"][k]) == score_map[key_tuple][k] for k in score_map[key_tuple]),
                f"score CSV diverge do cache: {model}/{case}")
            require(cached["token_audit"] == token_audit[position[key_tuple]],
                    f"auditoria de tokens fora de ordem: {model}/{case}")
            lineage += 1
            name = round_of(case)
            cost[name] += value
            finishes[name][choice["finish_reason"]] += 1
            request_models[req["model"]] += 1
            raw = {
                "model": model, "case": case, "doc_id": a["doc_id"], "arm": a["arm"],
                "request_path": str(request_path.relative_to(ROOT)),
                "request_sha256": digest(request_path),
                "response_path": str(response_path.relative_to(ROOT)),
                "response_sha256": digest(response_path),
            }
            if case < ROUND_FIRST:
                old = frozen_raw[(model, case)]
                require(all(str(raw[k]) == old[k] for k in old),
                        f"artefato bruto anterior alterado: {model}/{case}")
            raw_rows.append(raw)
        round_attempts = [a for c in range(ROUND_FIRST, ROUND_LAST) for a in by_case[c]]
        for a in round_attempts:
            reserved += Decimal(str(a["reported_cost_usd"] if a["status"] == "accepted"
                                    else a["reserved_usd"]))
        round_report[model] = {
            "accepted": ROUND_LAST - ROUND_FIRST,
            "attempts": len(round_attempts),
            "attempt_statuses": dict(statuses),
            "retries_by_kind": dict(retry_kinds),
            "max_attempts_per_case": max(len(by_case[c]) for c in range(ROUND_FIRST, ROUND_LAST)),
            "request_model": expected_request_model(model, ROUND_FIRST),
            "request_provider": provider,
            "response_provider": response_provider,
            "reported_cost_usd": str(cost["321-520"]),
            "held_cost_usd": str(reserved),
            "round_cap_usd": new_amendment["round_budget_usd"][model],
            "finish_reasons": dict(finishes["321-520"]),
            "amendment_sha256_counts": dict(amendments) if model == "ling-free" else None,
        }
        model_report[model] = {
            "prepared_documents_sha256": doc_hash,
            "preflight_sha256": frozen_hashes[f"{model}/preflight.json"],
            "ledger_first_520_sha256": hashlib.sha256(
                json.dumps(prefix_attempts, sort_keys=True).encode()).hexdigest(),
            "ledger_history_before_round_preserved": prefix_ok,
            "accepted_first_520": len(accepted),
            "request_models": dict(request_models),
            "reported_cost_usd_by_round": {k: str(v) for k, v in cost.items()},
            "finish_reasons_by_round": {k: dict(v) for k, v in finishes.items()},
            "length_first_520": sum(v["length"] for v in finishes.values()),
            "maximum_preflight_prompt_tokens_first_520": max_tokens,
        }
        print(model, round_report[model], flush=True)
        del rows
    require(len(document_hashes) == 1, "fontes/referencias diferentes entre modelos")
    for model in MODELS:
        for arm in ARMS:
            for metric in METRICS:
                vals = [float(r[metric]) for r in scores if r["model"] == model and r["arm"] == arm]
                require(len(vals) == N_DOCS and math.isclose(
                    sum(vals) / N_DOCS, summary["models"][model][arm][metric], abs_tol=1e-12),
                    f"agregado incorreto: {model}/{arm}/{metric}")
        progress = summary["progress"][model]
        require(progress["accepted"] == ROUND_LAST and progress["complete_documents"] == N_DOCS
                and progress["length"] == model_report[model]["length_first_520"],
                f"cartao de progresso: {model}")
    total = sum(Decimal(r["reported_cost_usd"]) for r in round_report.values())
    report = {
        "scope": "first_520_eval_documents",
        "documents": N_DOCS,
        "accepted_responses": len(raw_rows),
        "scores": len(scores),
        "unique_score_keys": len(score_map),
        "lineage_checks": lineage,
        "truncated_bert_texts": 0,
        "company_split_overlap": 0,
        "first_320_scores_unchanged": True,
        "first_320_raw_artifacts_unchanged": True,
        "hypotheses_tested": False,
        "lock_id": snapshot_lock["lock_id"],
        "metrics_method_id": summary["method_id"],
        "scores_csv_sha256": digest(OUT / "scores-520.csv"),
        "summary_snapshot_sha256": digest(OUT / "summary-520.json"),
        "amendments_sha256": amendment_sha,
        "round_321_520": {"plan_id": new_amendment["plan_id"], "documents": 200,
                          "reported_cost_total_usd": str(total), "models": round_report},
        "model_checks": model_report,
        "raw_manifest_sha256": None,
        "limitations": [
            "Raw requests/responses and prepared artifacts are local in outputs and not in Git.",
            "BERTScore is not recomputed here; the cache, CSV and token audit are cross-checked.",
            "Embeddings and FAISS retrieval are not recomputed; frozen prompts are checked.",
            "Factual correctness and numeric preservation are not evaluated by these scores.",
            "Ling used the free endpoint for documents 1-160 and the paid endpoint of the same "
            "model and provider for 161-520; documents also differ between rounds.",
        ],
    }
    manifest_path = OUT / "raw-artifact-manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(raw_rows, key=lambda r: (r["model"], r["case"])))
    report["raw_manifest_sha256"] = digest(manifest_path)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("audit complete", report["accepted_responses"], report["scores"], str(total), flush=True)


if __name__ == "__main__":
    main()
