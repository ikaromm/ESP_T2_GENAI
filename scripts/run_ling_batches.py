"""Um lote independente de 100 por modelo, da coorte final congelada de 1000."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from findsum_rag.config import ExperimentConfig
from findsum_rag.full_lock import verify_full_lock
from findsum_rag.metrics import ensure_meteor_resources
from findsum_rag.openrouter import OpenRouterFreeClient
from findsum_rag.progress import activity, log
from full_common import ARMS, Limits, selected_records
from run_experiment_openrouter import load_key, score_results
from run_prepared_paid import TARGETS, load_prepared, money, request_payload, run, validate_response
from screen_openrouter import save


def batch_indices(records, rows):
    ids = [r["doc_id"] for r in records]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("exige a coorte integral de 1000")
    expected = [(doc, arm) for doc in ids for arm in ARMS]
    if [(r["doc_id"], r["arm"]) for r in rows] != expected:
        raise ValueError("ordem/bracos diferem da coorte congelada")
    return [list(range(i * 600, (i + 1) * 600)) for i in range(10)]


def accepted_cases(folder, rows, identity, model="ling-free"):
    path = folder / "ledger.json"
    if not path.exists():
        return set()
    ledger = json.loads(path.read_text())
    if ledger["identity"] != identity:
        raise ValueError("identidade da rodada alterada")
    latest = {}
    for entry in ledger["attempts"]:
        index = entry["case"]
        if not isinstance(index, int) or not 0 <= index < len(rows):
            raise ValueError("indice invalido no ledger")
        if (entry["doc_id"], entry["arm"]) != (rows[index]["doc_id"], rows[index]["arm"]):
            raise ValueError("caso do ledger diverge")
        latest[index] = entry
    accepted = set()
    for index, entry in latest.items():
        if entry["status"] != "accepted":
            continue
        response_path = folder / entry["response_file"]
        request_path = response_path.with_name(
            response_path.name.replace(".response.", ".request.")
        )
        if json.loads(request_path.read_text()) != request_payload(TARGETS[model], rows[index]):
            raise ValueError("request aceita difere do prompt congelado")
        validate_response(json.loads(response_path.read_text()), TARGETS[model], rows[index])
        accepted.add(index)
    return accepted


def next_batch(groups, accepted):
    return next((i + 1 for i, indices in enumerate(groups) if not set(indices) <= accepted), None)


def export_progress(output, rows, records, accepted):
    groups = batch_indices(records, rows)
    report = {
        "expected_documents": 1000,
        "accepted_generations": len(accepted),
        "expected_generations": 6000,
        "complete": len(accepted) == 6000,
        "next_batch": next_batch(groups, accepted),
        "batches": [
            {
                "batch": i + 1,
                "documents": [r["doc_id"] for r in records[i * 100 : (i + 1) * 100]],
                "accepted": len(set(indices) & accepted),
                "expected": 600,
                "complete": set(indices) <= accepted,
            }
            for i, indices in enumerate(groups)
        ],
    }
    save(output / "batch-status.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "batches"}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ling-free", "qwen37", "gemma26"), default="ling-free")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-usd", type=money)
    parser.add_argument("--batch", type=int, choices=range(1, 11))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    model = args.model
    slug = {"ling-free": "ling", "qwen37": "qwen", "gemma26": "gemma"}[model]
    args.prepared = args.prepared or Path(f"outputs/full-{slug}-prepared")
    args.output = args.output or Path(f"outputs/full-{slug}-batches")
    log(f"{model}: iniciando lote; modo={'EXECUTAR' if args.execute else 'VALIDAR SEM API'}")
    with activity("1/4 Conferindo codigo, dataset e modelos congelados"):
        frozen = verify_full_lock(prepared=args.prepared)
    split, records = selected_records(args.prepared)
    if split != "eval":
        raise ValueError("lotes exclusivos da coorte final")
    cfg = ExperimentConfig.from_yaml(args.prepared / "config.yaml")
    plan = json.loads(Path("configs/openrouter_full.json").read_text())
    target = plan["models"][model]
    if TARGETS[model][:2] != (target["model"], target["provider"]):
        raise ValueError("endpoint difere do congelado")
    ceiling = money(0 if model == "ling-free" else frozen["budgets_usd"][f"{slug}_generation"])
    budget = ceiling if args.budget_usd is None else args.budget_usd
    if budget > ceiling:
        raise ValueError("orcamento excede teto congelado do modelo")
    with activity("2/4 Validando integridade e tokens dos 6000 prompts"):
        rows = load_prepared(args.prepared, model, 1000)
    limits = Limits(
        cfg.generation.max_input_tokens, cfg.generation.max_new_tokens, target["context_length"]
    )
    for row in rows:
        limits.check(row["prompt_tokens"])
        if row["max_output_tokens"] != limits.max_output or not row.get("allow_length"):
            raise ValueError("politica de saida difere do congelado")
    groups = batch_indices(records, rows)
    identity = {
        "full_lock_id": json.loads((args.prepared / "full-lock.json").read_text())["lock_id"],
        "model": model,
        "split": "eval",
        "documents": [r["doc_id"] for r in records],
        "preflight_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "batch_size": 100,
    }
    log("3/4 Conferindo respostas salvas e selecionando lote pendente")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".batches.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("outra execucao usa estes lotes") from None
        folder = args.output / model
        accepted = accepted_cases(folder, rows, identity, model)
        report = export_progress(args.output, rows, records, accepted)
        selected = args.batch or report["next_batch"]
        if selected is None:
            print("Todos os lotes completos; nenhuma geracao pendente.")
        else:
            pending = set(groups[selected - 1]) - accepted
            print(
                f"Lote {selected}/10: {len(pending)} chamadas pendentes (sem retries).", flush=True
            )
        if not args.execute:
            log(
                f"Preflight dos 6000 prompts {model} validado; nenhuma chamada API. "
                f"Teto cumulativo de geracao: US$ {budget} para os dez lotes."
            )
            return
        with activity("4/4 Conferindo recursos das metricas"):
            ensure_meteor_resources()
        if selected is not None and pending:
            transport = OpenRouterFreeClient(load_key(), timeout=240)
            remaining = None
            if model == "ling-free":
                with activity("Consultando cota gratuita"):
                    quota = transport.quota()
                save(args.output / "quota-before.json", quota)
                remaining = (quota.get("free_model_daily_requests") or {}).get("remaining")
                if type(remaining) is not int or remaining < 0:
                    raise ValueError("cota gratuita indisponivel; nao iniciar chamadas")
                log(f"Cota restante: {remaining}")
            log(f"Teto cumulativo para os dez lotes: US$ {budget}; modelo {model}")
            try:
                run(
                    rows,
                    TARGETS[model],
                    folder,
                    budget,
                    identity,
                    transport,
                    max_new_calls=remaining,
                    case_indices=set(groups[selected - 1]),
                    retry_reset_after=3600,
                )
            finally:
                accepted = accepted_cases(folder, rows, identity, model)
                report = export_progress(args.output, rows, records, accepted)
        if report["complete"]:
            with activity(f"Calculando metricas dos 1000 documentos: {model}"):
                score_results({model: rows}, records, cfg, args.prepared, args.output, {}, "eval")
        else:
            print(
                "Lote encerrado. Repita o comando para retomar/avancar. Hipoteses somente aos 1000."
            )


if __name__ == "__main__":
    main()
