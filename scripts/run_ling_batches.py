"""Um lote de 100 do Ling gratuito, parte da coorte final congelada de 1000."""

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


def accepted_cases(folder, rows, identity):
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
        if json.loads(request_path.read_text()) != request_payload(
            TARGETS["ling-free"], rows[index]
        ):
            raise ValueError("request aceita difere do prompt congelado")
        validate_response(json.loads(response_path.read_text()), TARGETS["ling-free"], rows[index])
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
    parser.add_argument("--prepared", type=Path, default=Path("outputs/full-ling-prepared"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full-ling-batches"))
    parser.add_argument("--batch", type=int, choices=range(1, 11))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    frozen = verify_full_lock(prepared=args.prepared)
    split, records = selected_records(args.prepared)
    if split != "eval":
        raise ValueError("lotes exclusivos da coorte final")
    cfg = ExperimentConfig.from_yaml(args.prepared / "config.yaml")
    plan = json.loads(Path("configs/openrouter_full.json").read_text())
    target = plan["models"]["ling-free"]
    if TARGETS["ling-free"][:2] != (target["model"], target["provider"]):
        raise ValueError("endpoint difere do congelado")
    rows = load_prepared(args.prepared, "ling-free", 1000)
    limits = Limits(
        cfg.generation.max_input_tokens, cfg.generation.max_new_tokens, target["context_length"]
    )
    for row in rows:
        limits.check(row["prompt_tokens"])
        if row["max_output_tokens"] != limits.max_output or not row.get("allow_length"):
            raise ValueError("politica de saida difere do congelado")
    groups = batch_indices(records, rows)
    identity = {
        "full_lock_id": frozen["lock_id"],
        "model": "ling-free",
        "split": "eval",
        "documents": [r["doc_id"] for r in records],
        "preflight_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "batch_size": 100,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".batches.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("outra execucao usa estes lotes") from None
        folder = args.output / "ling-free"
        accepted = accepted_cases(folder, rows, identity)
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
            print("Preflight dos 6000 prompts Ling validado; nenhuma chamada API.")
            return
        ensure_meteor_resources()
        if selected is not None and pending:
            transport = OpenRouterFreeClient(load_key(), timeout=240)
            quota = transport.quota()
            save(args.output / "quota-before.json", quota)
            remaining = (quota.get("free_model_daily_requests") or {}).get("remaining")
            if type(remaining) is not int or remaining < 0:
                raise ValueError("cota gratuita indisponivel; nao iniciar chamadas")
            print(f"Cota restante: {remaining}; teto de custo: US$ 0.", flush=True)
            try:
                run(
                    rows,
                    TARGETS["ling-free"],
                    folder,
                    money(0),
                    identity,
                    transport,
                    max_new_calls=remaining,
                    case_indices=set(groups[selected - 1]),
                    retry_reset_after=3600,
                )
            finally:
                accepted = accepted_cases(folder, rows, identity)
                report = export_progress(args.output, rows, records, accepted)
        if report["complete"]:
            score_results({"ling-free": rows}, records, cfg, args.prepared, args.output, {}, "eval")
        else:
            print(
                "Lote encerrado. Repita o comando para retomar/avancar. Hipoteses somente aos 1000."
            )


if __name__ == "__main__":
    main()
