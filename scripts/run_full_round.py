"""Rodadas retomaveis dos tres modelos; o Bash executa e --dry-run apenas valida."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from findsum_rag.full_lock import DEFAULT_LOCK, verify_full_lock
from findsum_rag.progress import log
from run_ling_batches import (
    ARMS,
    document_indices,
    load_model,
    money,
    output_lock,
    run_selection,
)
from run_prepared_paid import TARGETS, request_payload
from screen_openrouter import save

MODELS = ("ling-free", "gemma26", "qwen37")
SLUGS = {"ling-free": "ling", "gemma26": "gemma", "qwen37": "qwen"}
ROOT = Path("outputs/full-rounds")


def checksum(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def case_count(value):
    count = int(value)
    if not 1 <= count <= 1000:
        raise argparse.ArgumentTypeError("casos deve estar entre 1 e 1000")
    return count


def paths(model):
    slug = SLUGS[model]
    return Path(f"outputs/full-{slug}-prepared"), Path(f"outputs/full-{slug}-batches")


def choose_documents(records, accepted, count):
    """Primeiros documentos faltantes em qualquer modelo; nunca avanca so o Ling."""
    result = []
    for i, record in enumerate(records):
        indices = set(range(i * len(ARMS), (i + 1) * len(ARMS)))
        if any(not indices <= accepted[model] for model in MODELS):
            result.append(record["doc_id"])
            if len(result) == count:
                break
    return result


def round_plan(root, records, accepted, count, *, persist):
    active = root / "active-round.json"
    cohort = checksum(records)
    if active.exists():
        plan = json.loads(active.read_text())
        payload = {key: value for key, value in plan.items() if key != "plan_id"}
        if plan.get("plan_id") != checksum(payload) or plan.get("cohort_sha256") != cohort:
            raise ValueError("plano da rodada alterado ou coorte diferente")
        if plan["requested_cases"] != count:
            raise ValueError(f"rodada pendente: repita com {plan['requested_cases']} casos")
        document_indices(records, plan["documents"])
        if not 0 < len(plan["documents"]) <= count:
            raise ValueError("quantidade de documentos invalida no plano")
        return plan
    documents = choose_documents(records, accepted, count)
    if not documents:
        return None
    plan = {"version": 1, "requested_cases": count, "cohort_sha256": cohort, "documents": documents}
    plan["plan_id"] = checksum(plan)
    if persist:
        save(active, plan)
    return plan


def remaining(records, documents, accepted):
    indices = document_indices(records, documents)
    return {model: len(indices - accepted[model]) for model in MODELS}


def audit_batch(state, batch=1):
    """Audita artefatos, sem calcular hipoteses parciais ou avaliar qualidade factual."""
    with output_lock(state.output):
        accepted = state.accepted()
        docs = [r["doc_id"] for r in state.records[(batch - 1) * 100 : batch * 100]]
        indices = document_indices(state.records, docs)
        path = state.folder / "ledger.json"
        attempts = json.loads(path.read_text())["attempts"] if path.exists() else []
        selected = [entry for entry in attempts if entry["case"] in indices]
        for entry in selected:
            response_path = state.folder / entry["response_file"]
            request_path = response_path.with_name(
                response_path.name.replace(".response.", ".request.")
            )
            if json.loads(request_path.read_text()) != request_payload(
                TARGETS[state.model], state.rows[entry["case"]]
            ):
                raise ValueError("request do historico difere do prompt congelado")
        successes = [entry for entry in selected if entry["status"] == "accepted"]
        if len({entry["case"] for entry in successes}) != len(successes):
            raise ValueError("mais de uma resposta aceita para o mesmo caso")
        finishes, arms = Counter(), Counter()
        cost = money(0)
        input_tokens = output_tokens = 0
        for entry in successes:
            response = json.loads((state.folder / entry["response_file"]).read_text())
            finishes[response["choices"][0]["finish_reason"]] += 1
            arms[entry["arm"]] += 1
            cost += money(response["usage"]["cost"])
            input_tokens += response["usage"]["prompt_tokens"]
            output_tokens += response["usage"]["completion_tokens"]
        report = {
            "checked_at": datetime.now(UTC).isoformat(),
            "model": state.model,
            "batch": batch,
            "documents": docs,
            "expected_generations": len(indices),
            "accepted_generations": len(indices & accepted),
            "missing_cases": sorted(indices - accepted),
            "complete": indices <= accepted,
            "attempts": len(selected),
            "attempt_statuses": dict(Counter(entry["status"] for entry in selected)),
            "accepted_per_arm": dict(arms),
            "finish_reasons": dict(finishes),
            "reported_cost_usd": str(cost),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prepared_lock_id": state.identity["full_lock_id"],
            "validation": "frozen_inputs_requests_responses_model_provider_tokens_and_completion",
            "hypotheses_computed": False,
        }
        (state.output / "audits").mkdir(exist_ok=True)
        save(state.output / f"audits/batch-{batch:02d}.json", report)
        log(
            f"Auditoria {state.model} lote {batch}: {report['accepted_generations']}/"
            f"{len(indices)} aceitas; {len(selected)} tentativas; custo US$ {cost}"
        )
        return report


def child(script, arguments):
    subprocess.run([sys.executable, "-u", str(Path("scripts") / script), *arguments], check=True)


def prepare_missing(model, frozen):
    prepared, _ = paths(model)
    if model == "gemma26":
        if prepared.exists():
            raise ValueError("preparacao Gemma incompleta; preserve os artefatos para auditoria")
        child(
            "prepare_gemma_from_common.py",
            ["--source", "outputs/full-ling-prepared", "--output", str(prepared)],
        )
    elif model == "qwen37":
        log("Qwen: calibracao PAGA integral dos 1000 documentos, antes das geracoes da rodada")
        child(
            "prepare_qwen_remote.py",
            [
                "--prepared",
                "outputs/full-ling-prepared",
                "--output",
                str(prepared),
                "--budget-usd",
                str(frozen["budgets_usd"]["qwen_calibration"]),
                "--allow-eval",
                "--execute",
            ],
        )
    else:
        raise ValueError("preparacao Ling integral obrigatoria antes de usar o Bash")


def inspect_models(records):
    states, accepted = {}, {}
    for model in MODELS:
        prepared, output = paths(model)
        ready = (prepared / "sha256.json").is_file() and (
            prepared / model / "preflight.json"
        ).is_file()
        if ready:
            state = load_model(model, prepared, output)
            if state.records != records:
                raise ValueError("modelos com coortes diferentes")
            with output_lock(output):
                accepted[model] = state.accepted()
            states[model] = state
        else:
            if (output / model / "ledger.json").exists():
                raise ValueError(f"{model}: ha tentativas salvas sem preparacao completa")
            if model == "ling-free":
                raise ValueError("preparacao Ling integral ausente")
            accepted[model] = set()
            log(f"{model}: preparacao pendente; sera feita ao executar")
    return states, accepted


def execute_round(plan, records, states, accepted, frozen, root):
    errors = {}
    for model in MODELS:
        try:
            if model not in states:
                prepare_missing(model, frozen)
                states[model] = load_model(model, *paths(model))
                if states[model].records != records:
                    raise ValueError("coorte preparada diverge")
            run_selection(states[model], documents=plan["documents"], execute=True)
        except Exception as exc:
            # Erros de um modelo nao impedem a tentativa dos outros na MESMA rodada.
            # Nao serializar corpos de erros HTTP, que podem conter dados privados.
            errors[model] = {"type": type(exc).__name__}
            if isinstance(exc, ValueError):
                errors[model]["message"] = str(exc)
            log(f"{model}: etapa interrompida ({errors[model]}); rodada preservada")
        finally:
            if model in states:
                accepted[model] = states[model].accepted()
    pending = remaining(records, plan["documents"], accepted)
    result = {
        "plan_id": plan["plan_id"],
        "documents": plan["documents"],
        "pending_generations": pending,
        "errors": errors,
        "complete": not any(pending.values()) and not errors,
    }
    save(root / "last-result.json", result)
    if result["complete"]:
        (root / "history").mkdir(exist_ok=True)
        save(root / "history" / f"{plan['plan_id']}.json", {"plan": plan, "result": result})
        (root / "active-round.json").unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=case_count, nargs="?", default=100)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="valida e planeja sem API")
    modes.add_argument(
        "--audit-ling-batch",
        type=int,
        choices=range(1, 11),
        help="somente auditoria local do lote Ling indicado",
    )
    args = parser.parse_args()
    if args.audit_ling_batch:
        state = load_model("ling-free")
        report = audit_batch(state, args.audit_ling_batch)
        if not report["complete"]:
            raise SystemExit(1)
        return
    log(
        "MODO SEM API"
        if args.dry_run
        else "MODO EXECUCAO: Ling gratuito + Gemma/Qwen pagos; calibracao Qwen paga se pendente"
    )
    with output_lock(ROOT):
        frozen = verify_full_lock(DEFAULT_LOCK)
        records = frozen["cohort"]
        states, accepted = inspect_models(records)
        audit_batch(states["ling-free"], 1)
        plan = round_plan(ROOT, records, accepted, args.cases, persist=not args.dry_run)
        if plan is None:
            log("Os tres modelos ja concluiram todos os documentos; nenhuma geracao pendente")
            return
        pending = remaining(records, plan["documents"], accepted)
        preview = {
            "plan": plan,
            "pending_generations": pending,
            "preparation_pending": [m for m in MODELS if m not in states],
            "dry_run": args.dry_run,
        }
        save(ROOT / "preview.json", preview)
        log(f"Rodada de {len(plan['documents'])} documentos; pendencias: {pending}")
        log("Tetos cumulativos: Ling US$0; Gemma US$9; Qwen geracao US$18 + calibracao US$18")
        if args.dry_run:
            log("Validacao encerrada sem API. Nenhuma rodada ativa criada ou avancada")
            return
        result = execute_round(plan, records, states, accepted, frozen, ROOT)
        log(
            f"Rodada {'concluida' if result['complete'] else 'pendente'}: "
            f"{result['pending_generations']}"
        )
        if not result["complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
