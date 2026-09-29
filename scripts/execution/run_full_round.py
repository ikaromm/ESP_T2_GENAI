"""Rodadas retomaveis dos tres modelos; o Bash executa e --dry-run apenas valida."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from findsum_rag.full_lock import DEFAULT_LOCK, verify_full_lock
from findsum_rag.progress import log
from scripts.evaluation.render_readable_progress import render_readable_progress
from scripts.evaluation.update_progress_metrics import update_metrics
from scripts.execution.run_ling_batches import (
    ARMS,
    document_indices,
    load_model,
    money,
    output_lock,
    run_selection,
)
from scripts.execution.run_prepared_paid import (
    TARGETS,
    paid_ling_amendments,
    paid_ling_budgets,
    target_for_saved_attempt,
)
from scripts.experiments.screen_openrouter import save

MODELS = ("ling-free", "gemma26", "qwen37")
SLUGS = {"ling-free": "ling", "gemma26": "gemma", "qwen37": "qwen"}
ROOT = Path("outputs/full-rounds")


def refresh_progress(records, states):
    """Calcula somente pares pendentes e atualiza as figuras legiveis a partir do resumo."""
    update_metrics(records, states)
    render_readable_progress()


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
            try:
                target_for_saved_attempt(
                    TARGETS[state.model], state.folder, entry, state.rows[entry["case"]]
                )
            except ValueError as exc:
                raise ValueError("request do historico difere do prompt congelado") from exc
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
    subprocess.run([sys.executable, "-u", "-m", script, *arguments], check=True)


def prepare_missing(model, frozen):
    prepared, _ = paths(model)
    if model == "gemma26":
        if prepared.exists():
            raise ValueError("preparacao Gemma incompleta; preserve os artefatos para auditoria")
        child(
            "scripts.preparation.prepare_gemma_from_common",
            ["--source", "outputs/full-ling-prepared", "--output", str(prepared)],
        )
    elif model == "qwen37":
        if prepared.exists():
            raise ValueError("preparacao Qwen incompleta; preserve os artefatos para auditoria")
        log("Qwen: preparacao LOCAL dos 1000 documentos; zero sondas pagas")
        child(
            "scripts.preparation.prepare_gemma_from_common",
            [
                "--model",
                "qwen37",
                "--source",
                "outputs/full-ling-prepared",
                "--output",
                str(prepared),
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


def round_amendment(plan, records):
    """Adendo registrado para exatamente este plano; None quando nao existe."""
    for amendment, amendment_sha in paid_ling_amendments():
        if amendment["plan_id"] != plan.get("plan_id"):
            continue
        first, last = amendment["first_case"], amendment["last_case_exclusive"]
        if (
            amendment["cohort_sha256"] != plan["cohort_sha256"]
            or plan["documents"] != [r["doc_id"] for r in records[first // 6 : last // 6]]
            or document_indices(records, plan["documents"]) != set(range(first, last))
        ):
            raise ValueError("rodada ativa difere do adendo pago autorizado")
        return amendment, amendment_sha
    return None


def paid_ling_for_plan(plan, records):
    found = round_amendment(plan, records)
    if found is None:
        raise ValueError("rodada ativa difere do adendo pago autorizado")
    return found[1]


def amendment_round_budgets(amendment):
    """Tetos incrementais por modelo; o adendo historico v1 nao os define."""
    if amendment is None or amendment["version"] == 1:
        return {}
    return {model: money(value) for model, value in amendment["round_budget_usd"].items()}


def execute_round(
    plan, records, states, accepted, frozen, root, *, paid_ling=None, round_budgets=None
):
    errors = {}
    stop_event = Event()
    round_budgets = round_budgets or {}

    def execute_model(model):
        error = None
        try:
            if model not in states:
                prepare_missing(model, frozen)
                states[model] = load_model(model, *paths(model))
                if states[model].records != records:
                    raise ValueError("coorte preparada diverge")
            kwargs = {"paid_ling": paid_ling} if model == "ling-free" and paid_ling else {}
            # Ling gratuito custa zero por contrato; o teto so vale nos executores pagos.
            if model in round_budgets and (model != "ling-free" or paid_ling):
                kwargs["round_budget"] = round_budgets[model]
            run_selection(
                states[model],
                documents=plan["documents"],
                execute=True,
                stop_event=stop_event,
                **kwargs,
            )
        except Exception as exc:
            error = {"type": type(exc).__name__}
            if isinstance(exc, ValueError):
                error["message"] = str(exc)
            log(f"{model}: etapa interrompida ({error}); rodada preservada")
        finally:
            if model in states:
                accepted[model] = states[model].accepted()
        return error

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(execute_model, model): model for model in MODELS}
        while futures:
            try:
                done, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    model = futures.pop(future)
                    try:
                        error = future.result()
                    except Exception as exc:
                        error = {"type": type(exc).__name__}
                    if error:
                        errors[model] = error
            except KeyboardInterrupt:
                stop_event.set()
                log(
                    "Interrompendo envios; recolhendo respostas ja em voo para preservar a retomada"
                )
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
        "--metrics-only", action="store_true", help="atualiza metricas e painel sem geracao"
    )
    modes.add_argument(
        "--audit-ling-batch",
        type=int,
        choices=range(1, 11),
        help="somente auditoria local do lote Ling indicado",
    )
    parser.add_argument(
        "--ling-paid-this-round",
        action="store_true",
        help="usa o adendo registrado de Ling pago cujo plano coincide com esta rodada",
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
        if args.dry_run or args.metrics_only
        else "MODO EXECUCAO: Ling pago Novita nesta rodada + Gemma/Qwen pagos"
        if args.ling_paid_this_round
        else "MODO EXECUCAO: Ling gratuito + Gemma/Qwen pagos; preparacao Qwen sem API"
    )
    with output_lock(ROOT):
        frozen = verify_full_lock(DEFAULT_LOCK)
        records = frozen["cohort"]
        states, accepted = inspect_models(records)
        if args.metrics_only:
            refresh_progress(records, states)
            return
        audit_batch(states["ling-free"], 1)
        # O plano so e salvo depois de casar com o adendo, se o Ling pago foi pedido.
        plan = round_plan(ROOT, records, accepted, args.cases, persist=False)
        if plan is None:
            log("Os tres modelos ja concluiram todos os documentos; nenhuma geracao pendente")
            if not args.dry_run:
                refresh_progress(records, states)
            return
        found = round_amendment(plan, records)
        if args.ling_paid_this_round and found is None:
            raise ValueError(
                "rodada difere de todos os adendos de Ling pago registrados; nada foi salvo"
            )
        paid_ling = found[1] if args.ling_paid_this_round else None
        budgets = amendment_round_budgets(found[0] if found else None)
        if not args.dry_run:
            plan = round_plan(ROOT, records, accepted, args.cases, persist=True)
        pending = remaining(records, plan["documents"], accepted)
        preview = {
            "plan": plan,
            "pending_generations": pending,
            "preparation_pending": [m for m in MODELS if m not in states],
            "dry_run": args.dry_run,
            "amendment_sha256": found[1] if found else None,
            "ling_paid": paid_ling is not None,
            "round_budget_usd": {model: str(value) for model, value in budgets.items()},
        }
        save(ROOT / "preview.json", preview)
        log(f"Rodada de {len(plan['documents'])} documentos; pendencias: {pending}")
        if budgets:
            log(
                "Tetos incrementais desta rodada (custo retido nos casos da rodada): "
                + "; ".join(f"{model} US${budgets[model]}" for model in MODELS)
                + f"; total US${sum(budgets.values())}"
            )
        ling_cap = paid_ling_budgets(found[0])[0] if paid_ling else 0
        log(
            f"Tetos cumulativos: Ling US${ling_cap}; "
            "Gemma US$9; Qwen geracao US$18; preparacao local US$0"
        )
        if args.dry_run:
            log("Validacao encerrada sem API. Nenhuma rodada ativa criada ou avancada")
            return
        result = execute_round(
            plan,
            records,
            states,
            accepted,
            frozen,
            ROOT,
            paid_ling=paid_ling,
            round_budgets=budgets,
        )
        log(
            f"Rodada {'concluida' if result['complete'] else 'pendente'}: "
            f"{result['pending_generations']}"
        )
        refresh_progress(records, states)
        if not result["complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
