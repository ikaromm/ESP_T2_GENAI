"""Teste de rajadas pagas em casos reais, sem repetir respostas aceitas.

Um coordenador por modelo grava ledger/reservas; threads somente fazem HTTP.
100 casos por patamar nao comprovam vazao sustentada por um minuto.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import statistics
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path

from findsum_rag.openrouter import OpenRouterFreeClient, OpenRouterHTTPError
from findsum_rag.progress import log
from scripts.execution.run_experiment_openrouter import load_key
from scripts.execution.run_ling_batches import export_progress, load_model, output_lock
from scripts.execution.run_paid_concurrent import dispatch_delay, held_cost, invoke
from scripts.execution.run_prepared_paid import (
    TARGETS,
    money,
    request_payload,
    reservation,
    run,
    validate_response,
)
from scripts.experiments.screen_openrouter import save

RATES = (500, 600, 700, 800, 900, 1000)
MODELS = ("qwen37", "gemma26")
ROOT = Path("outputs/paid-rate-benchmark-20260927")


def summarize(attempts, started, ended, rpm, peak, requested, reason):
    accepted = [a for a in attempts if a["status"] == "accepted"]
    latencies = sorted(a["elapsed_seconds"] for a in attempts if "elapsed_seconds" in a)
    launches = sorted(a["at"] for a in attempts)
    span = launches[-1] - launches[0] if len(launches) > 1 else 0
    elapsed = ended - started
    return {
        "target_rpm": rpm,
        "requested_cases": requested,
        "attempts": len(attempts),
        "accepted_cases": len({a["case"] for a in accepted}),
        "statuses": dict(Counter(a["status"] for a in attempts)),
        "elapsed_seconds": elapsed,
        "dispatch_span_seconds": span,
        "dispatch_rpm": (len(launches) - 1) * 60 / span if span else None,
        "completed_rpm": len(accepted) * 60 / elapsed if elapsed else None,
        "peak_in_flight": peak,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
        "latency_p95_seconds": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        if latencies
        else None,
        "reported_cost_usd": str(sum(money(a.get("reported_cost_usd", 0)) for a in attempts)),
        "stop_reason": reason,
        "sustained_rpm_validated": False,
    }


def stage(state, indices, rpm, transport, report_folder, *, max_workers=100):
    """Ao primeiro erro, para novas submissoes; recolhe todas as ja em voo."""
    state.folder.mkdir(parents=True, exist_ok=True)
    ledger_path = state.folder / "ledger.json"
    window_path = state.folder / "request-window.json"
    with (state.folder / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else {
                "identity": state.identity,
                "attempts": [],
            }
        )
        if ledger["identity"] != state.identity:
            raise ValueError("identidade do ledger diverge")
        old_count = len(ledger["attempts"])
        previous = {a["case"]: a for a in ledger["attempts"]}
        todo = [i for i in indices if previous.get(i, {}).get("status") != "accepted"]
        if any(i in previous for i in todo):
            raise ValueError("pendencias anteriores exigem retomada/auditoria antes do benchmark")
        sent = json.loads(window_path.read_text()) if window_path.exists() else []
        started, peak, reason = time.time(), 0, None
        cursor, futures = 0, {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while futures or (cursor < len(todo) and reason is None):
                now = time.time()
                delay = dispatch_delay(sent, rpm, now)
                if (
                    reason is None
                    and cursor < len(todo)
                    and len(futures) < max_workers
                    and delay <= 0
                ):
                    index = todo[cursor]
                    row = state.rows[index]
                    reserve = reservation(TARGETS[state.model], row)
                    if held_cost(ledger) + reserve > state.budget:
                        reason = "budget_limit"
                        continue
                    number = sum(a["case"] == index for a in ledger["attempts"])
                    stem = f"{index:04d}-{number}"
                    payload = request_payload(TARGETS[state.model], row)
                    entry = {
                        "case": index,
                        "doc_id": row["doc_id"],
                        "arm": row["arm"],
                        "reserved_usd": str(reserve),
                        "status": "pending",
                        "at": now,
                        "response_file": stem + ".response.json",
                        "benchmark_rpm": rpm,
                    }
                    save(state.folder / (stem + ".request.json"), payload)
                    ledger["attempts"].append(entry)
                    save(ledger_path, ledger)
                    sent = [t for t in sent if t > now - 60] + [now]
                    save(window_path, sent)
                    futures[pool.submit(invoke, transport, payload)] = entry
                    peak = max(peak, len(futures))
                    cursor += 1
                if futures:
                    done, _ = wait(futures, timeout=0.01, return_when=FIRST_COMPLETED)
                    for future in done:
                        entry = futures.pop(future)
                        response, error, elapsed = future.result()
                        entry["elapsed_seconds"] = elapsed
                        entry["completed_at"] = time.time()
                        if response is not None:
                            save(state.folder / entry["response_file"], response)
                            if "cost" in response.get("usage", {}):
                                entry["reported_cost_usd"] = response["usage"]["cost"]
                            try:
                                validate_response(
                                    response, TARGETS[state.model], state.rows[entry["case"]]
                                )
                                entry["status"] = "accepted"
                            except Exception as exc:
                                error = exc
                        if error is not None:
                            if isinstance(error, OpenRouterHTTPError):
                                entry["status"] = (
                                    "http429"
                                    if error.status == 429
                                    else (
                                        "http_retryable"
                                        if error.status in {408, 500, 502, 503, 504}
                                        else "rejected"
                                    )
                                )
                                entry["error"] = error.details
                                reason = reason or f"http{error.status}"
                            else:
                                entry["status"] = "audit_required"
                                entry["error_type"] = type(error).__name__
                                reason = reason or "audit_required"
                        save(ledger_path, ledger)
                        completed = sum(
                            a["status"] != "pending" for a in ledger["attempts"][old_count:]
                        )
                        if completed % 10 == 0 or reason:
                            log(
                                f"{state.model} {rpm} RPM: {completed}/{len(todo)} concluidas; "
                                f"{len(futures)} em voo; parada={reason}"
                            )
                elif reason is None and cursor < len(todo):
                    time.sleep(min(max(delay, 0.01), 0.1))
        # .run.lock e liberado antes do executor serial de retries.
    report = summarize(
        ledger["attempts"][old_count:], started, time.time(), rpm, peak, len(todo), reason
    )
    report["retry_recovery"] = None
    save(report_folder / f"{rpm}.json", report)
    retry_cases = {
        a["case"]
        for a in ledger["attempts"][old_count:]
        if a["status"] in {"http429", "http_retryable"}
    }
    if retry_cases:
        delays = [
            OpenRouterHTTPError(a["error"]).retry_delay(1)
            for a in ledger["attempts"][old_count:]
            if a["case"] in retry_cases
        ]
        delay = max(delays)
        if delay <= 60:
            log(
                f"{state.model}: recuperando {len(retry_cases)} falhas transitorias a 100 RPM, "
                f"apos {delay:.1f}s; nao sobe ao proximo patamar"
            )
            time.sleep(delay)
            try:
                run(
                    state.rows,
                    TARGETS[state.model],
                    state.folder,
                    state.budget,
                    state.identity,
                    transport,
                    case_indices=retry_cases,
                )
                report["retry_recovery"] = "completed"
            except Exception as exc:
                report["retry_recovery"] = type(exc).__name__
        else:
            report["retry_recovery"] = "deferred_retry_after_exceeds_60s"
        save(report_folder / f"{rpm}.json", report)
    return report


def model_sweep(state, transport, root, *, lower_rates=False):
    folder = root / state.model
    folder.mkdir(exist_ok=True)
    reports = []
    with output_lock(state.output):
        try:
            for position, rpm in enumerate((100, 200, 300, 400) if lower_rates else RATES):
                report_path = folder / f"{rpm}.json"
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                else:
                    log(f"{state.model}: iniciando {rpm} RPM, 100 geracoes reais")
                    case_plan = folder / f"{rpm}-cases.json"
                    if case_plan.exists():
                        indices = json.loads(case_plan.read_text())
                    elif lower_rates:
                        accepted = state.accepted()
                        indices = [i for i in range(600) if i not in accepted][:100]
                    else:
                        indices = list(range(position * 100, (position + 1) * 100))
                    if not indices:
                        break
                    save(case_plan, indices)
                    report = stage(
                        state,
                        indices,
                        rpm,
                        transport,
                        folder,
                    )
                reports.append(report)
                if report["stop_reason"] or report["accepted_cases"] != 100:
                    break
        finally:
            export_progress(state.output, state.rows, state.records, state.accepted())
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--lower-rates",
        action="store_true",
        help="apos falha em 500, testa 100/200/300/400 nas pendencias do mesmo lote",
    )
    args = parser.parse_args()
    root = ROOT / "lower-rates" if args.lower_rates else ROOT
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(output_lock(Path("outputs/full-rounds")))
        stack.enter_context(output_lock(root))
        states = {model: load_model(model) for model in MODELS}
        if states[MODELS[0]].records != states[MODELS[1]].records:
            raise ValueError("coortes diferentes")
        plan = {
            "rates": list((100, 200, 300, 400) if args.lower_rates else RATES),
            "cases_per_rate": 100,
            "models": {name: state.identity for name, state in states.items()},
            "first_document_ids": [r["doc_id"] for r in states[MODELS[0]].records[:100]],
            "max_concurrent_per_model": 100,
            "sustained_test": False,
        }
        path = root / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("plano do benchmark mudou")
        save(path, plan)
        if not args.execute:
            log("Benchmark validado sem API; 6 patamares x 100 casos x 2 modelos")
            return
        client = OpenRouterFreeClient(load_key(), timeout=240)
        credits = client._request("/credits")["data"]
        remaining = money(credits["total_credits"]) - money(credits["total_usage"])
        worst = sum(
            reservation(TARGETS[name], state.rows[0]) * 600 for name, state in states.items()
        )
        save(
            root / "credit-check.json",
            {"remaining_usd": str(remaining), "conservative_1200_calls_usd": str(worst)},
        )
        if remaining < worst:
            raise ValueError("saldo insuficiente para reserva conservadora dos 1200 casos")
        started = time.time()
        results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(model_sweep, state, client, root, lower_rates=args.lower_rates): name
                for name, state in states.items()
            }
            for future, name in futures.items():
                try:
                    results[name] = future.result()
                except Exception as exc:
                    results[name] = {"error_type": type(exc).__name__}
                save(
                    root / "results.json",
                    {"elapsed_seconds": time.time() - started, "models": results},
                )
        log(f"Benchmark encerrado: {root}/results.json; respostas ficam nos ledgers do full")


if __name__ == "__main__":
    main()
