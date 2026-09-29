"""Geracao paga concorrente com pacing, backoff persistente e orcamento em voo."""

from __future__ import annotations

import fcntl
import json
import math
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from findsum_rag.openrouter import OpenRouterHTTPError
from findsum_rag.progress import log
from scripts.execution.run_prepared_paid import (
    TARGETS,
    money,
    request_payload,
    reservation,
    validate_response,
)
from scripts.experiments.screen_openrouter import save

MAX_RPM = 500
MAX_CONCURRENT = 100
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


class EmbeddedProviderError(RuntimeError):
    """Erro explicito do provedor dentro de HTTP 200; nao inventa status HTTP."""

    def __init__(self, details):
        self.details = details
        self.status = details["code"]
        super().__init__(f"erro embutido do provedor: {details['code']}")

    def retry_delay(self, default):
        return default


def embedded_error(response):
    """Somente choices[0] com finish_reason=error e codigo transitorio explicito."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    error = choices[0].get("error")
    if choices[0].get("finish_reason") != "error" or not isinstance(error, dict):
        return None
    code = error.get("code")
    if type(code) is not int or code not in TRANSIENT_STATUSES:
        return None
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    return EmbeddedProviderError(
        {
            "code": code,
            "message": error.get("message"),
            "metadata": {k: metadata[k] for k in ("error_type", "provider_name") if k in metadata},
        }
    )


def dispatch_delay(sent, rpm, now):
    recent = [t for t in sent if t > now - 60]
    spacing = max(0.0, sent[-1] + 60 / rpm - now) if sent else 0.0
    rolling = max(0.0, recent[-rpm] + 60 - now) if len(recent) >= rpm else 0.0
    return max(spacing, rolling)


def held_cost(ledger, cases=None):
    """Custo retido: aceito pelo custo reportado; demais pela reserva maxima."""
    return sum(
        (
            money(a["reported_cost_usd"] if a["status"] == "accepted" else a["reserved_usd"])
            for a in ledger["attempts"]
            if cases is None or a["case"] in cases
        ),
        money(0),
    )


def invoke(transport, payload):
    started = time.monotonic()
    try:
        return transport._request("/chat/completions", payload), None, time.monotonic() - started
    except Exception as exc:
        return None, exc, time.monotonic() - started


def transient(error):
    if isinstance(error, EmbeddedProviderError):
        return error.status in TRANSIENT_STATUSES
    if not isinstance(error, OpenRouterHTTPError):
        return False
    meta = error.details.get("metadata", {})
    return error.status in TRANSIENT_STATUSES or (
        error.status == 402
        and meta.get("limit_source") == "openrouter_in_flight_budget"
        and meta.get("reason") == "in_flight_budget_exhausted"
    )


class AdaptiveRate:
    def __init__(self, data=None):
        self.data = data or dict(
            rpm=MAX_RPM, pause_until=0, last_error=0, last_increase=0, successes=0
        )
        if not 50 <= self.data["rpm"] <= MAX_RPM:
            raise ValueError("estado adaptativo invalido")

    @property
    def concurrent(self):
        return max(1, math.ceil(MAX_CONCURRENT * self.data["rpm"] / MAX_RPM))

    def failure(self, error, attempt_in_cycle, now, jitter):
        delay = error.retry_delay(2 ** (attempt_in_cycle - 1)) + jitter
        self.data.update(
            rpm=max(50, self.data["rpm"] // 2),
            successes=0,
            last_error=now,
            pause_until=max(self.data["pause_until"], now + delay),
        )
        return now + delay

    def success(self, now):
        self.data["successes"] += 1
        if (
            self.data["successes"] >= 25
            and now - max(self.data["last_error"], self.data["last_increase"]) >= 30
        ):
            self.data.update(
                rpm=min(MAX_RPM, self.data["rpm"] + 50), successes=0, last_increase=now
            )


def run(
    rows,
    target,
    output,
    budget,
    identity,
    transport,
    *,
    case_indices=None,
    stop_event=None,
    clock=time.time,
    sleep=time.sleep,
    jitter=None,
    amendment_sha256=None,
    round_budget=None,
):
    """`budget` limita o ledger inteiro; `round_budget`, so os casos selecionados."""
    if target[0].endswith(":free"):
        raise ValueError("executor concorrente exclusivo dos modelos pagos")
    if amendment_sha256 is not None and target != TARGETS["ling-paid-novita"]:
        raise ValueError("adendo pago exclusivo do Ling via Novita")
    if round_budget is not None:
        round_budget = money(round_budget)
        if case_indices is None:
            raise ValueError("teto incremental exige os casos da rodada")
    jitter = jitter or (lambda: random.uniform(0, 0.5))
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "ledger.json"
        if not path.exists() and any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("diretorio contem artefatos sem ledger; exige auditoria")
        ledger = (
            json.loads(path.read_text()) if path.exists() else dict(identity=identity, attempts=[])
        )
        if ledger["identity"] != identity:
            raise ValueError("identidade da retomada mudou")
        selected = sorted(range(len(rows)) if case_indices is None else case_indices)
        if any(type(i) is not int or not 0 <= i < len(rows) for i in selected):
            raise ValueError("indice invalido")
        history = {}
        for a in ledger["attempts"]:
            history.setdefault(a["case"], []).append(a)
        queue, cycles = {}, {}
        for index in selected:
            old = history.get(index, [])
            if old and old[-1]["status"] == "accepted":
                response_path = output / old[-1]["response_file"]
                request_path = response_path.with_name(
                    response_path.name.replace(".response.", ".request.")
                )
                if json.loads(request_path.read_text()) != request_payload(target, rows[index]):
                    raise ValueError("request aceita diverge")
                validate_response(json.loads(response_path.read_text()), target, rows[index])
                continue
            if old and old[-1]["status"] not in {"http429", "http_retryable"}:
                raise ValueError("tentativa incerta/rejeitada exige auditoria antes de retomar")
            end = old[-1].get("retry_cycle_end", 6) if old else 6
            if old and amendment_sha256 is not None:
                if "response_file" in old[-1]:
                    last_request = output / old[-1]["response_file"].replace(
                        ".response.", ".request."
                    )
                elif index == 960 and len(old) == 1:
                    last_request = output / "0960-0.request.json"
                else:
                    raise ValueError("tentativa anterior sem request auditavel")
                previous_model = json.loads(last_request.read_text())["model"]
                if previous_model != target[0]:
                    if (
                        previous_model != "inclusionai/ling-3.0-flash-fin:free"
                        or old[-1].get("error", {}).get("http_status") != 404
                        or "unavailable for free"
                        not in str(old[-1]["error"].get("message", "")).lower()
                    ):
                        raise ValueError("troca de endpoint sem auditoria do erro anterior")
                    end = len(old) + 6
            if len(old) >= end:
                if clock() - old[-1]["at"] < 3600:
                    raise ValueError("seis tentativas; circuito aberto por uma hora")
                end = len(old) + 6
            cycles[index] = end
            ready = old[-1].get("retry_at", 0) if old else 0
            if old and old[-1].get("error"):
                ready = max(
                    ready, old[-1]["at"] + OpenRouterHTTPError(old[-1]["error"]).retry_delay(1)
                )
            queue[index] = ready
        if not queue:
            return ledger
        round_cases = set(selected)
        window_path, rate_path = output / "request-window.json", output / "adaptive-rate.json"
        sent = json.loads(window_path.read_text()) if window_path.exists() else []
        rate = AdaptiveRate(json.loads(rate_path.read_text()) if rate_path.exists() else None)
        reason, futures = None, {}
        completed = len(selected) - len(queue)
        last_log = clock()
        log(
            f"{target[0]}: concorrente, teto 500 RPM; ritmo atual {rate.data['rpm']} RPM; "
            f"{len(queue)} pendencias; retido US$ {held_cost(ledger)}/{budget}"
            + (
                f"; retido na rodada US$ {held_cost(ledger, round_cases)}/{round_budget}"
                if round_budget is not None
                else ""
            )
        )
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            while futures or (queue and reason is None):
                try:
                    now = clock()
                    if stop_event is not None and stop_event.is_set():
                        reason = "interrompido"
                    # Processa resultados antes de decidir novos envios/retries.
                    done, _ = (
                        wait(futures, timeout=0.01, return_when=FIRST_COMPLETED)
                        if futures
                        else ([], [])
                    )
                    for future in done:
                        entry = futures.pop(future)
                        response, error, elapsed = future.result()
                        entry.update(elapsed_seconds=elapsed, completed_at=clock())
                        if response is not None:
                            save(output / entry["response_file"], response)
                            if "cost" in response.get("usage", {}):
                                entry["reported_cost_usd"] = response["usage"]["cost"]
                            same_endpoint = (response.get("model"), response.get("provider")) == (
                                target[0],
                                target[2],
                            )
                            error = embedded_error(response) if same_endpoint else None
                            if error is None:
                                try:
                                    validate_response(response, target, rows[entry["case"]])
                                    entry["status"] = "accepted"
                                    completed += 1
                                    rate.success(clock())
                                except Exception as exc:
                                    error = exc
                        if error is not None:
                            if isinstance(error, OpenRouterHTTPError):
                                entry["error"] = error.details
                                entry["status"] = (
                                    "http429"
                                    if error.status == 429
                                    else "http_retryable"
                                    if transient(error)
                                    else "rejected"
                                )
                            elif isinstance(error, EmbeddedProviderError):
                                # Resposta preservada; o custo retido continua sendo a reserva.
                                entry.update(
                                    status="http_retryable",
                                    embedded_error=error.details,
                                    retry_classification="embedded_provider_error",
                                )
                            else:
                                entry.update(
                                    status="audit_required", error_type=type(error).__name__
                                )
                            if transient(error):
                                index = entry["case"]
                                attempt = len(history[index]) - (cycles[index] - 6)
                                ready = rate.failure(error, attempt, clock(), jitter())
                                entry["retry_at"] = ready
                                if attempt >= 6:
                                    reason = reason or "seis tentativas; circuito aberto"
                                elif ready - clock() > 60:
                                    reason = reason or "Retry-After longo; retomar apos prazo salvo"
                                else:
                                    queue[index] = ready
                                log(
                                    f"{target[0]}: "
                                    + (
                                        "erro embutido"
                                        if isinstance(error, EmbeddedProviderError)
                                        else "HTTP"
                                    )
                                    + f" {error.status}; retry {attempt}/6; "
                                    f"ritmo reduzido a {rate.data['rpm']} RPM, "
                                    f"pausa {max(0, rate.data['pause_until'] - clock()):.1f}s"
                                )
                            else:
                                reason = reason or "falha nao transitoria; exige auditoria"
                        save(path, ledger)
                        save(rate_path, rate.data)
                    now = clock()
                    ready_cases = [i for i, ready in queue.items() if ready <= now]
                    delay = max(
                        dispatch_delay(sent, rate.data["rpm"], now), rate.data["pause_until"] - now
                    )
                    if (
                        reason is None
                        and ready_cases
                        and len(futures) < rate.concurrent
                        and delay <= 0
                    ):
                        # Retries vencidos primeiro; nunca duas chamadas do mesmo caso em voo.
                        index = min(ready_cases, key=lambda i: (not bool(history.get(i)), i))
                        row = rows[index]
                        reserve = reservation(target, row)
                        if held_cost(ledger) + reserve > budget or (
                            round_budget is not None
                            and held_cost(ledger, round_cases) + reserve > round_budget
                        ):
                            if not futures:
                                reason = "teto de gasto atingido antes de enviar"
                        else:
                            old = history.setdefault(index, [])
                            stem = f"{index:04d}-{len(old)}"
                            payload = request_payload(target, row)
                            entry = dict(
                                case=index,
                                doc_id=row["doc_id"],
                                arm=row["arm"],
                                reserved_usd=str(reserve),
                                status="pending",
                                at=now,
                                response_file=stem + ".response.json",
                                retry_cycle_end=cycles[index],
                                dispatch_rpm=rate.data["rpm"],
                            )
                            if amendment_sha256 is not None:
                                entry["amendment_sha256"] = amendment_sha256
                            save(output / (stem + ".request.json"), payload)
                            ledger["attempts"].append(entry)
                            old.append(entry)
                            save(path, ledger)
                            sent = [t for t in sent if t > now - 60] + [now]
                            save(window_path, sent)
                            del queue[index]
                            futures[pool.submit(invoke, transport, payload)] = entry
                    if clock() - last_log >= 5:
                        log(
                            f"{target[0]}: {completed}/{len(selected)} aceitas; "
                            f"{len(futures)} em voo; {len(queue)} na fila; "
                            f"{rate.data['rpm']}/500 RPM; estado={reason or 'executando'}"
                        )
                        last_log = clock()
                    if not futures and queue and reason is None:
                        sleep(min(0.1, max(0.01, delay, min(queue.values()) - clock())))
                except KeyboardInterrupt:
                    reason = "interrompido; recolhendo chamadas em voo"
                    if stop_event is not None:
                        stop_event.set()
        save(rate_path, rate.data)
        if reason:
            raise ValueError(reason)
        return ledger
