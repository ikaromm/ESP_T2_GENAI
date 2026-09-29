"""Executa prompts dev ja preparados, com provedor fixo e orcamento explicito.

Sem --execute, apenas verifica os artefatos e calcula reservas. Nao prepara
novos dados, nao abre eval, nao implementa fallback entre provedores.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from findsum_rag.openrouter import OpenRouterFreeClient, OpenRouterHTTPError
from findsum_rag.progress import activity, log
from scripts.experiments.screen_openrouter import RequestWindow, save

TARGETS = {
    "ling-free": ("inclusionai/ling-3.0-flash-fin:free", "novita", "Novita", "0", "0"),
    "ling-paid-novita": (
        "inclusionai/ling-3.0-flash-fin",
        "novita",
        "Novita",
        "0.042",
        "0.1232",
    ),
    "gemma26": ("google/gemma-4-26b-a4b-it", "darkbloom", "Darkbloom", "0.042", "0.22"),
    "ling-paid": ("inclusionai/ling-3.0-flash-fin", "deepinfra/fp4", "DeepInfra", "0.06", "0.18"),
    "qwen37": ("qwen/qwen3.7-flash", "alibaba", "Alibaba", "0.1", "0.4"),
}
MAX_OUTPUT = 3072
MAX_INPUT = 49152


def requests_per_minute(target):
    """Teto operacional por executor; variantes gratuitas mantem 20 RPM."""
    return 20 if target[0].endswith(":free") else 100


def money(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("valor monetario invalido")
    return number


def output_limit(row=None):
    value = (row or {}).get("max_output_tokens", MAX_OUTPUT)
    if not isinstance(value, int) or not 0 < value <= 8192:
        raise ValueError("limite de saida invalido")
    return value


def reservation(target, row=None):
    return (MAX_INPUT * money(target[3]) + output_limit(row) * money(target[4])) / 1000000


def request_payload(target, row):
    return {
        "model": target[0],
        "messages": row["messages"],
        "max_tokens": output_limit(row),
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "reasoning": {"enabled": False},
        "provider": {
            "only": [target[1]],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {"prompt": float(target[3]), "completion": float(target[4])},
        },
    }


PAID_LING_AMENDMENT = Path("configs/ling-paid-round-161-320.json")


def paid_ling_amendment():
    raw = PAID_LING_AMENDMENT.read_bytes()
    data = json.loads(raw)
    target = TARGETS["ling-paid-novita"]
    if (
        data.get("version") != 1
        or data.get("scope") != "explicit_paid_ling_for_active_round_only"
        or (data.get("paid_model"), data.get("provider")) != target[:2]
        or data.get("free_model") != TARGETS["ling-free"][0]
        or (data.get("max_input_usd_per_million"), data.get("max_output_usd_per_million"))
        != target[3:5]
        or (data.get("first_case"), data.get("last_case_exclusive")) != (960, 1920)
        or money(data.get("budget_usd", -1)) != money("2.00")
    ):
        raise ValueError("adendo do Ling pago difere do contrato autorizado")
    return data, hashlib.sha256(raw).hexdigest()


def target_for_saved_attempt(default_target, folder, entry, row):
    if "response_file" in entry:
        request_path = folder / entry["response_file"].replace(".response.", ".request.")
    elif entry["case"] == 960 and entry.get("error", {}).get("http_status") == 404:
        request_path = folder / "0960-0.request.json"
    else:
        raise ValueError("tentativa sem caminho de request auditavel")
    request = json.loads(request_path.read_text())
    if request == request_payload(default_target, row):
        return default_target
    if default_target != TARGETS["ling-free"]:
        raise ValueError("request aceita difere do prompt congelado")
    if request.get("model") != TARGETS["ling-paid-novita"][0]:
        raise ValueError("request aceita difere do prompt congelado")
    amendment, amendment_sha = paid_ling_amendment()
    if (
        not amendment["first_case"] <= entry["case"] < amendment["last_case_exclusive"]
        or entry.get("amendment_sha256") != amendment_sha
        or request != request_payload(TARGETS["ling-paid-novita"], row)
    ):
        raise ValueError("request paga fora do adendo autorizado")
    return TARGETS["ling-paid-novita"]


def validate_response(response, target, row):
    if response.get("model") != target[0] or response.get("provider") != target[2]:
        raise ValueError("modelo/provedor inesperado")
    usage = response["usage"]
    if money(usage["cost"]) > reservation(target, row):
        raise ValueError("custo acima da reserva")
    if usage.get("prompt_tokens") != row["prompt_tokens"]:
        raise ValueError("contagem local/API diverge; exige auditoria do tokenizer/template")
    if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
        raise ValueError("reasoning inesperado")
    if not 0 < usage.get("completion_tokens", 0) <= output_limit(row):
        raise ValueError("contagem de saida invalida")
    choice = response["choices"][0]
    allowed = {"stop", "length"} if row.get("allow_length", False) else {"stop"}
    if choice.get("finish_reason") not in allowed:
        raise ValueError("saida incompleta")
    if (
        not isinstance(choice["message"].get("content"), str)
        or not choice["message"]["content"].strip()
    ):
        raise ValueError("resposta sem texto")


def load_prepared(folder, name, n_docs, *, validation_workers=4):
    log(f"{name}: conferindo integridade dos artefatos em {folder}")
    hashes = json.loads((folder / "sha256.json").read_text())
    from findsum_rag.full_lock import verify_hashes_parallel

    verify_hashes_parallel(
        {folder / rel: expected for rel, expected in hashes.items()},
        workers=validation_workers,
        label=f"artefatos {name}",
    )
    selection = json.loads((folder / "selection.json").read_text())
    manifest_path = Path("data/interim/splits-liquidity.json")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != selection["manifest_sha256"]:
        raise ValueError("manifesto alterado")
    manifest = json.loads(manifest_path.read_text())
    if "documents" in selection:
        from scripts.common.full_common import selected_records

        _, records = selected_records(folder)
        if len(records) != n_docs:
            raise ValueError("nao reduzir a coorte preparada")
    else:
        if selection["dev"] != manifest["sets"]["dev"][:50]:
            raise ValueError("selecao difere do dev50 reservado")
        records = selection["dev"][:n_docs]
    doc_ids = [r["doc_id"] for r in records]
    rows = [
        r
        for r in json.loads((folder / name / "preflight.json").read_text())
        if r["doc_id"] in doc_ids
    ]
    if len(rows) != 6 * n_docs or len({(r["doc_id"], r["arm"]) for r in rows}) != len(rows):
        raise ValueError("lote incompleto/duplicado")
    tokenizer = None
    if name != "qwen37" or (folder / name / "tokenizer").is_dir():
        from transformers import AutoTokenizer

        if name == "qwen37":
            from scripts.preparation.qwen_local_tokenizer import load_qwen_tokenizer

            tokenizer = load_qwen_tokenizer(folder / name / "tokenizer")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                folder / name / "tokenizer", local_files_only=True
            )
    accepted_probes = None
    if tokenizer is None:
        ledger = json.loads((folder / name / "calibration/ledger.json").read_text())
        accepted_probes = {r["key"]: r for r in ledger if r["status"] == "accepted"}

    def validate_tokens(row):
        if tokenizer is not None:
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            count = len(tokenizer.encode(text, add_special_tokens=False))
        else:
            count = validate_remote_probe(folder / name / "calibration", row, accepted_probes)
        if count != row["prompt_tokens"] or count > MAX_INPUT:
            raise ValueError("contagem local alterada ou overflow")

    log(f"{name}: validando {len(rows)} prompts com {validation_workers} trabalhadores")
    with ThreadPoolExecutor(max_workers=validation_workers) as pool:
        for position, _ in enumerate(pool.map(validate_tokens, rows), 1):
            if position % 100 == 0 or position == len(rows):
                log(
                    f"{name}: prompts validados {position}/{len(rows)} ({position / len(rows):.0%})"
                )
    for doc in doc_ids:
        group = {r["arm"]: r for r in rows if r["doc_id"] == doc}
        if set(group) != {"C1", "C1t", "C2", "C3", "C4", "C5"}:
            raise ValueError("bracos incompletos")
        if any(group["C1t"][f] != group["C2"][f] for f in ("context_tokens", "prompt_tokens")):
            raise ValueError("orcamentos C1t/C2 divergem")
        if len({group[a]["context"] for a in ("C2", "C3", "C4", "C5")}) != 1:
            raise ValueError("contextos RAG divergem")
    return rows


def validate_remote_probe(folder, row, accepted=None):
    """Verifica contagens medidas; nunca finge que Qwen usa tokenizer local."""
    from scripts.preparation.prepare_qwen_remote import RemoteCounter, probe_key, probe_payload

    if accepted is None:
        ledger = json.loads((folder / "ledger.json").read_text())
        accepted = {r["key"]: r for r in ledger if r["status"] == "accepted"}

    def count(messages, expected_key):
        key = probe_key(messages)
        if key != expected_key or key not in accepted:
            raise ValueError("sonda de tokens ausente ou alterada")
        entry = accepted[key]
        if json.loads((folder / entry["request_file"]).read_text()) != probe_payload(messages):
            raise ValueError("mensagens divergem da sonda")
        response = json.loads((folder / entry["response_file"]).read_text())
        if response["model"] != TARGETS["qwen37"][0] or response["provider"] != "Alibaba":
            raise ValueError("provedor da sonda diverge")
        return response["usage"]["prompt_tokens"]

    empty = RemoteCounter.context_messages("")
    contextual = count(RemoteCounter.context_messages(row["context"]), row["context_probe_key"])
    if (
        row["context_count_method"] != "api_fixed_message_increment"
        or contextual - count(empty, probe_key(empty)) != row["context_tokens"]
    ):
        raise ValueError("contagem do contexto remoto diverge")
    return count(row["messages"], row["token_probe_key"])


def pending_document_blocks(rows, free_run):
    """Contingencia retoma apenas documentos sem os seis bracos completos."""
    report = json.loads((free_run / "report.json").read_text())
    if report["models"]["ling"]["status"] not in {"blocked", "incomplete"}:
        raise ValueError("rodada gratuita precisa estar interrompida/incompleta")
    free_rows = json.loads((free_run / "ling/preflight.json").read_text())
    expected = {(r["doc_id"], r["arm"]): r for r in free_rows}
    for row in rows:
        other = expected[(row["doc_id"], row["arm"])]
        if any(other[k] != value for k, value in row.items()):
            raise ValueError("prompts da contingencia diferem da rodada gratuita")
    completed = set()
    for path in (free_run / "ling").glob("*/predictions.jsonl"):
        for line in path.read_text().splitlines():
            result = json.loads(line)
            meta = result["generation_metadata"]
            if (
                meta["model"] != "inclusionai/ling-3.0-flash-fin:free"
                or meta["provider"] != "Novita"
                or meta["usage"]["cost"] != 0
                or meta["finish_reason"] != "stop"
            ):
                raise ValueError("resultado gratuito fora do contrato")
            completed.add((result["doc_id"], result["arm"]))
    finished = {
        doc
        for doc, _ in expected
        if all((doc, arm) in completed for arm in ("C1", "C1t", "C2", "C3", "C4", "C5"))
    }
    return [row for row in rows if row["doc_id"] not in finished]


def run(
    rows,
    target,
    output,
    budget,
    identity,
    transport,
    *,
    sleep=time.sleep,
    max_new_calls=None,
    retry_reset_after=None,
    request_window=None,
    case_indices=None,
    stop_event=None,
):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("outra execucao ja usa esta rodada") from None
        return _run_locked(
            rows,
            target,
            output,
            budget,
            identity,
            transport,
            sleep=sleep,
            max_new_calls=max_new_calls,
            retry_reset_after=retry_reset_after,
            request_window=request_window,
            case_indices=case_indices,
            stop_event=stop_event,
        )


def _run_locked(
    rows,
    target,
    output,
    budget,
    identity,
    transport,
    *,
    sleep,
    max_new_calls=None,
    retry_reset_after=None,
    request_window=None,
    case_indices=None,
    stop_event=None,
):
    ledger_path = output / "ledger.json"
    if not ledger_path.exists() and any(p.name != ".run.lock" for p in output.iterdir()):
        raise ValueError("diretorio contem artefatos de outra rodada")
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {"identity": identity, "attempts": []}
    )
    if ledger["identity"] != identity:
        raise ValueError("identidade da retomada mudou")
    rpm = requests_per_minute(target)
    limiter = RequestWindow(request_window or output / "request-window.json", limit=rpm)
    calls_this_session = 0
    current_document = None
    for index, row in enumerate(rows):
        if stop_event is not None and stop_event.is_set():
            raise ValueError("execucao interrompida")
        if case_indices is not None and index not in case_indices:
            continue
        reserve = reservation(target, row)
        previous = [a for a in ledger["attempts"] if a["case"] == index]
        if previous and previous[-1]["status"] == "accepted":
            response = json.loads((output / previous[-1]["response_file"]).read_text())
            saved_target = target_for_saved_attempt(target, output, previous[-1], row)
            validate_response(response, saved_target, row)
            continue
        if max_new_calls is not None and current_document != row["doc_id"]:
            accepted_cases = {a["case"] for a in ledger["attempts"] if a["status"] == "accepted"}
            needed = sum(
                r["doc_id"] == row["doc_id"] and i not in accepted_cases for i, r in enumerate(rows)
            )
            if max_new_calls - calls_this_session < needed:
                raise ValueError("cota diaria insuficiente para o bloco; retome pendencias depois")
            current_document = row["doc_id"]
        if previous and previous[-1]["status"] not in {"http429", "http_retryable"}:
            raise ValueError("tentativa pendente ou rejeitada; auditar antes de retomar")
        retry_start = len(previous)
        retry_end = 6
        if retry_start >= 6:
            if retry_reset_after is None or time.time() - previous[-1]["at"] < retry_reset_after:
                raise ValueError("limite de seis tentativas atingido; circuito aberto")
            retry_end = retry_start + 6
        for attempt_number in range(retry_start, retry_end):
            if (
                sum(
                    money(
                        a["reported_cost_usd"] if a["status"] == "accepted" else a["reserved_usd"]
                    )
                    for a in ledger["attempts"]
                )
                + reserve
                > budget
            ):
                raise ValueError("teto de gasto atingido antes de enviar")
            if max_new_calls is not None and calls_this_session >= max_new_calls:
                raise ValueError("cota diaria consumida; retome pendencias depois")
            log(
                f"{target[0]}: caso {index + 1}/{len(rows)} "
                f"{row['doc_id']}/{row['arm']}; tentativa {attempt_number + 1}; "
                "aguardando limite de requisicoes"
            )
            with activity(f"Controle de frequencia ({rpm} requisicoes/minuto)"):
                limiter.acquire()
            if stop_event is not None and stop_event.is_set():
                raise ValueError("execucao interrompida")
            calls_this_session += 1
            stem = f"{index:04d}-{attempt_number}"
            attempt = {
                "case": index,
                "doc_id": row["doc_id"],
                "arm": row["arm"],
                "reserved_usd": str(reserve),
                "status": "pending",
                "at": time.time(),
                "response_file": stem + ".response.json",
            }
            save(output / (stem + ".request.json"), request_payload(target, row))
            ledger["attempts"].append(attempt)
            save(ledger_path, ledger)
            try:
                started = time.monotonic()
                with activity(f"API {row['doc_id']}/{row['arm']} (timeout 240s)"):
                    response = transport._request("/chat/completions", request_payload(target, row))
                attempt["elapsed_seconds"] = time.monotonic() - started
                save(output / attempt["response_file"], response)
                if "cost" in response.get("usage", {}):
                    attempt["reported_cost_usd"] = response["usage"]["cost"]
                validate_response(response, target, row)
            except OpenRouterHTTPError as exc:
                free_unavailable = (
                    target[0].endswith(":free")
                    and exc.status == 404
                    and "unavailable for free" in str(exc.details.get("message", "")).lower()
                )
                attempt.update(
                    status=(
                        "http429"
                        if exc.status == 429
                        else "http_retryable"
                        if exc.status in {408, 500, 502, 503, 504} or free_unavailable
                        else "rejected"
                    ),
                    error=exc.details,
                )
                save(ledger_path, ledger)
                if free_unavailable:
                    log("Endpoint gratuito indisponivel; retomar quando voltar ao catalogo")
                    raise
                delay = exc.retry_delay(
                    2 ** (attempt_number - (retry_start if retry_start >= 6 else 0))
                )
                log(
                    f"HTTP {exc.status}; tentativa {attempt_number + 1}/{retry_end}; "
                    f"espera indicada {delay:.1f}s"
                )
                if (
                    exc.status not in {408, 429, 500, 502, 503, 504}
                    or attempt_number == retry_end - 1
                    or delay > 60
                ):
                    raise
                with activity(f"Retry em {delay:.1f}s"):
                    sleep(delay)
                continue
            except Exception as exc:
                attempt.update(status="audit_required", error_type=type(exc).__name__)
                save(ledger_path, ledger)
                raise
            attempt["status"] = "accepted"
            save(ledger_path, ledger)
            usage = response["usage"]
            log(
                f"Aceito {index + 1}/{len(rows)} {row['doc_id']}/{row['arm']}; "
                f"{attempt['elapsed_seconds']:.1f}s; tokens "
                f"entrada={usage['prompt_tokens']} saida={usage['completion_tokens']}; "
                f"custo US$ {usage['cost']}; "
                f"fim={response['choices'][0]['finish_reason']}"
            )
            break
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", choices=TARGETS, required=True)
    parser.add_argument("--n-docs", type=int, choices=(5, 50), default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget-usd", type=money)
    parser.add_argument(
        "--pending-from",
        type=Path,
        help="Ling pago: blocos pendentes de uma rodada dev50 gratuita interrompida",
    )
    args = parser.parse_args()
    if args.execute and (not args.output or not args.budget_usd):
        parser.error("execucao exige --output e --budget-usd positivo")
    if args.pending_from and (args.model != "ling-paid" or args.n_docs != 50):
        parser.error("--pending-from exige ling-paid e --n-docs 50")
    rows = load_prepared(args.prepared, args.model, args.n_docs)
    if args.pending_from:
        rows = pending_document_blocks(rows, args.pending_from)
    target = TARGETS[args.model]
    estimate = reservation(target) * len(rows)
    identity = {
        "target": list(target),
        "max_output": MAX_OUTPUT,
        "max_input": MAX_INPUT,
        "budget_usd": str(args.budget_usd),
        "preflight_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
    }
    print(
        json.dumps(identity | {"calls": len(rows), "conservative_reserve_usd": str(estimate)}),
        flush=True,
    )
    if not args.execute or not rows:
        return
    # Usa apenas o transporte sanitizado; as protecoes de custo do cliente free nao mudam.
    transport = OpenRouterFreeClient(os.environ.get("OPEN_ROUTER_KEY", ""), timeout=240)
    if args.model == "ling-paid":
        quota = transport.quota().get("free_model_daily_requests") or {}
        if quota.get("remaining") != 0:
            raise ValueError("contingencia Ling exige esgotamento diario confirmado")
    run(rows, target, args.output, args.budget_usd, identity, transport)


if __name__ == "__main__":
    main()
