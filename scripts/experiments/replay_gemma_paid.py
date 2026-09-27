"""Replay limitado do preflight dev Gemma31, separado do cliente gratuito.

Nao prepara dados, nao executa eval e nao muda prompts. Reserva o custo maximo
de cada tentativa antes do envio, inclusive em erros de resultado desconhecido.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from decimal import Decimal
from pathlib import Path

from findsum_rag.openrouter import OpenRouterFreeClient, OpenRouterHTTPError
from scripts.experiments.screen_openrouter import RequestWindow, save

MODEL = "google/gemma-4-31b-it"
PROVIDER = "reka"
MAX_OUTPUT = 3072
CAP = Decimal("0.10")


def ceiling(tokens):
    return (Decimal(tokens) * Decimal("0.08") + MAX_OUTPUT * Decimal("0.30")) / 1000000


def payload(row):
    return {
        "model": MODEL,
        "messages": row["messages"],
        "max_tokens": MAX_OUTPUT,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "reasoning": {"enabled": False},
        "provider": {
            "only": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {"prompt": 0.08, "completion": 0.30},
        },
    }


def validate(response, row, reserved):
    if response.get("model") != MODEL or response.get("provider") != "Reka":
        raise ValueError("modelo/provedor inesperado")
    choice = response["choices"][0]
    usage = response["usage"]
    cost = Decimal(str(usage.get("cost")))
    if not cost.is_finite() or cost < 0 or cost > reserved:
        raise ValueError("custo ausente/invalido ou acima da reserva")
    if usage.get("prompt_tokens") != row["prompt_tokens"]:
        raise ValueError(
            f"tokens local/API divergem: {row['prompt_tokens']}/{usage.get('prompt_tokens')}"
        )
    if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
        raise ValueError("reasoning inesperado")
    if not 0 < usage.get("completion_tokens", 0) <= MAX_OUTPUT:
        raise ValueError("contagem de saida invalida")
    if choice.get("finish_reason") != "stop" or not choice["message"].get("content"):
        raise ValueError("saida incompleta/vazia")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source = Path("outputs/screen-5docs-20260926/gemma/preflight.json")
    out = Path("outputs/screen-gemma31-paid-5docs-20260926")
    rows = json.loads(source.read_text())
    assert len(rows) == 30 and len({r["doc_id"] for r in rows}) == 5
    assert len({(r["doc_id"], r["arm"]) for r in rows}) == 30
    for doc in {r["doc_id"] for r in rows}:
        group = {r["arm"]: r for r in rows if r["doc_id"] == doc}
        assert set(group) == {"C1", "C1t", "C2", "C3", "C4", "C5"}
        for field in ("context_tokens", "prompt_tokens"):
            assert group["C1t"][field] == group["C2"][field]
        assert len({group[a]["context"] for a in ("C2", "C3", "C4", "C5")}) == 1
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it", trust_remote_code=False)
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert len(tokenizer.encode(rendered, add_special_tokens=False)) == row["prompt_tokens"]
        assert row["prompt_tokens"] <= 16384
    identity = {
        "model": MODEL,
        "provider": PROVIDER,
        "budget_usd": str(CAP),
        "preflight_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "max_output": MAX_OUTPUT,
        "input_price_per_m": "0.08",
        "output_price_per_m": "0.30",
    }
    print(
        json.dumps(
            identity | {"maximum_30_calls_usd": str(sum(ceiling(r["prompt_tokens"]) for r in rows))}
        ),
        flush=True,
    )
    if not args.execute:
        return
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "ledger.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {"identity": identity, "attempts": []}
    )
    if ledger["identity"] != identity:
        raise ValueError("identidade da rodada mudou")
    save(out / "preflight.json", rows)
    # Reuso somente do transporte sanitizado; complete()/payload() gratuitos permanecem intactos.
    transport = OpenRouterFreeClient(os.environ.get("OPEN_ROUTER_KEY", ""), timeout=240)
    limiter = RequestWindow(out / "request-window.json")
    for i, row in enumerate(rows):
        accepted = out / f"{i:02d}.accepted.json"
        if accepted.exists():
            continue
        previous = [a for a in ledger["attempts"] if a["case"] == i]
        if previous and previous[-1]["status"] != "http429":
            raise ValueError("tentativa anterior precisa de auditoria antes de retomar")
        for retry in range(len(previous), 6):
            reserve = ceiling(row["prompt_tokens"])
            used = sum(Decimal(a["reserve_usd"]) for a in ledger["attempts"])
            if used + reserve > CAP:
                raise ValueError("orcamento conservador esgotado")
            limiter.acquire()
            stem = out / f"{i:02d}.{retry}"
            save(stem.with_suffix(stem.suffix + ".request.json"), payload(row))
            attempt = {
                "case": i,
                "doc_id": row["doc_id"],
                "arm": row["arm"],
                "reserve_usd": str(reserve),
                "status": "pending",
                "at": time.time(),
            }
            ledger["attempts"].append(attempt)
            save(ledger_path, ledger)
            started = time.monotonic()
            try:
                response = transport._request("/chat/completions", payload(row))
                save(stem.with_suffix(stem.suffix + ".response.json"), response)
                validate(response, row, reserve)
            except OpenRouterHTTPError as exc:
                attempt.update(
                    status="http429" if exc.status == 429 else "error", error=exc.details
                )
                save(ledger_path, ledger)
                delay = exc.retry_delay(2**retry)
                if exc.status != 429 or retry == 5 or delay > 60:
                    raise
                time.sleep(delay)
                continue
            except Exception as exc:
                attempt.update(status="audit_required", error_type=type(exc).__name__)
                save(ledger_path, ledger)
                raise
            attempt.update(status="accepted", cost_usd=response["usage"]["cost"])
            save(ledger_path, ledger)
            save(
                accepted,
                row
                | {
                    "prediction": response["choices"][0]["message"]["content"],
                    "usage": response["usage"],
                    "model": response["model"],
                    "provider": response["provider"],
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(
                f"{i + 1}/30 {row['doc_id']} {row['arm']} custo={response['usage']['cost']}",
                flush=True,
            )
            break
    print(
        "Concluido: 30 casos; custo USD="
        + str(sum(Decimal(str(a.get("cost_usd", 0))) for a in ledger["attempts"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
