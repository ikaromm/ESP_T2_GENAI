"""Calibra o dev via usage da API, sem atribuir um tokenizer local ao Qwen.

As sondas geram no maximo um token, descartado como resumo. Contexto usa o
incremento de tokens em um envelope fixo de mensagem; prompt usa usage integral.
Nao equivale a afirmar acesso aos IDs nativos dos tokens do contexto.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import time
from pathlib import Path

from findsum_rag.config import ExperimentConfig
from findsum_rag.examples import Example
from findsum_rag.openrouter import OpenRouterFreeClient, OpenRouterHTTPError
from findsum_rag.progress import activity, log
from findsum_rag.prompts import build_prompt
from run_prepared_paid import MAX_INPUT, money
from screen_openrouter import save

MODEL, PROVIDER, RESPONSE_PROVIDER = "qwen/qwen3.7-flash", "alibaba", "Alibaba"


def probe_bounds(messages):
    upper = sum(len(m["content"].encode()) for m in messages) + 1024
    prices = (
        ("0.03", "0.13")
        if upper < 32000
        else (("0.1", "0.4") if upper < 256000 else ("0.2", "0.8"))
    )
    return upper, prices


def probe_payload(messages):
    _, prices = probe_bounds(messages)
    return {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "reasoning": {"enabled": False},
        "provider": {
            "only": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {"prompt": float(prices[0]), "completion": float(prices[1])},
        },
    }


def probe_key(messages):
    return hashlib.sha256(json.dumps(probe_payload(messages), sort_keys=True).encode()).hexdigest()


class RemoteCounter:
    def __init__(self, folder, budget, transport, *, sleep=time.sleep):
        self.folder, self.budget, self.transport, self.sleep = folder, budget, transport, sleep
        folder.mkdir(parents=True, exist_ok=True)
        from screen_openrouter import RequestWindow

        self.limiter = RequestWindow(folder / "request-window.json")
        self.ledger_path = folder / "ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []

    def count(self, messages):
        key = probe_key(messages)
        previous = [r for r in self.ledger if r["key"] == key]
        if previous and previous[-1]["status"] == "accepted":
            return previous[-1]["prompt_tokens"]
        if previous and previous[-1]["status"] not in {"http429", "http_retryable"}:
            raise ValueError("sonda pendente/rejeitada exige auditoria; nao repetir")
        if len(previous) >= 6:
            raise ValueError("seis 429 na mesma sonda; circuito aberto")
        # Limite conservador por bytes UTF-8 mais margem para o envelope do chat.
        upper, prices = probe_bounds(messages)
        if upper + 1 >= 983616:
            raise ValueError("sonda excede limite conservador de entrada")
        reserve = (money(upper) * money(prices[0]) + money(prices[1])) / 1000000
        for attempt in range(len(previous), 6):
            if (
                sum(
                    money(
                        r["reported_cost_usd"] if r["status"] == "accepted" else r["reserved_usd"]
                    )
                    for r in self.ledger
                )
                + reserve
                > self.budget
            ):
                raise ValueError("teto de calibracao atingido antes de enviar")
            with activity(f"Sonda Qwen {len(self.ledger) + 1}: controle de frequencia"):
                self.limiter.acquire()
            stem = f"{key}-{attempt}"
            entry = {
                "key": key,
                "status": "pending",
                "reserved_usd": str(reserve),
                "request_file": stem + ".request.json",
                "response_file": stem + ".response.json",
            }
            save(self.folder / entry["request_file"], probe_payload(messages))
            self.ledger.append(entry)
            save(self.ledger_path, self.ledger)
            try:
                with activity(f"Sonda Qwen {len(self.ledger)}: aguardando API"):
                    result = self.transport._request("/chat/completions", probe_payload(messages))
                save(self.folder / entry["response_file"], result)
                usage = result["usage"]
                entry["reported_cost_usd"] = usage.get("cost")
                if (
                    result.get("model") != MODEL
                    or result.get("provider") != RESPONSE_PROVIDER
                    or money(usage["cost"]) > reserve
                    or not isinstance(usage.get("prompt_tokens"), int)
                    or not 0 < usage["prompt_tokens"] <= upper
                    or not 0 <= usage.get("completion_tokens", -1) <= 1
                    or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                ):
                    raise ValueError("contrato da sonda invalido")
            except OpenRouterHTTPError as exc:
                entry.update(
                    status=(
                        "http429"
                        if exc.status == 429
                        else "http_retryable"
                        if exc.status in {408, 500, 502, 503, 504}
                        else "rejected"
                    ),
                    error=exc.details,
                )
                save(self.ledger_path, self.ledger)
                delay = exc.retry_delay(2**attempt)
                if exc.status not in {408, 429, 500, 502, 503, 504} or attempt == 5 or delay > 60:
                    raise
                log(f"Sonda Qwen HTTP {exc.status}: retry em {delay:.1f}s")
                self.sleep(delay)
                continue
            except Exception:
                entry["status"] = "audit_required"
                save(self.ledger_path, self.ledger)
                raise
            entry.update(status="accepted", prompt_tokens=usage["prompt_tokens"])
            save(self.ledger_path, self.ledger)
            log(f"Sonda aceita: {usage['prompt_tokens']} tokens; custo US$ {usage['cost']}")
            return usage["prompt_tokens"]
        raise ValueError("calibracao nao concluida")

    @staticmethod
    def context_messages(text):
        return [{"role": "user", "content": f"CONTEXT_BEGIN\n{text}\nCONTEXT_END"}]

    def context_count(self, text):
        return self.count(self.context_messages(text)) - self.count(self.context_messages(""))


def safe_character_prefix(text, end):
    result = text[:end]
    start = text.rfind("\n", 0, end) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    # Mesma regra dos tokenizers locais: so blocos explicitamente serializados.
    header = r"^(?:\[\d+\] )?\[tabela "
    boundary = text.rfind("\n\n", 0, start)
    block_start = boundary + 2 if boundary >= 0 else 0
    table_line = bool(
        re.match(header, text[start:line_end]) or re.search(header, text[block_start:start], re.M)
    )
    if end < line_end and table_line:
        result = text[:start].rstrip()
    if result and re.match(header, result.splitlines()[-1]):
        result = result.rsplit("\n", 1)[0].rstrip() if "\n" in result else ""
    return result


def fit_prefix(text, limit, count):
    lo, hi, best, best_count = 0, len(text), "", 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = safe_character_prefix(text, mid)
        n = count(candidate)
        if n <= limit:
            if len(candidate) > len(best):
                best, best_count = candidate, n
            lo = mid + 1
        else:
            hi = mid - 1
    if not best:
        raise ValueError("nenhum prefixo nao vazio no orcamento")
    return best, best_count


def match_contexts(source, retrieved, budget, context_count, prompt_count):
    ceiling = budget
    for _ in range(64):
        right, nr = fit_prefix(retrieved, ceiling, context_count)
        left, nl = fit_prefix(source, nr, context_count)
        if nl == nr and prompt_count(left) == prompt_count(right):
            return left, right, nl
        ceiling = min(nl, nr) - 1
        if ceiling <= 0:
            break
    raise ValueError("nao foi possivel igualar contexto e prompt pelas contagens da API")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-usd", required=True, type=money)
    parser.add_argument("--allow-eval", action="store_true")
    parser.add_argument("--execute", action="store_true", help="autoriza sondas pagas")
    args = parser.parse_args()
    from findsum_rag.full_lock import verify_hashes_parallel

    with activity("Qwen: conferindo insumos; nenhuma API nesta etapa"):
        verify_hashes_parallel(
            {
                args.prepared / path: sha
                for path, sha in json.loads((args.prepared / "sha256.json").read_text()).items()
            }
        )
    cfg = ExperimentConfig.from_yaml(args.prepared / "config.yaml")
    from full_common import selected_records

    split, selected = selected_records(args.prepared)
    if split == "eval" and not args.allow_eval:
        raise ValueError("eval exige --allow-eval explicito")
    if len(selected) != cfg.data.n_eval_docs:
        raise ValueError("selecao difere da configuracao")
    if split == "eval":
        from findsum_rag.full_lock import verify_full_lock

        frozen = verify_full_lock(prepared=args.prepared)
        local_report = json.loads((args.prepared / "report.json").read_text())
        if local_report.get("models", {}).get("ling-free", {}).get("status") != "preflight_passed":
            raise ValueError("exige preparacao comum Ling integral validada")
        if args.budget_usd > money(frozen["budgets_usd"]["qwen_calibration"]):
            raise ValueError("orcamento excede teto congelado de calibracao")
    if not args.execute:
        print("Calibracao nao executada; sondas exigem --execute e orcamento.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".prepare.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("outra preparacao usa esta pasta") from None
    common = json.loads((args.prepared / "documents.json").read_text())
    examples = {
        r["doc_id"]: Example(**r) for r in json.loads((args.prepared / "examples.json").read_text())
    }
    for filename in ["selection.json", "documents.json", "examples.json", "config.yaml"]:
        source = (args.prepared / filename).read_bytes()
        dest = args.output / filename
        if dest.exists() and dest.read_bytes() != source:
            raise ValueError("insumos da retomada mudaram")
        dest.write_bytes(source)
    if split == "eval":
        save(args.output / "full-lock.json", frozen)
    folder = args.output / "qwen37"
    from run_experiment_openrouter import load_key

    counter = RemoteCounter(
        folder / "calibration",
        args.budget_usd,
        OpenRouterFreeClient(load_key(), timeout=240),
    )
    rows = []
    for doc in common:

        def prompt(arm, context, doc=doc):
            return build_prompt(
                task=cfg.data.task,
                arm=arm,
                context_chunks=[],
                context_text=context,
                examples=[examples[i] for i in doc["example_ids"][arm.id]],
                example_max_words=cfg.data.example_max_words,
            )

        zero = cfg.arm("C2")
        retrieved = "\n\n".join(
            f"[{i + 1}] {c['text']}" for i, c in enumerate(doc["retrieved_candidates"])
        )
        left, right, n = match_contexts(
            doc["source"],
            retrieved,
            cfg.retrieval.context_max_tokens,
            counter.context_count,
            lambda text, zero=zero, prompt=prompt: counter.count(prompt(zero, text).as_messages()),
        )
        contexts = {"full": doc["source"], "truncated": left, "retrieved": right}
        for arm in cfg.arms:
            context = contexts[arm.context_mode]
            messages = prompt(arm, context).as_messages()
            native = counter.count(messages)
            if native > MAX_INPUT:
                raise ValueError("prompt Qwen acima do teto; nao gerar")
            rows.append(
                {
                    "doc_id": doc["doc_id"],
                    "max_output_tokens": cfg.generation.max_new_tokens,
                    "allow_length": True,
                    "arm": arm.id,
                    "messages": messages,
                    "context": context,
                    "example_ids": doc["example_ids"][arm.id],
                    "context_tokens": n if arm.id != "C1" else counter.context_count(context),
                    "prompt_tokens": native,
                    "token_probe_key": probe_key(messages),
                    "context_probe_key": probe_key(counter.context_messages(context)),
                    "context_count_method": "api_fixed_message_increment",
                }
            )
        save(folder / "preflight.json", rows)
        print(
            f"Qwen preflight {len(rows)}/{len(common) * 6}; probes {len(counter.ledger)}",
            flush=True,
        )
    save(
        folder / "endpoint-plan.json",
        {
            "model": MODEL,
            "provider": PROVIDER,
            "max_input_tokens": MAX_INPUT,
            "token_count_method": "api_probe_fixed_envelope",
        },
    )
    save(
        args.output / "report.json",
        {
            "scope": "preparation_" + split,
            "summary_generations": 0,
            "calibration_calls": len(counter.ledger),
            "prompts_validated": len(rows),
            "calibration_reported_cost_usd": float(
                sum(money(r.get("reported_cost_usd", 0)) for r in counter.ledger)
            ),
            "token_count_method": "api_probe_fixed_envelope",
        },
    )
    save(
        args.output / "sha256.json",
        {
            str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.rglob("*")
            if p.is_file() and p.name != "sha256.json"
        },
    )


if __name__ == "__main__":
    main()
