"""Metricas descritivas incrementais e painel SVG para GitHub; nenhuma geracao/API."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

from findsum_rag.data import clean_text
from findsum_rag.generate import clean_generation
from findsum_rag.metrics import (
    SUPPLEMENTARY_METRICS,
    bertscore_components,
    bertscore_token_audit,
    ensure_meteor_resources,
    meteor,
    score_document,
)
from findsum_rag.progress import activity, log

MODELS = ("ling-free", "qwen37", "gemma26")
ARMS = ("C1", "C1t", "C2", "C3", "C4", "C5")
METRICS = ("bertscore", "bertscore_precision", "bertscore_recall", "rougeL", "meteor")
LABELS = {"ling-free": "Ling Flash Fin", "qwen37": "Qwen3.7 Flash", "gemma26": "Gemma 4 26B A4B"}
COLORS = ("#059669", "#2563eb", "#9333ea")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, bytes):
        tmp.write_bytes(value)
    else:
        tmp.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def common_prefix(records, accepted):
    for i in range(len(records)):
        if any(not set(range(i * 6, (i + 1) * 6)) <= accepted[m] for m in MODELS):
            return i
    return len(records)


def aggregate(rows, count):
    result = {}
    for model in MODELS:
        result[model] = {}
        for arm in ARMS:
            group = [r for r in rows if r["model"] == model and r["arm"] == arm]
            if len(group) != count or len({r["doc_id"] for r in group}) != count:
                raise ValueError("grupos incompletos ou duplicados no painel")
            result[model][arm] = (
                {key: statistics.fmean(r[key] for r in group) for key in METRICS} if count else {}
            )
    return result


def dashboard(summary, history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"svg.hashsalt": "findsum-progress-v1", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(15, 12), facecolor="#f8fafc", layout="constrained")
    grid = fig.add_gridspec(4, 3, height_ratios=[0.75, 0.7, 2.3, 2.2])
    title = fig.add_subplot(grid[0, :])
    title.axis("off")
    n = summary["documents_compared"]
    title.text(0, 0.8, "FINDSum / RAG + few-shot", fontsize=26, weight="bold")
    title.text(
        0,
        0.4,
        f"{n}/{summary['expected_documents']} documentos comparáveis · "
        f"{n * 18:,} respostas na matriz",
        fontsize=16,
    )
    title.text(
        0, 0, "Mesmos documentos · médias descritivas · sem testes confirmatórios", fontsize=12
    )
    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(grid[1, i])
        ax.axis("off")
        p = summary["progress"][model]
        ax.text(0, 0.85, LABELS[model], color=COLORS[i], fontsize=17, weight="bold")
        ax.text(
            0,
            0.4,
            f"{p['accepted']:,}/{summary['expected_documents'] * 6:,} respostas",
            fontsize=19,
        )
        ax.text(
            0,
            0,
            f"{p['complete_documents']} docs completos · {p['length']} saídas length",
            fontsize=12,
        )
    for panel, (metric, label) in enumerate(
        [("bertscore", "BERTScore F1"), ("rougeL", "ROUGE-L F1"), ("meteor", "METEOR")]
    ):
        ax = fig.add_subplot(grid[2, panel])
        for i, model in enumerate(MODELS):
            if n:
                ax.plot(
                    ARMS,
                    [summary["models"][model][a][metric] for a in ARMS],
                    "o-",
                    color=COLORS[i],
                    label=LABELS[model],
                    linewidth=2,
                )
        ax.set_title(label, weight="bold", pad=14)
        ax.set_ylim(0, 1 if metric == "bertscore" else max(0.3, ax.get_ylim()[1]))
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
    ax = fig.add_subplot(grid[3, :])
    for i, model in enumerate(MODELS):
        ax.plot(
            [h["documents"] for h in history],
            [h["bertscore_macro"][model] for h in history],
            "o-",
            color=COLORS[i],
            label=LABELS[model],
            linewidth=2,
        )
    ax.set_title("Evolução cumulativa · BERTScore F1 médio dos seis braços", weight="bold", pad=14)
    ax.set_xlim(0, summary["expected_documents"])
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Documentos em ordem congelada · pontos acumulados, não lotes independentes")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    svg, png = io.StringIO(), io.BytesIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.savefig(png, format="png", dpi=130, metadata={"Software": "FINDSum"})
    plt.close(fig)
    return svg.getvalue(), png.getvalue()


def render(out, rows, summary, history):
    svg, png = dashboard(summary, history)
    save(out / "dashboard.svg", svg)
    save(out / "dashboard.png", png)
    stream = io.StringIO()
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save(out / "scores.csv", stream.getvalue())
    save(out / "summary.json", summary)
    save(out / "history.json", history)
    lines = [
        "# Acompanhamento do experimento FINDSum",
        "",
        "![Painel das métricas](dashboard.png)",
        "",
        f"**{summary['documents_compared']}/{summary['expected_documents']} documentos "
        f"no prefixo comum completo.** Atualização local automática pelo Bash, sem "
        f"chamadas de geração adicionais para as métricas.",
        "",
        "Somente os documentos consecutivos da ordem congelada com seis respostas "
        "aceitas em todos os modelos entram na matriz. Pendências de um modelo não "
        "alteram a base de comparação. Progresso individual aparece nos cartões.",
        "",
        "Resultados descritivos: sem p-valores ou confirmação de hipóteses. A rotina "
        "confirmatória final permanece separada. BERTScore XLNet-base-cased, camada 5, "
        "sem IDF/rescale, textos integrais; ROUGE-L F1; METEOR. Todos os eixos partem "
        "de zero; seus limites estão explícitos no painel.",
        "",
        "| Modelo | Braço | BERT P | BERT R | BERT F1 | ROUGE-L | METEOR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if summary["documents_compared"]:
        for model in MODELS:
            for arm in ARMS:
                s = summary["models"][model][arm]
                lines.append(
                    "| "
                    + LABELS[model]
                    + " | "
                    + arm
                    + " | "
                    + " | ".join(
                        f"{s[k]:.4f}"
                        for k in (
                            "bertscore_precision",
                            "bertscore_recall",
                            "bertscore",
                            "rougeL",
                            "meteor",
                        )
                    )
                    + " |"
                )
    lines += [
        "",
        "[Scores por documento](scores.csv) · [Resumo e cobertura](summary.json) · "
        "[Histórico cumulativo](history.json)",
        "",
        "C1: fonte inteira; C1t: prefixo equiparado a C2; C2: RAG sem exemplos; "
        "C3/C4/C5: mesmo RAG com quatro exemplos fixos/aleatórios/similares. Saídas "
        "`length` são mantidas. Precisão/recall/F1 são médias individuais, não F1 "
        "derivado das médias.",
        "",
        "Execute `bash rodar_rodada.sh --metrics-only` para atualizar sem geração. O "
        "script não faz commit nem push automaticamente. O cálculo usa cache local por "
        "conteúdo e método; artefatos brutos e credenciais não são exportados.",
        "",
        "Os pontos históricos são recalculados com os dados atuais e representam "
        "médias acumuladas, não réplicas independentes. Não ajustar o protocolo com "
        "base neste acompanhamento da coorte eval.",
    ]
    save(out / "README.md", "\n".join(lines) + "\n")


def update_metrics(
    records,
    states,
    *,
    out=Path("results/progress"),
    cache=Path("outputs/progress-metrics/cache.json"),
):
    from scripts.execution.run_ling_batches import output_lock

    with ExitStack() as stack:
        for state in states.values():
            stack.enter_context(output_lock(state.output))
        accepted = {m: states[m].accepted() if m in states else set() for m in MODELS}
        count = common_prefix(records, accepted)
        progress, pairs, hashes = {}, [], {}
        documents = None
        for model in MODELS:
            state = states.get(model)
            if state is None:
                progress[model] = dict(accepted=0, complete_documents=0, length=0)
                continue
            path = state.prepared / "documents.json"
            doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if documents is None:
                documents = {d["doc_id"]: d for d in json.loads(path.read_text())}
                reference_hash = doc_hash
            elif doc_hash != reference_hash:
                raise ValueError("fontes/referencias divergem entre modelos")
            lp = state.folder / "ledger.json"
            attempts = json.loads(lp.read_text())["attempts"] if lp.exists() else []
            hashes[model] = hashlib.sha256(lp.read_bytes()).hexdigest() if lp.exists() else None
            entries = {e["case"]: e for e in attempts if e["status"] == "accepted"}
            if len(entries) != len([e for e in attempts if e["status"] == "accepted"]):
                raise ValueError("respostas aceitas duplicadas")
            length = 0
            for index in sorted(accepted[model]):
                entry = entries[index]
                response = json.loads((state.folder / entry["response_file"]).read_text())
                finish = response["choices"][0]["finish_reason"]
                length += finish == "length"
                if index < count * 6:
                    pairs.append(
                        dict(
                            model=model,
                            arm=entry["arm"],
                            doc_id=entry["doc_id"],
                            prediction=clean_generation(
                                response["choices"][0]["message"]["content"]
                            ),
                            reference=clean_text(documents[entry["doc_id"]]["reference"]),
                            finish_reason=finish,
                        )
                    )
            progress[model] = dict(
                accepted=len(accepted[model]),
                length=length,
                complete_documents=sum(
                    set(range(i * 6, (i + 1) * 6)) <= accepted[model] for i in range(len(records))
                ),
            )
        method = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("src/findsum_rag/metrics.py"),
                Path("src/findsum_rag/data.py"),
                Path("src/findsum_rag/generate.py"),
                Path("uv.lock"),
                Path("scripts/evaluation/update_progress_metrics.py"),
            ]
        }
        method_id = fingerprint(dict(files=method, metrics=SUPPLEMENTARY_METRICS, version=1))
        cached = json.loads(cache.read_text()) if cache.exists() else {}
        if cached.get("method") != method_id:
            cached = dict(method=method_id, scores={})
        keys = [fingerprint(p) for p in pairs]
        missing = [i for i, k in enumerate(keys) if k not in cached["scores"]]
        log(
            f"Metricas locais: {count} docs comuns; {len(pairs)} pares; "
            f"{len(missing)} novos; sem API"
        )
        if missing:
            ensure_meteor_resources()
        for start in range(0, len(missing), 120):
            indices = missing[start : start + 120]
            texts = [pairs[i]["prediction"] for i in indices]
            refs = [pairs[i]["reference"] for i in indices]
            with activity(f"BERTScore integral: {start}/{len(missing)} novos pares"):
                audits = bertscore_token_audit(texts + refs)
                components = bertscore_components(texts, refs, batch_size=1)
            for j, (i, c) in enumerate(zip(indices, components, strict=True)):
                p = pairs[i]
                row = score_document(p["doc_id"], p["prediction"], p["reference"], "").flat()
                row.update(
                    model=p["model"],
                    arm=p["arm"],
                    bertscore=c.f1,
                    bertscore_precision=c.precision,
                    bertscore_recall=c.recall,
                    meteor=meteor(p["prediction"], p["reference"]),
                    finish_reason=p["finish_reason"],
                )
                cached["scores"][keys[i]] = dict(
                    row=row, token_audit=[audits[j], audits[len(indices) + j]]
                )
            save(cache, cached)
            log(f"Metricas: {min(start + 120, len(missing))}/{len(missing)} novos pares salvos")
        rows = [cached["scores"][k]["row"] for k in keys]
        summary = dict(
            scope="cumulative_descriptive",
            expected_documents=len(records),
            documents_compared=count,
            document_ids=[r["doc_id"] for r in records[:count]],
            progress=progress,
            models=aggregate(rows, count),
            method_id=method_id,
            method=method,
            ledger_sha256=hashes,
            hypotheses_tested=False,
            token_audit=[cached["scores"][k]["token_audit"] for k in keys],
        )
        # Pontos deterministas a cada 100 docs, mais o prefixo atual; reconstruidos do CSV.
        checkpoints = sorted(set(range(100, count + 1, 100)) | ({count} if count else set()))
        history = []
        for n in checkpoints:
            ids = {r["doc_id"] for r in records[:n]}
            subset = [r for r in rows if r["doc_id"] in ids]
            history.append(
                dict(
                    documents=n,
                    bertscore_macro={
                        m: statistics.fmean(r["bertscore"] for r in subset if r["model"] == m)
                        for m in MODELS
                    },
                )
            )
        render(out, rows, summary, history)
        log(f"Painel atualizado: {out}/README.md; {len(missing)} pares calculados")
        return summary
