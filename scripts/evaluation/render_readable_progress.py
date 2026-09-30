"""Amplia os graficos do painel a partir de metricas ja calculadas; sem API ou rescoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from scripts.evaluation.update_progress_metrics import ARMS, COLORS, LABELS, MODELS

METRIC_LABELS = (
    ("bertscore", "BERTScore F1", "bertscore-f1"),
    ("rougeL", "ROUGE-L F1", "rouge-l-f1"),
    ("meteor", "METEOR", "meteor"),
)
CONTRASTS = (
    ("H1", "C2", "C1", "RAG versus fonte inteira"),
    ("H1b", "C2", "C1t", "RAG versus prefixo de igual orçamento"),
    ("H2", "C3", "C2", "Quatro exemplos fixos versus zero-shot RAG"),
    ("H3", "C4", "C3", "Aleatórios versus fixos: controle de sensibilidade"),
    ("H4", "C5", "C4", "Similares versus aleatórios"),
)
MARKERS = ("o", "s", "^")
STYLES = ("-", "--", "-.")
START = "<!-- readable-progress:start -->"
END = "<!-- readable-progress:end -->"
ROOT_START = "<!-- hypotheses-progress:start -->"
ROOT_END = "<!-- hypotheses-progress:end -->"
ROOT_SUMMARY_START = "<!-- progress-summary:start -->"
ROOT_SUMMARY_END = "<!-- progress-summary:end -->"


def full_analysis_available(summary: dict) -> bool:
    path = Path("results/progress/analysis.json")
    if summary["documents_compared"] != 1000 or not path.exists():
        return False
    data = json.loads(path.read_text())
    scores = Path("results/progress/scores.csv")
    comparisons = Path("results/progress/comparisons.json")
    return (
        data["documents"] == 1000
        and data["tests"] == 30
        and data["method_id"] == summary["method_id"]
        and data["scores_sha256"] == hashlib.sha256(scores.read_bytes()).hexdigest()
        and data["comparisons_sha256"] == hashlib.sha256(comparisons.read_bytes()).hexdigest()
    )


def focused_limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    pad = max(0.015, (high - low) * 0.25)
    return max(0, math.floor((low - pad) * 50) / 50), min(1, math.ceil((high + pad) * 50) / 50)


def metric_axis(ax, summary: dict, metric: str, label: str, *, detail: bool) -> None:
    values = []
    for i, model in enumerate(MODELS):
        series = [summary["models"][model][arm][metric] for arm in ARMS]
        values.extend(series)
        ax.plot(
            range(len(ARMS)), series, color=COLORS[i], marker=MARKERS[i],
            linestyle=STYLES[i], linewidth=2.7, markersize=7.5, label=LABELS[model],
        )
    ax.set_xlim(-0.2, len(ARMS) - 0.8)
    ax.set_ylim(*focused_limits(values))
    ax.set_xticks(range(len(ARMS)), ARMS)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_title(label, loc="left", fontsize=16 if detail else 14, fontweight="bold", pad=14)
    ax.set_ylabel("Pontuação média · eixo ampliado")
    ax.grid(axis="y", color="#d8dee8", linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")
    if detail:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)


def history_axis(ax, summary: dict, history: list[dict], *, detail: bool) -> None:
    values = []
    for i, model in enumerate(MODELS):
        series = [point["bertscore_macro"][model] for point in history]
        values.extend(series)
        ax.plot(
            [point["documents"] for point in history], series, color=COLORS[i],
            marker=MARKERS[i], linestyle=STYLES[i], linewidth=2.7, markersize=7.5,
            label=f"{LABELS[model]} · {series[-1]:.4f}",
        )
    ax.set_xlim(max(0, history[0]["documents"] - 30), summary["documents_compared"] + 30)
    ax.set_ylim(*focused_limits(values))
    ticks = [point["documents"] for point in history]
    if len(ticks) > 1 and ticks[-1] - ticks[-2] < 0.12 * (ticks[-1] - ticks[0]):
        ticks.pop(-2)
    ax.set_xticks(ticks)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_title("Evolução cumulativa · BERTScore F1", loc="left",
                 fontsize=16 if detail else 14, fontweight="bold", pad=14)
    ax.set_ylabel("Média dos seis braços · eixo ampliado")
    ax.set_xlabel("Primeiros N documentos da ordem congelada")
    ax.grid(color="#d8dee8", linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")
    if detail:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)


def save_figure(fig, out: Path, stem: str) -> None:
    for extension in ("png", "svg"):
        path = out / f"{stem}.{extension}"
        tmp = out / f"{stem}.tmp.{extension}"
        metadata = {"Software": "FINDSum"} if extension == "png" else {"Date": None}
        fig.savefig(
            tmp, dpi=165, format=extension, metadata=metadata, facecolor=fig.get_facecolor()
        )
        tmp.replace(path)


def render_figures(out: Path, summary: dict, history: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "svg.hashsalt": "findsum-readable-progress-v1", "svg.fonttype": "none",
        "font.size": 11, "axes.labelcolor": "#334155", "text.color": "#172033",
        "xtick.color": "#334155", "ytick.color": "#334155",
    })
    for metric, label, stem in METRIC_LABELS:
        fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="#f8fafc")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.79)
        metric_axis(ax, summary, metric, label, detail=True)
        fig.text(
            0.08, 0.035,
            "Mesmos documentos nos três modelos · escala vertical ampliada; "
            "valores exatos na tabela abaixo.",
            fontsize=10, color="#475569",
        )
        save_figure(fig, out, stem)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="#f8fafc")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.79)
    history_axis(ax, summary, history, detail=True)
    fig.text(0.08, 0.035, "Pontos acumulados, não lotes independentes · escala vertical ampliada.",
             fontsize=10, color="#475569")
    save_figure(fig, out, "evolucao")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), facecolor="#f8fafc")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.81, wspace=0.23, hspace=0.36)
    for ax, (metric, label, _) in zip(axes.flat[:3], METRIC_LABELS, strict=True):
        metric_axis(ax, summary, metric, label, detail=False)
    history_axis(axes[1, 1], summary, history, detail=False)
    fig.suptitle("FINDSum · RAG + few-shot", x=0.08, y=0.97, ha="left", fontsize=25,
                 fontweight="bold")
    fig.text(0.08, 0.91,
             f"{summary['documents_compared']}/{summary['expected_documents']} documentos · "
             f"{summary['documents_compared'] * 18:,} respostas · médias descritivas".replace(
                 ",", "."
             ),
             fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.91),
               ncol=3, frameon=False)
    fig.text(0.08, 0.045,
             "Eixos ampliados e identificados em cada painel. Evolução: pontos acumulados; "
             "testes confirmatórios na tabela separada aos 1.000 documentos.",
             fontsize=11, color="#475569")
    save_figure(fig, out, "dashboard")
    plt.close(fig)


def hypotheses_markdown(summary: dict) -> str:
    lines = [
        "## Hipóteses e contrastes observados",
        "",
        "Cada contraste usa os **mesmos documentos** em dois braços. Δ positivo significa "
        "maior similaridade média com a referência no primeiro braço; Δ negativo, menor. "
        "As diferenças abaixo são pontos da escala 0 a 1, não percentuais de acerto.",
        "",
        "| Hipótese | Comparação | Pergunta |",
        "|---|---|---|",
    ]
    for name, first, second, question in CONTRASTS:
        lines.append(f"| {name} | {first} - {second} | {question} |")
    lines += [
        "",
        "H3 é um **controle de sensibilidade** à escolha dos exemplos: uma diferença "
        "não significativa ao final não provaria equivalência. As métricas primárias "
        "são BERTScore F1 e ROUGE-L F1; METEOR e BERTScore precisão/recall são descritivos.",
        "",
        f"**Recorte atual: {summary['documents_compared']} documentos por modelo.** "
        "Contrastes entre médias dos braços, arredondados a quatro casas:",
        "",
        "| Modelo | Hipótese | Δ BERTScore F1 | Δ ROUGE-L F1 |",
        "|---|---|---:|---:|",
    ]
    for model in MODELS:
        for name, first, second, _ in CONTRASTS:
            left, right = summary["models"][model][first], summary["models"][model][second]
            bert = left["bertscore"] - right["bertscore"]
            rouge = left["rougeL"] - right["rougeL"]
            bert_text = f"{bert:+.4f}".replace(".", ",")
            rouge_text = f"{rouge:+.4f}".replace(".", ",")
            lines.append(f"| {LABELS[model]} | {name} | {bert_text} | {rouge_text} |")
    lines += [
        "",
        "Nos três modelos, **H1 é negativa** e **H1b é positiva**: o RAG ficou abaixo da "
        "fonte inteira, mas acima do prefixo com o mesmo orçamento de contexto. "
        "**H2 é negativa**: quatro exemplos fixos reduziram as duas métricas primárias "
        "frente ao RAG sem exemplos. **H3 e H4 são positivas**, com ganho menor em H4. "
        "Isso descreve este recorte e não estabelece eficácia causal ou qualidade factual.",
        "",
        ("**Este painel apresenta diferenças descritivas.** Os testes completos estão "
         "na tabela de comparações, calculados a partir do CSV versionado. "
         if full_analysis_available(summary)
         else "**Nenhuma hipótese foi confirmada ou refutada neste recorte parcial.** "
         "O teste predefinido será executado aos 1.000 documentos completos. ")
        + "O plano usa Wilcoxon bilateral emparelhado e Holm sobre dez testes por modelo. "
        "As saídas `length` permanecem na análise e "
        "podem afetar as médias. Os contrastes entre modelos não isolam arquitetura, "
        "tokenizador ou provedor.",
    ]
    return "\n".join(lines)


def update_readme(out: Path, summary: dict) -> None:
    path = out / "README.md"
    content = path.read_text()
    content = re.sub(rf"{re.escape(START)}.*?{re.escape(END)}\n*", "", content, flags=re.S)
    charts = "\n".join([
        START,
        "",
        "**Gráficos ampliados:** os quatro painéis têm eixos verticais focados na faixa "
        "observada e limites visíveis. Abra os SVGs para ampliar sem perda de nitidez.",
        "",
        "| Métrica | Figura detalhada | Versão vetorial |",
        "|---|---|---|",
        "| BERTScore F1 | [PNG](bertscore-f1.png) | [SVG](bertscore-f1.svg) |",
        "| ROUGE-L F1 | [PNG](rouge-l-f1.png) | [SVG](rouge-l-f1.svg) |",
        "| METEOR | [PNG](meteor.png) | [SVG](meteor.svg) |",
        "| Evolução cumulativa | [PNG](evolucao.png) | [SVG](evolucao.svg) |",
        "",
        "![BERTScore F1 ampliado](bertscore-f1.png)",
        "",
        "![ROUGE-L F1 ampliado](rouge-l-f1.png)",
        "",
        "![METEOR ampliado](meteor.png)",
        "",
        "![Evolução cumulativa ampliada](evolucao.png)",
        "",
        hypotheses_markdown(summary),
        "",
        END,
    ])
    marker = "![Painel das métricas](dashboard.png)"
    if content.count(marker) != 1:
        raise ValueError("README do painel sem marcador unico da figura")
    content = content.replace(marker, marker + "\n\n" + charts, 1)
    content = content.replace(
        "Todos os eixos partem de zero; seus limites estão explícitos no painel.",
        "Os eixos ampliados e seus limites estão explícitos em cada figura.",
    )
    path.write_text(content)


def update_root_readme(summary: dict) -> None:
    path = Path("README.md")
    content = path.read_text()
    pattern = rf"{re.escape(ROOT_START)}.*?{re.escape(ROOT_END)}"
    if len(re.findall(pattern, content, flags=re.S)) != 1:
        raise ValueError("README principal sem marcadores unicos das hipoteses")
    block = ROOT_START + "\n\n" + hypotheses_markdown(summary) + "\n\n" + ROOT_END
    content = re.sub(pattern, lambda _: block, content, flags=re.S)
    summary_pattern = rf"{re.escape(ROOT_SUMMARY_START)}.*?{re.escape(ROOT_SUMMARY_END)}"
    if len(re.findall(summary_pattern, content, flags=re.S)) != 1:
        raise ValueError("README principal sem marcadores unicos do progresso")
    n = summary["documents_compared"]
    docs_label = f"{n:,}".replace(",", ".")
    scores_label = f"{n * 18:,}".replace(",", ".")
    summary_block = (
        ROOT_SUMMARY_START + "\n"
        f"**{docs_label} documentos concluídos nos três modelos: "
        f"{scores_label} respostas e scores.** "
        "O painel apresenta as médias descritivas por braço e modelo. "
        + (
            "[Testes das hipóteses na coorte completa](results/progress/comparisons.csv).\n"
            if full_analysis_available(summary)
            else "Ainda não há teste confirmatório das hipóteses: "
            "ele depende da coorte final completa.\n"
        )
        + ROOT_SUMMARY_END
    )
    path.write_text(re.sub(summary_pattern, lambda _: summary_block, content, flags=re.S))


def render_readable_progress(out: Path = Path("results/progress")) -> None:
    summary = json.loads((out / "summary.json").read_text())
    history = json.loads((out / "history.json").read_text())
    if not summary["documents_compared"] or not history:
        return
    if history[-1]["documents"] != summary["documents_compared"]:
        raise ValueError("historico e resumo do painel discordam")
    render_figures(out, summary, history)
    update_readme(out, summary)
    update_root_readme(summary)


if __name__ == "__main__":
    render_readable_progress()
