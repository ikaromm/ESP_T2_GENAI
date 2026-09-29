"""Figura e tabela descritivas para os 320 primeiros documentos de eval.

Execute da raiz do repositório: uv run --locked python results/article320/figure.py
Nenhum teste confirmatório nem chamada de geração é executado aqui.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.hashsalt"] = "findsum-article320"

ROOT = Path.cwd()
OUT = ROOT / "results/article320"
SCORES = OUT / "scores-320.csv"
MODELS = (
    ("ling-free", "Ling Flash Fin"),
    ("qwen37", "Qwen3.7 Flash"),
    ("gemma26", "Gemma 4 26B A4B"),
)
CONTRASTS = (
    ("H1", "C2", "C1"),
    ("H1b", "C2", "C1t"),
    ("H2", "C3", "C2"),
    ("H3", "C4", "C3"),
    ("H4", "C5", "C4"),
)
METRICS = (("bertscore", "BERTScore F1"), ("rougeL", "ROUGE-L F1"))
COLORS = {"bertscore": "#116A8A", "rougeL": "#B26722"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    with SCORES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["model"]
            doc = row["doc_id"]
            arm = row["arm"]
            if arm in data[key][doc]:
                raise ValueError(f"Duplicata: {key}/{doc}/{arm}")
            data[key][doc][arm] = {metric: float(row[metric]) for metric, _ in METRICS}

    cohort = list(csv.DictReader((ROOT / "configs/full-eval-cohort.csv").open(newline="")))
    doc_ids = [row["doc_id"] for row in cohort[:320]]
    assert len(doc_ids) == len(set(doc_ids)) == 320
    assert set(data) == {model for model, _ in MODELS}
    assert all(set(data[model]) == set(doc_ids) for model, _ in MODELS)
    assert all(set(data[model][doc]) == {"C1", "C1t", "C2", "C3", "C4", "C5"}
               for model, _ in MODELS for doc in doc_ids)

    rng = np.random.default_rng(20260929)
    result = []
    for metric, _ in METRICS:
        for model, _ in MODELS:
            for hypothesis, arm_a, arm_b in CONTRASTS:
                paired = np.asarray([
                    data[model][doc][arm_a][metric] - data[model][doc][arm_b][metric]
                    for doc in doc_ids
                ])
                boot_indices = rng.integers(0, len(paired), size=(10000, len(paired)))
                boot_means = paired[boot_indices].mean(axis=1)
                lo, hi = np.quantile(boot_means, [0.025, 0.975])
                result.append({
                    "model": model,
                    "metric": metric,
                    "hypothesis": hypothesis,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "documents": len(paired),
                    "mean_delta": float(paired.mean()),
                    "bootstrap_95_low": float(lo),
                    "bootstrap_95_high": float(hi),
                    "positive_pairs": int(np.sum(paired > 0)),
                    "negative_pairs": int(np.sum(paired < 0)),
                    "ties": int(np.sum(paired == 0)),
                })
    with (OUT / "contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)

    wave_rows = []
    for model, _ in MODELS:
        for metric, _ in METRICS:
            for hypothesis, arm_a, arm_b in CONTRASTS:
                for wave, subset_ids in (("1-160", doc_ids[:160]), ("161-320", doc_ids[160:])):
                    values = [data[model][doc][arm_a][metric] - data[model][doc][arm_b][metric]
                              for doc in subset_ids]
                    wave_rows.append({
                        "model": model, "metric": metric, "hypothesis": hypothesis,
                        "wave": wave, "documents": len(values),
                        "mean_delta": float(np.mean(values)),
                    })
    with (OUT / "wave-contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wave_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(wave_rows)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), sharex=True, sharey=True)
    xlim = (-0.067, 0.067)
    for r, (metric, metric_label) in enumerate(METRICS):
        for c, (model, model_label) in enumerate(MODELS):
            ax = axes[r, c]
            subset = [row for row in result if row["metric"] == metric and row["model"] == model]
            ax.axvline(0, color="#3F4753", lw=1.4, zorder=0)
            for i, row in enumerate(subset):
                y = 4 - i
                mean = row["mean_delta"]
                lo = row["bootstrap_95_low"]
                hi = row["bootstrap_95_high"]
                ax.plot([lo, hi], [y, y], color=COLORS[metric], lw=2.2, zorder=2)
                ax.plot([lo, lo], [y - .10, y + .10], color=COLORS[metric], lw=1.5)
                ax.plot([hi, hi], [y - .10, y + .10], color=COLORS[metric], lw=1.5)
                ax.scatter([mean], [y], color=COLORS[metric], s=57, zorder=3)
                label_x = min(max(mean + (0.005 if mean >= 0 else -0.005), -0.063), 0.063)
                ax.text(label_x, y + 0.22, f"{mean:+.3f}", fontsize=8.3,
                        ha="left" if mean >= 0 else "right", color="#263241")
            ax.set_title(model_label, fontsize=12, weight="bold") if r == 0 else None
            ax.set_xlim(*xlim)
            ax.set_ylim(-.55, 4.65)
            ax.set_yticks(range(4, -1, -1))
            ax.set_yticklabels([f"{h}: {a}-{b}" for h, a, b in CONTRASTS], fontsize=10)
            ax.set_xticks(np.arange(-.06, .061, .02))
            ax.grid(axis="x", alpha=.16)
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel(metric_label, weight="bold", fontsize=12)
            if r == 1:
                ax.set_xlabel("Diferença média emparelhada", fontsize=10)
    fig.suptitle("FINDSum: cinco contrastes nos primeiros 320 documentos",
                 fontsize=17, weight="bold", y=.99)
    fig.text(.5, .925,
             "Pontos = média; barras = intervalo bootstrap de 95% por documento "
             "(10.000 reamostragens). "
             "À direita de zero, o primeiro braço teve pontuação maior.",
             ha="center", fontsize=10)
    fig.text(.5, .02,
             "Análise descritiva parcial (320/1.000). Sem teste de hipótese nesta etapa. "
             "H3 é controle de sensibilidade; resultado próximo de zero não prova equivalência.",
             ha="center", fontsize=9.3, color="#3F4753")
    fig.subplots_adjust(top=.86, bottom=.09, left=.12, right=.98, wspace=.19, hspace=.24)
    for suffix in ("png", "svg", "pdf"):
        path = OUT / f"contrasts-320.{suffix}"
        metadata = (
            {"Date": None} if suffix == "svg"
            else {"CreationDate": None, "ModDate": None} if suffix == "pdf"
            else None
        )
        fig.savefig(path, dpi=190, facecolor="white", metadata=metadata)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())
                            + "\n")
    plt.close(fig)
    print(f"Criados {len(result)} contrastes e figura em {OUT}")


if __name__ == "__main__":
    main()
