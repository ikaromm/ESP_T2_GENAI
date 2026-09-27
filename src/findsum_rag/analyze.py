"""Comparacoes emparelhadas predefinidas; contraste exploratorio opcional."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

COMPARISON_METRICS = (
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore",
)


@dataclass
class PairedComparison:
    """Resultado do teste emparelhado entre duas configuracoes numa metrica."""

    metric: str
    arm_a: str
    arm_b: str
    n_pairs: int
    mean_a: float
    mean_b: float
    delta: float
    statistic: float
    p_value: float
    p_adjusted: float | None = None
    effect_size: float | None = None
    hypothesis: str | None = None

    @property
    def favours(self) -> str:
        """Qual configuracao teve a media maior (sem juizo de significancia)."""
        if self.delta > 0:
            return self.arm_a
        if self.delta < 0:
            return self.arm_b
        return "empate"

    def significant(self, alpha: float = 0.05) -> bool:
        """Usa o p ajustado quando disponivel."""
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return p < alpha


def load_scores(path: Path | str) -> dict[str, dict[str, float]]:
    """Le um `scores.csv` e devolve `{doc_id: {metrica: valor}}`."""
    rows: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            doc_id = row.pop("doc_id")
            if doc_id in rows:
                raise ValueError(f"doc_id duplicado: {doc_id}")
            values: dict[str, float] = {}
            for key, raw in row.items():
                if raw in ("", None):
                    continue
                try:
                    values[key] = float(raw)
                except ValueError:
                    continue
            rows[doc_id] = values
    return rows


def load_run(run_dir: Path | str) -> dict[str, dict[str, dict[str, float]]]:
    """Carrega todos os `scores.csv` de uma rodada: `{arm: {doc_id: metricas}}`."""
    run = Path(run_dir)
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted(run.glob("*/scores.csv")):
        scores[path.parent.name] = load_scores(path)
    return scores


def paired_values(
    a: dict[str, dict[str, float]],
    b: dict[str, dict[str, float]],
    metric: str,
) -> tuple[list[float], list[float]]:
    """Extrai os pares (a, b) da metrica apenas dos documentos comuns.

    A ordem e a dos `doc_id` ordenados, para que o resultado nao dependa da ordem
    de leitura dos arquivos.
    """
    common = sorted(set(a) & set(b))
    xs: list[float] = []
    ys: list[float] = []
    for doc_id in common:
        if metric in a[doc_id] and metric in b[doc_id]:
            xs.append(a[doc_id][metric])
            ys.append(b[doc_id][metric])
    return xs, ys


def compare_pair(
    arm_a: str,
    arm_b: str,
    scores_a: dict[str, dict[str, float]],
    scores_b: dict[str, dict[str, float]],
    metric: str,
) -> PairedComparison | None:
    """Teste de Wilcoxon emparelhado numa metrica. `None` se nao houver dados.

    Devolve `None` quando nao ha pares ou quando todas as diferencas sao zero --
    caso em que o teste e indefinido e reportar um p-valor seria enganoso.
    """
    import statistics

    from scipy.stats import wilcoxon

    xs, ys = paired_values(scores_a, scores_b, metric)
    if len(xs) < 2:
        return None

    deltas = [x - y for x, y in zip(xs, ys, strict=True)]
    if all(d == 0 for d in deltas):
        return None

    result = wilcoxon(xs, ys, zero_method="wilcox", alternative="two-sided")

    # Tamanho de efeito nao parametrico: proporcao de documentos em que A supera
    # B, descontados os empates. Taxa descritiva de vitorias, nao Vargha-Delaney.
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    effect = wins / (wins + losses) if (wins + losses) else None

    return PairedComparison(
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        n_pairs=len(xs),
        mean_a=statistics.fmean(xs),
        mean_b=statistics.fmean(ys),
        delta=statistics.fmean(deltas),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size=effect,
    )


def holm_bonferroni(comparisons: list[PairedComparison]) -> list[PairedComparison]:
    """Preenche `p_adjusted` pelo metodo de Holm-Bonferroni, in place.

    Holm e uniformemente mais poderoso que Bonferroni simples e mantem o controle
    do erro tipo I familiar sem supor independencia entre os testes.
    """
    ordered = sorted(comparisons, key=lambda c: c.p_value)
    n = len(ordered)
    running = 0.0
    for i, comparison in enumerate(ordered):
        adjusted = min(1.0, (n - i) * comparison.p_value)
        running = max(running, adjusted)  # monotonicidade
        comparison.p_adjusted = running
    return comparisons


def compare_against(
    scores: dict[str, dict[str, dict[str, float]]],
    reference: str,
    *,
    metrics: tuple[str, ...] = COMPARISON_METRICS,
) -> list[PairedComparison]:
    """Compara a configuracao de referencia com todas as outras.

    Raises:
        KeyError: se a configuracao de referencia nao estiver na rodada.
    """
    if reference not in scores:
        raise KeyError(
            f"configuracao de referencia {reference!r} ausente; disponiveis: {sorted(scores)}"
        )

    comparisons: list[PairedComparison] = []
    for arm in sorted(scores):
        if arm == reference:
            continue
        for metric in metrics:
            result = compare_pair(reference, arm, scores[reference], scores[arm], metric)
            if result is not None:
                comparisons.append(result)
    return holm_bonferroni(comparisons)


def format_table(comparisons: list[PairedComparison], *, alpha: float = 0.05) -> str:
    """Tabela de texto com os resultados, ordenada por metrica e configuracao."""
    if not comparisons:
        return "nenhuma comparacao disponivel"

    header = (
        f"{'metrica':<18}{'ref':<5}{'vs':<5}{'n':>4}{'media_ref':>11}"
        f"{'media_vs':>10}{'delta':>9}{'p':>9}{'p_ajust':>9}{'efeito':>8}  sig"
    )
    lines = [header, "-" * len(header)]
    for c in sorted(comparisons, key=lambda c: (c.metric, c.arm_b)):
        effect = f"{c.effect_size:.2f}" if c.effect_size is not None else "  -"
        adjusted = f"{c.p_adjusted:.4f}" if c.p_adjusted is not None else "     -"
        lines.append(
            f"{c.metric:<18}{c.arm_a:<5}{c.arm_b:<5}{c.n_pairs:>4}"
            f"{c.mean_a:>11.4f}{c.mean_b:>10.4f}{c.delta:>+9.4f}"
            f"{c.p_value:>9.4f}{adjusted:>9}{effect:>8}  "
            f"{'sim' if c.significant(alpha) else 'nao'}"
        )
    return "\n".join(lines)


HYPOTHESES = (
    ("H1", "C2", "C1"),
    ("H1b", "C2", "C1t"),
    ("H2", "C3", "C2"),
    ("H3", "C4", "C3"),
    ("H4", "C5", "C4"),
)
PRIMARY_METRICS = ("bertscore", "rougeL")
ANALYSIS_PLAN = {
    "version": 3,
    "bertscore_model": "xlnet-base-cased",
    "bertscore_truncation": "reject",
    "scope": "reference_similarity_only",
    "descriptive_metrics": ("rouge1", "rouge2"),
    "hypotheses": HYPOTHESES,
    "primary_metrics": PRIMARY_METRICS,
    "correction": "Holm over all 10 tests",
    "test": "paired two-sided Wilcoxon",
    "H3": "sensitivity control; non-significance does not establish equivalence",
}


def compare_hypotheses(scores) -> list[PairedComparison]:
    """Familia predefinida completa; falhas e ausencias nao reduzem a familia."""
    import math
    import statistics

    expected = {arm for _, a, b in HYPOTHESES for arm in (a, b)}
    if not expected <= scores.keys():
        raise ValueError(f"configuracoes ausentes: {sorted(expected - scores.keys())}")
    doc_ids = set(scores["C1"])
    if len(doc_ids) < 2:
        raise ValueError("pelo menos dois documentos emparelhados sao necessarios")
    for arm in sorted(expected):
        if set(scores[arm]) != doc_ids:
            raise ValueError(
                f"{arm}: documentos diferentes de C1; nao excluir pares silenciosamente"
            )
        for doc_id in sorted(doc_ids):
            for metric in PRIMARY_METRICS:
                if metric not in scores[arm][doc_id] or not math.isfinite(
                    scores[arm][doc_id][metric]
                ):
                    raise ValueError(
                        f"{arm}/{doc_id}: metrica primaria ausente ou invalida: {metric}"
                    )
    comparisons = []
    for hypothesis, a, b in HYPOTHESES:
        for metric in PRIMARY_METRICS:
            result = compare_pair(a, b, scores[a], scores[b], metric)
            if result is None:
                xs, ys = paired_values(scores[a], scores[b], metric)
                result = PairedComparison(
                    metric, a, b, len(xs), statistics.fmean(xs), statistics.fmean(ys), 0.0, 0.0, 1.0
                )
            result.hypothesis = hypothesis
            comparisons.append(result)
    return holm_bonferroni(comparisons)
