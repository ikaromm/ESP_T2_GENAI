"""Testes da comparacao estatistica entre configuracoes."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from findsum_rag.analyze import (
    compare_against,
    compare_pair,
    format_table,
    holm_bonferroni,
    load_run,
    load_scores,
    paired_values,
)


def write_scores(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(tmp_path: Path, *, n: int = 20) -> Path:
    """Rodada sintetica: C5 melhor que C1 de forma consistente, C2 igual a C5."""
    run = tmp_path / "run"
    for arm, bump in (("C1", 0.0), ("C2", 0.10), ("C5", 0.10)):
        rows = [
            {
                "doc_id": f"d{i}",
                "rouge1": 0.30 + bump + i * 0.001,
                "bertscore": 0.60 + bump,
                "numeric_f1": 0.40 + bump,
            }
            for i in range(n)
        ]
        write_scores(run / arm / "scores.csv", rows)
    return run


def test_load_scores_parses_and_skips_blanks(tmp_path: Path):
    path = tmp_path / "scores.csv"
    write_scores(
        path,
        [
            {"doc_id": "d1", "rouge1": "0.5", "bertscore": ""},
            {"doc_id": "d2", "rouge1": "0.6", "bertscore": "0.8"},
        ],
    )
    scores = load_scores(path)
    assert scores["d1"] == {"rouge1": 0.5}
    assert scores["d2"]["bertscore"] == 0.8


def test_load_run_finds_all_arms(tmp_path: Path):
    scores = load_run(make_run(tmp_path))
    assert sorted(scores) == ["C1", "C2", "C5"]
    assert len(scores["C1"]) == 20


def test_paired_values_uses_only_common_docs():
    a = {"d1": {"m": 1.0}, "d2": {"m": 2.0}, "d3": {"m": 3.0}}
    b = {"d2": {"m": 9.0}, "d3": {"m": 8.0}, "d9": {"m": 7.0}}
    xs, ys = paired_values(a, b, "m")
    assert xs == [2.0, 3.0]
    assert ys == [9.0, 8.0]


def test_paired_values_skips_missing_metric():
    a = {"d1": {"m": 1.0}, "d2": {}}
    b = {"d1": {"m": 2.0}, "d2": {"m": 3.0}}
    assert paired_values(a, b, "m") == ([1.0], [2.0])


def test_compare_pair_detects_consistent_difference(tmp_path: Path):
    scores = load_run(make_run(tmp_path))
    result = compare_pair("C5", "C1", scores["C5"], scores["C1"], "rouge1")
    assert result is not None
    assert result.n_pairs == 20
    assert result.delta == pytest.approx(0.10, abs=1e-9)
    assert result.favours == "C5"
    assert result.p_value < 0.05
    # C5 vence em todos os documentos.
    assert result.effect_size == 1.0


def test_compare_pair_returns_none_for_identical_scores(tmp_path: Path):
    scores = load_run(make_run(tmp_path))
    # C2 e C5 sao identicos em bertscore: o teste e indefinido.
    assert compare_pair("C5", "C2", scores["C5"], scores["C2"], "bertscore") is None


def test_compare_pair_returns_none_with_too_few_pairs():
    a = {"d1": {"m": 1.0}}
    b = {"d1": {"m": 2.0}}
    assert compare_pair("A", "B", a, b, "m") is None


def test_compare_pair_returns_none_for_absent_metric():
    a = {"d1": {"m": 1.0}, "d2": {"m": 2.0}}
    b = {"d1": {"m": 1.0}, "d2": {"m": 2.0}}
    assert compare_pair("A", "B", a, b, "inexistente") is None


def test_favours_reports_tie():
    a = {f"d{i}": {"m": 1.0 + (i % 2)} for i in range(10)}
    b = {f"d{i}": {"m": 2.0 - (i % 2)} for i in range(10)}
    result = compare_pair("A", "B", a, b, "m")
    assert result is not None
    assert result.delta == pytest.approx(0.0)
    assert result.favours == "empate"


def test_holm_bonferroni_is_monotonic_and_bounded(tmp_path: Path):
    scores = load_run(make_run(tmp_path))
    comparisons = compare_against(scores, "C5")
    adjusted = [c.p_adjusted for c in sorted(comparisons, key=lambda c: c.p_value)]
    assert all(p is not None for p in adjusted)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    # Nao decrescente na ordem de p crescente.
    assert adjusted == sorted(adjusted)
    # Sempre pelo menos tao conservador quanto o p bruto.
    for c in comparisons:
        assert c.p_adjusted >= c.p_value - 1e-12


def test_holm_bonferroni_single_comparison_is_unchanged(tmp_path: Path):
    scores = load_run(make_run(tmp_path))
    one = compare_pair("C5", "C1", scores["C5"], scores["C1"], "rouge1")
    assert one is not None
    holm_bonferroni([one])
    assert one.p_adjusted == pytest.approx(one.p_value)


def test_compare_against_skips_reference_itself(tmp_path: Path):
    comparisons = compare_against(load_run(make_run(tmp_path)), "C5")
    assert comparisons
    assert all(c.arm_a == "C5" for c in comparisons)
    assert all(c.arm_b != "C5" for c in comparisons)


def test_compare_against_unknown_reference(tmp_path: Path):
    with pytest.raises(KeyError, match="C9"):
        compare_against(load_run(make_run(tmp_path)), "C9")


def test_significant_uses_adjusted_p(tmp_path: Path):
    scores = load_run(make_run(tmp_path))
    result = compare_pair("C5", "C1", scores["C5"], scores["C1"], "rouge1")
    assert result is not None
    result.p_value = 0.01
    result.p_adjusted = 0.20
    assert result.significant(alpha=0.05) is False
    result.p_adjusted = None
    assert result.significant(alpha=0.05) is True


def test_format_table_contains_rows(tmp_path: Path):
    table = format_table(compare_against(load_run(make_run(tmp_path)), "C5"))
    assert "metrica" in table
    assert "rouge1" in table
    assert "C1" in table


def test_format_table_empty():
    assert "nenhuma comparacao" in format_table([])
