"""Testes das metricas numericas e de fidelidade.

ROUGE e BERTScore sao delegados a bibliotecas consolidadas e nao sao reexercidos
aqui; o foco esta nas metricas proprias do projeto, que sao as que sustentam as
dimensoes de preservacao numerica e fidelidade factual.
"""

from __future__ import annotations

import pytest

from findsum_rag.metrics import (
    aggregate,
    extract_numbers,
    ngram_grounding,
    numeric_grounding,
    numeric_prf,
    reference_baseline,
    score_document,
)


def values(text: str) -> list[float]:
    return [m.value for m in extract_numbers(text)]


def test_extract_plain_numbers():
    assert values("net loss of 50.0 and 12 units") == [50.0, 12.0]


def test_extract_handles_thousand_separators():
    assert values("$ 1,250,000 of debt") == [1250000.0]


def test_extract_applies_textual_scale():
    assert values("$ 50.0 million") == [50_000_000.0]
    assert values("$ 2.5 billion") == [2_500_000_000.0]


def test_scale_can_be_disabled():
    assert [m.value for m in extract_numbers("$ 50.0 million", apply_scale=False)] == [50.0]


def test_percentages_keep_face_value():
    assert values("margin of 41.2 %") == [41.2]


def test_negative_numbers():
    assert values("a change of -3.5 percent") == [-3.5]


def test_numbers_inside_words_are_ignored():
    # Identificadores como "10-k" ou "q4" nao sao quantidades do relatorio.
    assert values("filed the 10-k form") == [10.0]
    assert values("segment abc123 grew") == []


def test_scaled_and_plain_forms_match():
    prf = numeric_prf("revenue was $ 50.0 million", "revenue was $ 50,000,000")
    assert prf.precision == 1.0
    assert prf.recall == 1.0
    assert prf.f1 == 1.0


def test_numeric_prf_partial_match():
    # Gerado tem 2 numeros, 1 correto; referencia tem 2.
    prf = numeric_prf("cash was 50.0 and debt 99.0", "cash was 50.0 and debt 12.5")
    assert prf.precision == pytest.approx(0.5)
    assert prf.recall == pytest.approx(0.5)


def test_numeric_prf_without_numbers_is_zero():
    prf = numeric_prf("no figures here", "cash was 50.0")
    assert prf.precision == 0.0
    assert prf.recall == 0.0
    assert prf.f1 == 0.0


def test_numeric_grounding_detects_hallucination():
    source = "cash of 50.0 million and debt of 12.5 million"
    assert numeric_grounding("cash of 50.0 million", source) == 1.0
    # 777.0 nao existe na fonte: metade dos numeros e alucinada.
    assert numeric_grounding("cash of 50.0 million and 777.0 million", source) == pytest.approx(0.5)


def test_numeric_grounding_is_one_without_numbers():
    # Sem numeros nao ha numero inventado; a cobertura e quem penaliza a omissao.
    assert numeric_grounding("no figures at all", "cash of 50.0") == 1.0


def test_ngram_grounding_bounds():
    source = "the company repaid twelve million dollars of debt during the year"
    assert ngram_grounding("the company repaid twelve million dollars", source, n=4) == 1.0
    assert ngram_grounding("completely unrelated invented sentence here", source, n=4) == 0.0


def test_ngram_grounding_short_text_is_zero():
    # Texto menor que n nao gera n-gramas; devolver 0 evita premiar resumo vazio.
    assert ngram_grounding("two words", "two words here", n=4) == 0.0


def test_reference_baseline_reports_calibration():
    # A referencia cita 99.0, que nao esta na fonte: ancoragem numerica 0.5.
    references = ["cash was 50.0 and 99.0"]
    sources = ["cash was 50.0 in the period"]
    baseline = reference_baseline(references, sources)
    assert baseline["n_docs"] == 1.0
    assert baseline["reference_numeric_grounding_mean"] == pytest.approx(0.5)
    assert 0.0 <= baseline["reference_ngram_grounding_mean"] <= 1.0
    assert baseline["reference_n_words_mean"] == 5.0


def test_reference_baseline_empty():
    assert reference_baseline([], []) == {}


def test_reference_baseline_rejects_length_mismatch():
    with pytest.raises(ValueError, match="referencias para"):
        reference_baseline(["a"], ["a", "b"])


def test_score_document_and_aggregate():
    source = "cash flow was $ 50.0 million and debt was $ 12.5 million in 2016"
    reference = "cash flow was $ 50.0 million"
    scores = [
        score_document("d1", "cash flow was $ 50.0 million", reference, source),
        score_document("d2", "cash flow was $ 99.9 million", reference, source),
    ]
    assert scores[0].numeric.f1 == 1.0
    assert scores[0].numeric_grounding == 1.0
    assert scores[1].numeric_grounding == 0.0

    summary = aggregate(scores)
    assert summary["n_docs"] == 2.0
    assert summary["numeric_grounding_mean"] == pytest.approx(0.5)
    assert "rouge1_mean" in summary and "rougeL_median" in summary


def test_aggregate_empty():
    assert aggregate([]) == {}


def test_flat_includes_all_metric_keys():
    source = "cash of 50.0"
    flat = score_document("d", "cash of 50.0", "cash of 50.0", source).flat()
    for key in (
        "rouge1",
        "rouge2",
        "rougeL",
        "bertscore",
        "numeric_precision",
        "numeric_recall",
        "numeric_f1",
        "numeric_grounding",
        "ngram_grounding",
        "n_words",
    ):
        assert key in flat
