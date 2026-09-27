"""Testes da camada de geracao.

O pos-processamento e testado sem modelo. O teste de fumaca que carrega uma LLM
de verdade fica marcado como `slow` e usa um modelo minusculo, apenas para
exercitar a API do transformers (chat template, truncamento, contagem de tokens)
sem depender de download grande.
"""

from __future__ import annotations

import pytest

from findsum_rag.config import GenerationConfig
from findsum_rag.generate import clean_generation

SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Summary:\nthe company repaid debt", "the company repaid debt"),
        ("summary: the company repaid debt", "the company repaid debt"),
        ("Here is the summary: cash flow fell", "cash flow fell"),
        ("Here is a summary of the liquidity section: cash flow fell", "cash flow fell"),
        ("  the company repaid debt  ", "the company repaid debt"),
        ("Summary:\n\nSummary:\ncash fell", "cash fell"),
    ],
)
def test_clean_generation_strips_preamble(raw: str, expected: str):
    assert clean_generation(raw) == expected


def test_clean_generation_keeps_legitimate_text():
    # "summary" no meio da frase nao e rotulo e deve permanecer.
    text = "the summary of operations shows a decline"
    assert clean_generation(text) == text


def test_clean_generation_collapses_blank_lines():
    assert clean_generation("a\n\n\n\nb") == "a\n\nb"


def test_only_openrouter_generation_is_accepted():
    assert GenerationConfig().backend == "openrouter"
    with pytest.raises(ValueError):
        GenerationConfig(backend="local")
    with pytest.raises(ValueError):
        GenerationConfig(load_in_4bit=True)
    with pytest.raises(ValueError):
        GenerationConfig(temperature=0.5)


def test_tokenizer_has_no_local_generation_backend():
    from findsum_rag.generate import PromptTokenizer

    tokenizer = PromptTokenizer(GenerationConfig())
    assert not hasattr(tokenizer, "generate") and not hasattr(tokenizer, "load")
    with pytest.raises(RuntimeError, match="nao carregado"):
        _ = tokenizer.tokenizer
