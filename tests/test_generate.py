"""Testes da camada de geracao.

O pos-processamento e testado sem modelo. O teste de fumaca que carrega uma LLM
de verdade fica marcado como `slow` e usa um modelo minusculo, apenas para
exercitar a API do transformers (chat template, truncamento, contagem de tokens)
sem depender de download grande.
"""

from __future__ import annotations

import pytest

from findsum_rag.chunking import Chunk
from findsum_rag.config import DEFAULT_ARMS, GenerationConfig
from findsum_rag.data import Task
from findsum_rag.generate import Summarizer, clean_generation
from findsum_rag.prompts import build_prompt

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


def test_do_sample_reflects_temperature():
    assert GenerationConfig(temperature=0.0).do_sample is False
    assert GenerationConfig(temperature=0.5).do_sample is True


def test_tokenizer_requires_load():
    summarizer = Summarizer(GenerationConfig(model_name=SMOKE_MODEL))
    with pytest.raises(RuntimeError, match="nao carregado"):
        _ = summarizer.tokenizer


@pytest.fixture(scope="module")
def smoke_summarizer():
    """Carrega um modelo minusculo em fp32; pula se o Hub nao estiver acessivel."""
    config = GenerationConfig(
        model_name=SMOKE_MODEL,
        load_in_4bit=False,
        dtype="float32",
        max_new_tokens=16,
        max_input_tokens=512,
    )
    summarizer = Summarizer(config)
    try:
        summarizer.load()
    except Exception as exc:
        pytest.skip(f"modelo de fumaca indisponivel: {exc}")
    return summarizer


def make_prompt(n_chunks: int = 2):
    chunks = [
        Chunk("d", i, f"net cash used in operating activities was $ {i}0.0 million in 2016")
        for i in range(n_chunks)
    ]
    arm = next(a for a in DEFAULT_ARMS if a.id == "C2")
    return build_prompt(task=Task.LIQUIDITY, arm=arm, context_chunks=chunks, examples=[])


@pytest.mark.slow
def test_smoke_generate(smoke_summarizer):
    prompt = make_prompt()
    assert smoke_summarizer.count_tokens(prompt) > 0

    generation = smoke_summarizer.generate(prompt)
    assert generation.text
    assert generation.completion_tokens > 0
    assert generation.prompt_tokens > 0
    assert generation.truncated_prompt is False
    # O pos-processamento roda dentro de generate().
    assert not generation.text.lower().startswith("summary:")


@pytest.mark.slow
def test_smoke_long_prompt_is_truncated_and_flagged(smoke_summarizer):
    # 400 trechos estouram com folga o max_input_tokens de 512.
    prompt = make_prompt(n_chunks=400)
    assert smoke_summarizer.count_tokens(prompt) > smoke_summarizer.config.max_input_tokens

    generation = smoke_summarizer.generate(prompt)
    assert generation.truncated_prompt is True
    assert generation.prompt_tokens == smoke_summarizer.config.max_input_tokens


@pytest.mark.slow
def test_smoke_greedy_is_deterministic(smoke_summarizer):
    prompt = make_prompt()
    assert smoke_summarizer.generate(prompt).text == smoke_summarizer.generate(prompt).text
