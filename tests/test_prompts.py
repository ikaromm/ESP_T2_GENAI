"""Testes da montagem dos prompts.

O ponto central: a instrucao e a estrutura precisam ser identicas entre as
configuracoes, para que a diferenca medida venha do contexto e dos exemplos, e
nao da redacao do prompt.
"""

from __future__ import annotations

from findsum_rag.chunking import Chunk
from findsum_rag.config import DEFAULT_ARMS
from findsum_rag.data import Task
from findsum_rag.examples import Example
from findsum_rag.prompts import SYSTEM_PROMPT, TASK_INSTRUCTIONS, build_prompt

CHUNKS = [
    Chunk(doc_id="d", index=0, text="cash flow was $ 50.0 million"),
    Chunk(doc_id="d", index=1, text="debt was $ 12.5 million"),
]
EXAMPLES = [Example("e1", "outro relatorio com muitas palavras aqui", "resumo do exemplo")]


def arm(arm_id: str):
    return next(a for a in DEFAULT_ARMS if a.id == arm_id)


def test_system_prompt_is_identical_across_arms():
    prompts = [
        build_prompt(
            task=Task.LIQUIDITY,
            arm=arm(a.id),
            context_chunks=CHUNKS,
            examples=EXAMPLES if a.n_examples else [],
        )
        for a in DEFAULT_ARMS
    ]
    assert {p.system for p in prompts} == {SYSTEM_PROMPT}


def test_task_instruction_present_in_every_arm():
    for a in DEFAULT_ARMS:
        prompt = build_prompt(
            task=Task.LIQUIDITY, arm=a, context_chunks=CHUNKS, examples=[]
        )
        assert TASK_INSTRUCTIONS[Task.LIQUIDITY] in prompt.user


def test_roo_and_liquidity_use_different_instructions():
    assert TASK_INSTRUCTIONS[Task.ROO] != TASK_INSTRUCTIONS[Task.LIQUIDITY]
    roo = build_prompt(task=Task.ROO, arm=arm("C2"), context_chunks=CHUNKS, examples=[])
    assert "results of operations" in roo.user


def test_context_chunks_are_numbered():
    prompt = build_prompt(
        task=Task.LIQUIDITY, arm=arm("C2"), context_chunks=CHUNKS, examples=[]
    )
    assert "[1] cash flow was $ 50.0 million" in prompt.user
    assert "[2] debt was $ 12.5 million" in prompt.user
    assert prompt.n_context_chunks == 2


def test_no_example_block_when_no_examples():
    prompt = build_prompt(
        task=Task.LIQUIDITY, arm=arm("C1"), context_chunks=CHUNKS, examples=[]
    )
    assert "### Example" not in prompt.user
    assert prompt.n_examples == 0


def test_example_block_present_and_truncated():
    prompt = build_prompt(
        task=Task.LIQUIDITY,
        arm=arm("C5"),
        context_chunks=CHUNKS,
        examples=EXAMPLES,
        example_max_words=3,
    )
    assert "### Example 1" in prompt.user
    assert "outro relatorio com" in prompt.user
    assert "muitas palavras aqui" not in prompt.user
    assert "resumo do exemplo" in prompt.user
    assert prompt.n_examples == 1


def test_context_label_signals_rag():
    with_rag = build_prompt(
        task=Task.LIQUIDITY, arm=arm("C2"), context_chunks=CHUNKS, examples=[]
    )
    without = build_prompt(
        task=Task.LIQUIDITY, arm=arm("C1"), context_chunks=CHUNKS, examples=[]
    )
    assert "retrieved for this report" in with_rag.user
    assert "retrieved for this report" not in without.user


def test_messages_format():
    prompt = build_prompt(
        task=Task.LIQUIDITY, arm=arm("C2"), context_chunks=CHUNKS, examples=[]
    )
    messages = prompt.as_messages()
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_context_words_counted():
    prompt = build_prompt(
        task=Task.LIQUIDITY, arm=arm("C2"), context_chunks=CHUNKS, examples=[]
    )
    assert prompt.context_words > 0
