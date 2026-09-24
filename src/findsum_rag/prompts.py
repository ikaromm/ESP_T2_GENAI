"""Montagem dos prompts.

A instrucao e a estrutura do prompt sao IDENTICAS nas cinco configuracoes. O que
varia e apenas (a) se o contexto vem recuperado por RAG ou do inicio do
documento e (b) quais exemplos few-shot aparecem. Sem isso, diferencas de
desempenho poderiam vir da redacao do prompt em vez da tecnica investigada.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .config import ExperimentArm
from .data import Task
from .examples import Example

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. You summarize sections of annual "
    "reports (10-K filings). Follow these rules strictly:\n"
    "1. Use ONLY information present in the provided report content. Never add "
    "facts, figures or judgements of your own.\n"
    "2. Preserve every figure exactly as written in the source: amounts, "
    "percentages, dates and period labels.\n"
    "3. Write continuous prose, lower case, no headings, no bullet lists, no "
    "preamble and no closing remarks.\n"
    "4. If the content does not support a statement, omit it."
)

TASK_INSTRUCTIONS = {
    Task.ROO: (
        "Summarize the results of operations: revenues, costs, expenses, margins "
        "and the drivers of period-over-period changes."
    ),
    Task.LIQUIDITY: (
        "Summarize liquidity and capital resources: cash flows from operating, "
        "investing and financing activities, credit facilities, debt and the "
        "drivers of period-over-period changes."
    ),
}


@dataclass
class Prompt:
    """Um prompt pronto, com o rastro do que entrou nele."""

    system: str
    user: str
    n_examples: int
    n_context_chunks: int
    context_words: int

    def as_messages(self) -> list[dict[str, str]]:
        """Formato de chat esperado pelos `chat_template` do transformers."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def format_context(chunks: list[Chunk]) -> str:
    """Formata os trechos recuperados, numerados para rastreabilidade."""
    return "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))


def format_examples(examples: list[Example], max_words: int) -> str:
    """Formata as demonstracoes com o documento truncado a `max_words`."""
    blocks: list[str] = []
    for i, example in enumerate(examples, 1):
        truncated = example.truncated(max_words)
        blocks.append(
            f"### Example {i}\n"
            f"Report content:\n{truncated.document}\n\n"
            f"Summary:\n{example.summary}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    *,
    task: Task,
    arm: ExperimentArm,
    context_chunks: list[Chunk],
    examples: list[Example],
    example_max_words: int = 350,
) -> Prompt:
    """Monta o prompt de uma configuracao.

    Args:
        task: define a instrucao especifica (ROO ou Liquidity).
        arm: a configuracao experimental, usada apenas para rotular a origem do
            contexto no prompt.
        context_chunks: trechos do documento a resumir, ja selecionados.
        examples: demonstracoes few-shot (vazio em C1/C2).
        example_max_words: limite de palavras do documento de cada exemplo.
    """
    context = format_context(context_chunks)
    context_label = (
        "Report content retrieved for this report"
        if arm.use_rag
        else "Report content"
    )

    sections: list[str] = [TASK_INSTRUCTIONS[task]]
    if examples:
        sections.append(
            "Here are examples of reports and their summaries. Follow their "
            "style, level of detail and use of figures.\n\n"
            + format_examples(examples, example_max_words)
        )
    sections.append(f"### Report to summarize\n{context_label}:\n{context}")
    sections.append("Write the summary now, following the rules. Output only the summary.")

    return Prompt(
        system=SYSTEM_PROMPT,
        user="\n\n".join(sections),
        n_examples=len(examples),
        n_context_chunks=len(context_chunks),
        context_words=len(context.split()),
    )
