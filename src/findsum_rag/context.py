"""Orcamento auditavel com o tokenizer da LLM, sem cortar o prompt completo."""

from __future__ import annotations

import re
from collections.abc import Callable


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def prefix(tokenizer, text: str, budget: int, *, preserve_table_rows: bool = False) -> str:
    """Prefixo do texto original, sem inventar caracteres por decode parcial."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) <= budget:
        return text
    end = offsets[budget - 1][1] if budget else 0
    result = text[:end]
    # Retokenizar uma borda pode mudar a segmentacao BPE.
    while result and token_count(tokenizer, result) > budget:
        result = result[:-1]
    if not preserve_table_rows:
        return result
    # Olha a linha ORIGINAL inteira: o corte pode ocorrer antes do primeiro
    # separador da celula, e portanto o prefixo sozinho nao identifica a tabela.
    end = len(result)
    start = text.rfind("\n", 0, end) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    line = text[start:line_end]
    # Separadores em prosa (ex.: pagina | 37) nao identificam uma tabela.
    # Somente blocos introduzidos pelo marcador do serializador sao protegidos.
    header = r"^(?:\[\d+\] )?\[tabela "
    boundary = text.rfind("\n\n", 0, start)
    block_start = boundary + 2 if boundary >= 0 else 0
    table_line = bool(re.match(header, line) or re.search(header, text[block_start:start], re.M))
    if end < line_end and table_line:
        result = text[:start].rstrip()
    # Nao entregar um cabecalho de tabela sem nenhuma celula.
    if result and re.match(header, result.splitlines()[-1]):
        result = result.rsplit("\n", 1)[0].rstrip() if "\n" in result else ""
    return result


def matched_contexts(
    tokenizer,
    full: str,
    retrieved: str,
    budget: int,
    prompt_count: Callable[[str], int],
) -> tuple[str, str]:
    """Iguala tokens de evidencia E do prompt zero-shot C1t/C2.

    Reduz o teto comum quando a borda de uma tokenizacao nao e representavel.
    Recusa a comparacao se nao houver prefixos nao vazios compativeis.
    """
    ceiling = min(budget, token_count(tokenizer, full), token_count(tokenizer, retrieved))
    while ceiling > 0:
        a = prefix(tokenizer, full, ceiling, preserve_table_rows=True)
        b = prefix(tokenizer, retrieved, ceiling, preserve_table_rows=True)
        na, nb = token_count(tokenizer, a), token_count(tokenizer, b)
        if na == nb and na > 0 and prompt_count(a) == prompt_count(b):
            return a, b
        ceiling = min(ceiling - 1, na, nb)
    raise ValueError("nao foi possivel igualar os tokens de C1t/C2")
