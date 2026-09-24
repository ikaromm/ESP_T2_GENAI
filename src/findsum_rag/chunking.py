"""Fatiamento dos relatorios em trechos recuperaveis."""

from __future__ import annotations

from dataclasses import dataclass

from .data import Document, clean_text


@dataclass(frozen=True)
class Chunk:
    """Um trecho recuperavel de um relatorio."""

    doc_id: str
    index: int
    text: str
    source: str = "text"

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Divide o texto em janelas de `size` palavras com sobreposicao.

    Raises:
        ValueError: se `size` nao for positivo ou `overlap` nao for menor que `size`.
    """
    if size <= 0:
        raise ValueError(f"size deve ser positivo, recebeu {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap deve estar em [0, size), recebeu {overlap}")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    windows: list[str] = []
    start = 0
    while start < len(words):
        windows.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
        start += step
    return windows


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 220,
    overlap: int = 40,
    use_passages: bool = True,
    include_tables: bool = True,
    max_tables: int = 12,
) -> list[Chunk]:
    """Gera os trechos de um relatorio para indexacao no RAG.

    Por padrao respeita os trechos ja delimitados pelo dataset
    (`story_separator_special_tag`), subdividindo apenas os que excedem
    `chunk_size`. Isso preserva as fronteiras de conteudo da selecao original em
    vez de cortar o texto em pontos arbitrarios.

    As tabelas citadas no texto entram como trechos proprios, para que a
    recuperacao possa trazer evidencia numerica tabular, nao so textual.
    """
    chunks: list[Chunk] = []

    units = document.passages if use_passages else [document.document]
    for unit in units:
        text = clean_text(unit)
        if not text:
            continue
        pieces = (
            [text]
            if len(text.split()) <= chunk_size
            else split_words(text, chunk_size, overlap)
        )
        for piece in pieces:
            chunks.append(Chunk(doc_id=document.doc_id, index=len(chunks), text=piece))

    if include_tables:
        for table in document.referenced_tables()[:max_tables]:
            text = table.to_text()
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    index=len(chunks),
                    text=f"[tabela {table.index}]\n{text}",
                    source="table",
                )
            )

    return chunks
