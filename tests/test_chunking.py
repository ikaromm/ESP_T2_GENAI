"""Testes do fatiamento em trechos."""

from __future__ import annotations

from pathlib import Path

import pytest

from findsum_rag.chunking import chunk_document, split_words
from findsum_rag.data import Task, load_documents


def test_split_words_without_overlap():
    assert split_words("a b c d e f", size=2, overlap=0) == ["a b", "c d", "e f"]


def test_split_words_with_overlap():
    assert split_words("a b c d e", size=3, overlap=1) == ["a b c", "c d e"]


def test_split_words_shorter_than_size():
    assert split_words("a b", size=5, overlap=0) == ["a b"]


def test_split_words_empty():
    assert split_words("   ", size=5, overlap=0) == []


def test_split_words_covers_all_words():
    text = " ".join(str(i) for i in range(23))
    joined = " ".join(split_words(text, size=5, overlap=2))
    for word in text.split():
        assert word in joined.split()


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (5, 5), (5, 6), (5, -1)])
def test_split_words_rejects_invalid_params(size: int, overlap: int):
    with pytest.raises(ValueError):
        split_words("a b c", size=size, overlap=overlap)


def test_chunk_document_uses_passages_and_tables(fake_root: Path):
    document = load_documents(fake_root, Task.LIQUIDITY, "val")[0]
    chunks = chunk_document(document, chunk_size=200, overlap=20)

    assert chunks, "documento deve gerar trechos"
    assert all(c.doc_id == document.doc_id for c in chunks)
    # Indices sequenciais, sem buracos.
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Marcadores do dataset nao vazam para o trecho.
    text_chunks = [c for c in chunks if c.source == "text"]
    assert all("replace_table_token" not in c.text for c in text_chunks)
    assert all("story_separator_special_tag" not in c.text for c in text_chunks)
    # As tabelas citadas entram como trechos proprios.
    table_chunks = [c for c in chunks if c.source == "table"]
    assert len(table_chunks) == 2
    assert table_chunks[0].text.startswith("[tabela ")


def test_chunk_document_can_skip_tables(fake_root: Path):
    document = load_documents(fake_root, Task.LIQUIDITY, "val")[0]
    chunks = chunk_document(document, include_tables=False)
    assert all(c.source == "text" for c in chunks)


def test_chunk_document_subdivides_long_passages(fake_root: Path):
    document = load_documents(fake_root, Task.LIQUIDITY, "val")[0]
    few = chunk_document(document, chunk_size=200, overlap=0, include_tables=False)
    many = chunk_document(document, chunk_size=3, overlap=0, include_tables=False)
    assert len(many) > len(few)
    assert all(c.n_words <= 3 for c in many)
