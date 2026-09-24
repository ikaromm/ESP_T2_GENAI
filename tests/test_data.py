"""Testes da carga e remontagem do FINDSum."""

from __future__ import annotations

from pathlib import Path

import pytest

from findsum_rag.data import (
    FindSumPaths,
    Table,
    Task,
    clean_text,
    load_documents,
    strip_table_tokens,
)


def test_task_segment_counts():
    assert Task.ROO.n_segments == 2
    assert Task.LIQUIDITY.n_segments == 3
    assert Task.ROO.dir_name == "FINDSum-ROO"
    assert Task.LIQUIDITY.dir_name == "FINDSum-Liquidity"


def test_load_documents_reassembles_segments(fake_root: Path):
    documents = load_documents(fake_root, Task.LIQUIDITY, "val")
    assert len(documents) == 2

    first = documents[0]
    assert len(first.segments) == 3
    # O resumo completo e a concatenacao ordenada das porcoes.
    assert first.summary == (
        "cash flow net cash used in operating activities was $ 50.0 million "
        "the company repaid $ 12.5 million of debt during 2016 "
        "capital expenditures totalled $ 3.4 million"
    )
    assert "net cash used in operating activities" in first.document
    assert "capital expenditures totalled" in first.document


def test_doc_id_uses_stock_and_report_id(fake_root: Path):
    documents = load_documents(fake_root, Task.LIQUIDITY, "val")
    assert documents[0].doc_id == "AAA-0000000000-16-000001"
    assert documents[0].stock_name == "AAA"
    assert documents[1].report_id == "0000000000-20-000002"


def test_table_tokens_and_referenced_tables(fake_root: Path):
    documents = load_documents(fake_root, Task.LIQUIDITY, "val")
    first = documents[0]
    # Os marcadores 1 e 2 aparecem no texto; o 0 nao.
    assert first.table_indices == [1, 2]
    referenced = first.referenced_tables()
    assert [t.index for t in referenced] == [1, 2]
    assert "total debt" in referenced[0].to_text()


def test_passages_split_on_separator(fake_root: Path):
    documents = load_documents(fake_root, Task.LIQUIDITY, "val")
    # O segundo segmento do doc0 comeca com a tag separadora, que nao deve
    # gerar um trecho vazio.
    passages = documents[0].passages
    assert all(p.strip() for p in passages)
    assert any("repaid $ 12.5 million" in p for p in passages)


def test_missing_files_raise_with_instructions(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="fetch_findsum"):
        load_documents(tmp_path, Task.LIQUIDITY, "val")


def test_invalid_split_rejected(fake_root: Path):
    with pytest.raises(ValueError, match="split invalido"):
        load_documents(fake_root, Task.LIQUIDITY, "treino")


def test_mismatched_segment_lengths_raise(fake_root: Path):
    path = FindSumPaths(fake_root).segment_file(Task.LIQUIDITY, "val", 1)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        load_documents(fake_root, Task.LIQUIDITY, "val")


def test_limit_is_respected(fake_root: Path):
    assert len(load_documents(fake_root, Task.LIQUIDITY, "val", limit=1)) == 1


def test_strip_table_tokens():
    assert strip_table_tokens("a replace_table_token_12_th b") == "a b"
    assert "replace_table_token" not in strip_table_tokens("replace_table_token_0_th x")


def test_clean_text_removes_markers_and_collapses_space():
    raw = "a  story_separator_special_tag   b replace_table_token_3_th   c"
    assert clean_text(raw) == "a b c"


def test_table_to_text_omits_empty_fields_and_truncates():
    table = Table(index=0, cells=[["revenue", "", "100.0", "2020", 1, 2], ["", "", "", "", 0, 0]])
    text = table.to_text()
    assert text == "revenue | 100.0 (2020)"

    big = Table(index=1, cells=[["r", "", str(i), "", 0, 0] for i in range(10)])
    assert "celulas omitidas" in big.to_text(max_cells=3)


def test_word_counts(fake_root: Path):
    document = load_documents(fake_root, Task.LIQUIDITY, "val")[0]
    assert document.n_words == len(document.document.split())
    assert document.n_summary_words == len(document.summary.split())
