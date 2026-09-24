"""Testes da orquestracao, com LLM substituida por um dublê.

Roda as cinco configuracoes de ponta a ponta sobre o dataset sintetico, sem GPU
nem download, verificando o que o pipeline garante: contexto compartilhado entre
C2 a C5, exemplos corretos por configuracao e artefatos gravados em disco.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from findsum_rag.config import ExperimentConfig
from findsum_rag.data import Task, load_documents
from findsum_rag.generate import Generation
from findsum_rag.metrics import RougeScorer
from findsum_rag.pipeline import (
    _write_arm,
    build_example_store,
    prepare_documents,
    run_arm,
    select_context,
    truncation_report,
)
from findsum_rag.prompts import Prompt


class StubSummarizer:
    """Devolve um resumo fixo e registra os prompts recebidos."""

    def __init__(
        self,
        text: str = "cash flow was $ 50.0 million",
        *,
        truncated: bool = False,
    ) -> None:
        self.text = text
        self.truncated = truncated
        self.prompts: list[Prompt] = []

    def generate(self, prompt: Prompt) -> Generation:
        self.prompts.append(prompt)
        return Generation(
            text=self.text,
            prompt_tokens=100,
            completion_tokens=10,
            truncated_prompt=self.truncated,
        )


@pytest.fixture
def config(fake_root: Path, tmp_path: Path) -> ExperimentConfig:
    cfg = ExperimentConfig(name="teste", output_dir=tmp_path / "outputs")
    cfg.data.root = fake_root
    cfg.data.task = Task.LIQUIDITY
    cfg.data.n_eval_docs = 2
    cfg.data.n_example_docs = 2
    cfg.retrieval.top_k = 2
    cfg.retrieval.chunk_size = 20
    cfg.retrieval.chunk_overlap = 5
    return cfg


@pytest.fixture
def prepared(config: ExperimentConfig, stub_encoder):
    documents = load_documents(config.data.root, config.data.task, config.data.eval_split)
    return prepare_documents(
        documents,
        stub_encoder,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
        query_words=config.retrieval.query_words,
        include_tables=config.retrieval.include_tables,
    )


def test_prepare_documents_produces_vectors(prepared, stub_encoder):
    assert len(prepared) == 2
    for item in prepared:
        assert item.chunks
        assert item.chunk_vectors.shape == (len(item.chunks), stub_encoder.dimension)
        assert item.query_vector.shape == (stub_encoder.dimension,)
        assert "replace_table_token" not in item.source_text


def test_select_context_without_rag_keeps_document_order(prepared, config):
    arm = config.arm("C1")
    chunks = select_context(prepared[0], arm, top_k=2)
    assert [c.index for c in chunks] == [0, 1]


def test_select_context_with_rag_respects_top_k_and_order(prepared, config):
    chunks = select_context(prepared[0], config.arm("C2"), top_k=2)
    assert len(chunks) == 2
    # Mesmo escolhidos por similaridade, os trechos voltam em ordem de leitura.
    assert [c.index for c in chunks] == sorted(c.index for c in chunks)


def test_rag_and_non_rag_can_differ(prepared, config):
    plain = select_context(prepared[0], config.arm("C1"), top_k=1)
    rag = select_context(prepared[0], config.arm("C2"), top_k=1)
    assert len(plain) == len(rag) == 1


def test_example_store_excludes_empty_summaries(config):
    store = build_example_store(
        config.data.root,
        config.data.task,
        config.data.example_split,
        limit=config.data.n_example_docs,
        max_words=config.data.example_max_words,
    )
    assert len(store) == 2
    assert all(e.summary.strip() for e in store.examples)
    assert all("story_separator_special_tag" not in e.document for e in store.examples)


@pytest.mark.parametrize("arm_id", ["C1", "C2", "C3", "C4", "C5"])
def test_run_arm_end_to_end(arm_id, config, prepared, stub_encoder):
    store = build_example_store(
        config.data.root,
        config.data.task,
        config.data.example_split,
        limit=config.data.n_example_docs,
        max_words=config.data.example_max_words,
    )
    store.build_index(stub_encoder, query_words=config.retrieval.query_words)

    arm = config.arm(arm_id)
    summarizer = StubSummarizer()
    result = run_arm(
        arm,
        prepared,
        config=config,
        summarizer=summarizer,
        store=store,
        rouge=RougeScorer(),
    )

    assert len(result.scores) == len(prepared)
    assert len(result.predictions) == len(prepared)
    assert result.summary["n_docs"] == float(len(prepared))
    # Numero de exemplos no prompt coincide com a configuracao.
    for prompt in summarizer.prompts:
        assert prompt.n_examples == arm.n_examples
    # Nenhum documento e exemplo de si mesmo.
    for row in result.predictions:
        assert row["doc_id"] not in row["example_ids"]
        assert row["arm"] == arm_id


def test_fixed_and_dynamic_may_choose_different_examples(config, prepared, stub_encoder):
    store = build_example_store(
        config.data.root,
        config.data.task,
        config.data.example_split,
        limit=config.data.n_example_docs,
        max_words=config.data.example_max_words,
    )
    store.build_index(stub_encoder, query_words=config.retrieval.query_words)

    picks = {}
    for arm_id in ("C3", "C5"):
        result = run_arm(
            config.arm(arm_id),
            prepared,
            config=config,
            summarizer=StubSummarizer(),
            store=store,
            rouge=RougeScorer(),
        )
        picks[arm_id] = [tuple(r["example_ids"]) for r in result.predictions]
    # C3 usa o mesmo exemplo para todos; C5 escolhe por documento.
    assert len(set(picks["C3"])) == 1


def test_truncation_report_is_silent_when_nothing_truncated(config, prepared):
    result = run_arm(
        config.arm("C1"),
        prepared,
        config=config,
        summarizer=StubSummarizer(),
        store=None,
        rouge=RougeScorer(),
    )
    assert truncation_report(result) is None


def test_truncation_report_warns_with_counts(config, prepared):
    result = run_arm(
        config.arm("C1"),
        prepared,
        config=config,
        summarizer=StubSummarizer(truncated=True),
        store=None,
        rouge=RougeScorer(),
    )
    warning = truncation_report(result)
    assert warning is not None
    assert "C1" in warning
    assert f"{len(prepared)}/{len(prepared)}" in warning
    assert "max_input_tokens" in warning


def test_write_arm_creates_artifacts(config, prepared, tmp_path: Path):

    result = run_arm(
        config.arm("C1"),
        prepared,
        config=config,
        summarizer=StubSummarizer(),
        store=None,
        rouge=RougeScorer(),
    )
    out = tmp_path / "run"
    _write_arm(out, result)

    arm_dir = out / "C1"
    assert (arm_dir / "predictions.jsonl").exists()
    assert (arm_dir / "scores.csv").exists()
    summary = json.loads((arm_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_docs"] == float(len(prepared))

    lines = (arm_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(prepared)
    assert json.loads(lines[0])["arm"] == "C1"
