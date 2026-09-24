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
from findsum_rag.data import Task
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
from findsum_rag.splits import load_set


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
def config(fake_root: Path, fake_manifest, tmp_path: Path) -> ExperimentConfig:
    _, manifest_path = fake_manifest
    cfg = ExperimentConfig(name="teste", output_dir=tmp_path / "outputs")
    cfg.data.root = fake_root
    cfg.data.task = Task.LIQUIDITY
    cfg.data.manifest = manifest_path
    cfg.data.eval_set = "eval"
    cfg.data.example_set = "examples"
    cfg.data.n_eval_docs = None
    cfg.data.n_example_docs = None
    cfg.retrieval.top_k = 2
    cfg.retrieval.chunk_size = 20
    cfg.retrieval.chunk_overlap = 5
    return cfg


@pytest.fixture
def prepared(config: ExperimentConfig, fake_manifest, stub_encoder):
    manifest, _ = fake_manifest
    documents = load_set(config.data.root, manifest, config.data.eval_set)
    return prepare_documents(
        documents,
        stub_encoder,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
        query_words=config.retrieval.query_words,
        include_tables=config.retrieval.include_tables,
    )


@pytest.fixture
def store(config: ExperimentConfig, fake_manifest, stub_encoder):
    manifest, _ = fake_manifest
    s = build_example_store(
        config.data.root,
        manifest,
        config.data.example_set,
        max_words=config.data.example_max_words,
    )
    s.build_index(stub_encoder, query_words=config.retrieval.query_words)
    return s


def test_prepare_documents_produces_vectors(prepared, stub_encoder):
    assert prepared
    for item in prepared:
        assert item.chunks
        assert item.chunk_vectors.shape == (len(item.chunks), stub_encoder.dimension)
        assert item.query_vector.shape == (stub_encoder.dimension,)
        assert "replace_table_token" not in item.source_text


def test_full_context_uses_every_chunk(prepared, config):
    """C1 recebe o documento inteiro, nao um recorte."""
    item = prepared[0]
    chunks = select_context(item, config.arm("C1"), top_k=2)
    assert len(chunks) == len(item.chunks)
    assert [c.index for c in chunks] == [c.index for c in item.chunks]


def test_truncated_context_respects_budget_and_order(prepared, config):
    """C1t corta nos primeiros top_k, na ordem original."""
    chunks = select_context(prepared[0], config.arm("C1t"), top_k=2)
    assert len(chunks) == 2
    assert [c.index for c in chunks] == [0, 1]


def test_retrieved_context_respects_budget_and_reorders(prepared, config):
    chunks = select_context(prepared[0], config.arm("C2"), top_k=2)
    assert len(chunks) == 2
    # Escolhidos por similaridade, devolvidos em ordem de leitura.
    assert [c.index for c in chunks] == sorted(c.index for c in chunks)


def test_full_context_is_larger_than_the_limited_modes(prepared, config):
    item = prepared[0]
    full = select_context(item, config.arm("C1"), top_k=2)
    trunc = select_context(item, config.arm("C1t"), top_k=2)
    rag = select_context(item, config.arm("C2"), top_k=2)
    assert len(full) > len(trunc) == len(rag) == 2


def test_example_store_excludes_empty_summaries(store):
    assert len(store) == 1
    assert all(e.summary.strip() for e in store.examples)
    assert all("story_separator_special_tag" not in e.document for e in store.examples)


def test_example_store_uses_sec_identifiers_from_the_manifest(store, fake_manifest):
    """Os ids dos exemplos vem do manifesto, nao do fallback por linha."""
    manifest, _ = fake_manifest
    expected = {r.doc_id for r in manifest.sets["examples"]}
    assert {e.doc_id for e in store.examples} == expected
    # O fallback teria a forma "<task>-<split>-<linha>".
    assert all("-train-" not in e.doc_id for e in store.examples)


def test_example_store_can_truncate_example_summaries(config, fake_manifest):
    manifest, _ = fake_manifest
    full = build_example_store(config.data.root, manifest, "examples", max_words=350)
    cut = build_example_store(
        config.data.root, manifest, "examples", max_words=350, max_summary_words=3
    )
    assert len(cut.examples[0].summary.split()) == 3
    assert len(full.examples[0].summary.split()) > 3


@pytest.mark.parametrize("arm_id", ["C1", "C1t", "C2", "C3", "C4", "C5"])
def test_run_arm_end_to_end(arm_id, config, prepared, store):
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
    # O seletor nunca devolve mais exemplos do que existem na base: o pool
    # sintetico tem 1, entao o esperado e min(n_examples, len(store)).
    expected = min(arm.n_examples, len(store))
    for prompt in summarizer.prompts:
        assert prompt.n_examples == expected
    for row in result.predictions:
        assert row["doc_id"] not in row["example_ids"]
        assert row["arm"] == arm_id


def test_fixed_examples_are_identical_across_documents(config, prepared, store):
    result = run_arm(
        config.arm("C3"),
        prepared,
        config=config,
        summarizer=StubSummarizer(),
        store=store,
        rouge=RougeScorer(),
    )
    picks = {tuple(r["example_ids"]) for r in result.predictions}
    assert len(picks) == 1


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
