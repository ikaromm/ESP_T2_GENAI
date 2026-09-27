"""Testes da orquestracao, com LLM substituida por um dublê.

Roda as cinco configuracoes de ponta a ponta sobre o dataset sintetico, sem GPU
nem download, verificando o que o pipeline garante: contexto compartilhado entre
C2 a C5, exemplos corretos por configuracao e artefatos gravados em disco.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


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

    tokenizer = CharacterTokenizer()

    def load_tokenizer(self):
        pass

    def count_tokens(self, prompt):
        return len(prompt.system + prompt.user)

    def generate(self, prompt: Prompt) -> Generation:
        self.prompts.append(prompt)
        return Generation(
            text=self.text,
            prompt_tokens=self.count_tokens(prompt),
            completion_tokens=10,
            truncated_prompt=self.truncated,
        )


@pytest.fixture
def config(fake_root: Path, fake_manifest, tmp_path: Path) -> ExperimentConfig:
    manifest, manifest_path = fake_manifest
    manifest.sets["dev"] = manifest.sets.pop("eval")
    manifest.save(manifest_path)
    cfg = ExperimentConfig(name="teste", output_dir=tmp_path / "outputs")
    cfg.data.root = fake_root
    cfg.data.task = Task.LIQUIDITY
    cfg.data.manifest = manifest_path
    cfg.data.eval_set = "dev"
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
        include_tables=config.retrieval.include_tables,
    )


@pytest.fixture
def store(config: ExperimentConfig, fake_manifest, stub_encoder):
    manifest, _ = fake_manifest
    s = build_example_store(
        config.data.root,
        manifest,
        config.data.example_set,
    )
    s.build_index(stub_encoder)
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
    assert len(chunks) == 1
    assert chunks[0].text == item.source_text


def test_truncated_context_respects_budget_and_order(prepared, config):
    """C1t corta nos primeiros top_k, na ordem original."""
    chunks = select_context(prepared[0], config.arm("C1t"), top_k=2)
    assert chunks[0].text == prepared[0].source_text


def test_retrieval_uses_task_vector_and_keeps_relevance_order(prepared, config):
    item = prepared[0]
    item.chunk_vectors = np.zeros_like(item.chunk_vectors)
    item.chunk_vectors[0, 0] = 1
    item.chunk_vectors[-1, 1] = 1
    item.query_vector = np.eye(item.chunk_vectors.shape[1], dtype=np.float32)[0]
    item.retrieval_vector = np.eye(item.chunk_vectors.shape[1], dtype=np.float32)[1]
    chunks = select_context(prepared[0], config.arm("C2"), top_k=2)
    assert len(chunks) == 2
    # O trecho final relevante deve vir antes do inicio, inclusive sob corte.
    assert chunks[0] == item.chunks[-1]


def test_context_token_fairness_and_shared_rag(prepared, config, store):
    config.retrieval.context_max_tokens = 80
    results = {}
    for arm in config.arms:
        results[arm.id] = run_arm(
            arm,
            prepared,
            config=config,
            summarizer=StubSummarizer(),
            store=store,
            rouge=RougeScorer(),
        )
    for i in range(len(prepared)):
        rows = {k: v.predictions[i] for k, v in results.items()}
        assert rows["C1t"]["context_tokens"] == rows["C2"]["context_tokens"] == 80
        assert rows["C1t"]["prompt_tokens"] == rows["C2"]["prompt_tokens"]
        assert len({rows[k]["context"] for k in ("C2", "C3", "C4", "C5")}) == 1
        assert rows["C1"]["context"] == prepared[i].source_text


def test_overflow_fails_before_generation(prepared, config):
    config.generation.max_input_tokens = 10
    summarizer = StubSummarizer()
    with pytest.raises(ValueError, match="max_input_tokens"):
        run_arm(
            config.arm("C1"),
            prepared,
            config=config,
            summarizer=summarizer,
            store=None,
            rouge=RougeScorer(),
        )
    assert not summarizer.prompts


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
    full = build_example_store(config.data.root, manifest, "examples")
    cut = build_example_store(config.data.root, manifest, "examples", max_summary_words=3)
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


def test_truncating_generator_is_rejected(config, prepared):
    with pytest.raises(ValueError, match="truncou"):
        run_arm(
            config.arm("C1"),
            prepared,
            config=config,
            summarizer=StubSummarizer(truncated=True),
            store=None,
            rouge=RougeScorer(),
        )


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


def test_full_orchestration_and_review_export(config, stub_encoder, monkeypatch, tmp_path):
    from findsum_rag import pipeline
    from findsum_rag.factual import evidence_digest, export_review

    summarizer = StubSummarizer()
    monkeypatch.setattr(pipeline, "SentenceTransformerEncoder", lambda _: stub_encoder)
    monkeypatch.setattr("findsum_rag.openrouter.OpenRouterSummarizer", lambda *a, **k: summarizer)
    results = pipeline.run_experiment(config, with_bertscore=False)
    assert set(results) == {"C1", "C1t", "C2", "C3", "C4", "C5"}
    out = config.output_dir / config.name
    assert (out / "analysis_plan.json").exists()
    review = tmp_path / "review.json"
    export_review(out, review)
    cases = json.loads(review.read_text())
    mapping = json.loads(review.with_suffix(".map.json").read_text())
    assert len(cases) == sum(len(result.predictions) for result in results.values())
    assert all("arm" not in case and not case["segmentation_complete"] for case in cases)
    assert all(
        evidence_digest(case) == mapping[case["case_id"]]["evidence_sha256"] for case in cases
    )
    with pytest.raises(ValueError, match="nao vazio"):
        pipeline.run_experiment(config, with_bertscore=False)


def test_all_arms_preflight_before_any_generation(config, stub_encoder, monkeypatch):
    from findsum_rag import pipeline

    summarizer = StubSummarizer()
    monkeypatch.setattr(pipeline, "SentenceTransformerEncoder", lambda _: stub_encoder)
    monkeypatch.setattr("findsum_rag.openrouter.OpenRouterSummarizer", lambda *a, **k: summarizer)
    config.generation.max_input_tokens = 10
    with pytest.raises(ValueError, match="max_input_tokens"):
        pipeline.run_experiment(config, with_bertscore=False)
    assert not summarizer.prompts


def test_table_values_are_in_source_grounding(prepared):
    from findsum_rag.chunking import Chunk
    from findsum_rag.metrics import numeric_grounding

    item = prepared[0]
    item.chunks.append(Chunk(item.doc_id, 99, "debt | 987654321", source="table"))
    assert numeric_grounding("debt was 987654321", item.source_text) == 1


def test_example_store_indexes_full_document_not_prompt_excerpt(config, fake_manifest):
    manifest, _ = fake_manifest
    store = build_example_store(config.data.root, manifest, "examples")
    assert len(store.examples[0].document.split()) > 1


@pytest.mark.parametrize(
    "model",
    [
        "qwen/qwen3.8-27b:free",
        "inclusionai/ling-3.0-flash-fin:free",
        "google/gemma-4-31b-it:free",
    ],
)
def test_openrouter_six_arms_and_preflight(config, stub_encoder, monkeypatch, model):
    from findsum_rag import pipeline
    from findsum_rag.openrouter import OpenRouterFreeClient, OpenRouterSummarizer
    from findsum_rag.remote_models import FREE_ENDPOINTS

    config.generation.backend = "openrouter"
    config.generation.model_name = model
    monkeypatch.setenv("OPEN_ROUTER_KEY", "test-key")
    monkeypatch.setattr(pipeline, "SentenceTransformerEncoder", lambda _: stub_encoder)
    monkeypatch.setattr(
        OpenRouterSummarizer,
        "load_tokenizer",
        lambda self: setattr(self, "_tokenizer", CharacterTokenizer()),
    )
    monkeypatch.setattr(
        OpenRouterSummarizer,
        "render",
        lambda self, p: p.system + p.user,
    )
    calls = []

    def complete(self, messages, *, max_tokens):
        calls.append(messages)
        return {
            "id": "test",
            "model": model,
            "provider": FREE_ENDPOINTS[model].response_provider,
            "text": "cash flow was $ 50.0 million",
            "finish_reason": "stop",
            "elapsed_seconds": 0.01,
            "usage": {
                "prompt_tokens": sum(len(m["content"]) for m in messages),
                "completion_tokens": 10,
                "cost": 0,
            },
        }

    monkeypatch.setattr(OpenRouterFreeClient, "complete", complete)
    assert pipeline.run_experiment(config, preflight_only=True) == {}
    assert not calls
    config.name = "remote-generation"
    results = pipeline.run_experiment(config, with_bertscore=False)
    assert set(results) == {"C1", "C1t", "C2", "C3", "C4", "C5"}
    for result in results.values():
        assert result.predictions[0]["generation_metadata"]["seed_sent"] is False
    assert (config.output_dir / config.name / "preflight.json").exists()
    assert len(calls) == sum(len(r.predictions) for r in results.values())
