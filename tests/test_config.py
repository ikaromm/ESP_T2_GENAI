"""Testes da configuracao do experimento."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from findsum_rag.config import (
    DEFAULT_ARMS,
    DataConfig,
    ExperimentArm,
    ExperimentConfig,
    RetrievalConfig,
)
from findsum_rag.data import Task


def test_default_arms_cover_all_configurations():
    assert [a.id for a in DEFAULT_ARMS] == ["C1", "C1t", "C2", "C3", "C4", "C5"]
    by_id = {a.id: a for a in DEFAULT_ARMS}

    # C1 le o documento inteiro; C1t corta no orcamento do RAG; o resto recupera.
    assert by_id["C1"].context_mode == "full"
    assert by_id["C1t"].context_mode == "truncated"
    assert all(by_id[i].context_mode == "retrieved" for i in ("C2", "C3", "C4", "C5"))

    # use_rag e derivado: so 'retrieved' conta como recuperacao.
    assert by_id["C1"].use_rag is False
    assert by_id["C1t"].use_rag is False
    assert by_id["C2"].use_rag is True
    assert by_id["C1"].uses_full_document is True
    assert by_id["C1t"].uses_full_document is False

    # As tres primeiras nao usam exemplos; as tres ultimas variam a estrategia.
    assert by_id["C1"].example_strategy == "none"
    assert by_id["C1t"].example_strategy == "none"
    assert by_id["C2"].example_strategy == "none"
    assert by_id["C3"].example_strategy == "fixed"
    assert by_id["C4"].example_strategy == "random"
    assert by_id["C5"].example_strategy == "dynamic"
    assert all(by_id[i].n_examples == 4 for i in ("C3", "C4", "C5"))


def test_arms_isolate_one_factor_at_a_time():
    """A cadeia de ablacao: cada par adjacente muda exatamente um fator."""
    by_id = {a.id: a for a in DEFAULT_ARMS}

    # C1 -> C1t: muda so o contexto (inteiro vs truncado), exemplos iguais.
    assert by_id["C1"].example_strategy == by_id["C1t"].example_strategy
    # C1t -> C2: mesmo orcamento, muda so truncar vs recuperar.
    assert by_id["C1t"].example_strategy == by_id["C2"].example_strategy
    assert by_id["C1t"].context_mode != by_id["C2"].context_mode
    # C2 -> C3: mesmo contexto, muda so a presenca de exemplos.
    assert by_id["C2"].context_mode == by_id["C3"].context_mode
    # C3 -> C4 -> C5: mesmo contexto e mesmo n, muda so a estrategia.
    for a, b in (("C3", "C4"), ("C4", "C5")):
        assert by_id[a].context_mode == by_id[b].context_mode
        assert by_id[a].n_examples == by_id[b].n_examples
        assert by_id[a].example_strategy != by_id[b].example_strategy


def test_arm_rejects_examples_with_none_strategy():
    with pytest.raises(ValidationError, match="nao aceita n_examples"):
        ExperimentArm(
            id="X", label="x", context_mode="retrieved", example_strategy="none", n_examples=2
        )


def test_arm_requires_examples_for_other_strategies():
    with pytest.raises(ValidationError, match="exige n_examples"):
        ExperimentArm(
            id="X", label="x", context_mode="retrieved", example_strategy="dynamic", n_examples=0
        )


def test_arm_rejects_unknown_strategy():
    with pytest.raises(ValidationError):
        ExperimentArm(
            id="X", label="x", context_mode="retrieved", example_strategy="magica", n_examples=1
        )


def test_arm_rejects_unknown_context_mode():
    with pytest.raises(ValidationError):
        ExperimentArm(id="X", label="x", context_mode="magico", example_strategy="none")


def test_retrieval_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        RetrievalConfig(chunk_size=100, chunk_overlap=100)


def test_data_config_rejects_same_set_for_eval_and_examples():
    with pytest.raises(ValidationError, match="vazamento"):
        DataConfig(eval_set="dev", example_set="dev")


def test_duplicate_arm_ids_rejected():
    arm = ExperimentArm(id="C1", label="a", context_mode="full", example_strategy="none")
    with pytest.raises(ValidationError, match="repetidos"):
        ExperimentConfig(arms=[arm, arm])


def test_empty_arms_rejected():
    with pytest.raises(ValidationError, match="pelo menos uma"):
        ExperimentConfig(arms=[])


def test_generation_do_sample_follows_temperature():
    config = ExperimentConfig()
    assert config.generation.temperature == 0.0
    assert config.generation.do_sample is False
    config.generation.temperature = 0.7
    assert config.generation.do_sample is True


def test_yaml_roundtrip(tmp_path: Path):
    original = ExperimentConfig(name="teste")
    original.data.task = Task.ROO
    original.retrieval.top_k = 5
    path = tmp_path / "cfg.yaml"
    original.to_yaml(path)

    loaded = ExperimentConfig.from_yaml(path)
    assert loaded.name == "teste"
    assert loaded.data.task == Task.ROO
    assert loaded.retrieval.top_k == 5
    assert [a.id for a in loaded.arms] == [a.id for a in original.arms]


def test_arm_lookup():
    config = ExperimentConfig()
    assert config.arm("C5").example_strategy == "dynamic"
    with pytest.raises(KeyError, match="C9"):
        config.arm("C9")
