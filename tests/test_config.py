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


def test_default_arms_cover_c1_to_c5():
    assert [a.id for a in DEFAULT_ARMS] == ["C1", "C2", "C3", "C4", "C5"]
    by_id = {a.id: a for a in DEFAULT_ARMS}
    # C1 e a unica sem RAG; C1 e C2 sao as unicas sem exemplos.
    assert by_id["C1"].use_rag is False
    assert all(by_id[i].use_rag for i in ("C2", "C3", "C4", "C5"))
    assert by_id["C1"].example_strategy == "none"
    assert by_id["C2"].example_strategy == "none"
    assert by_id["C3"].example_strategy == "fixed"
    assert by_id["C4"].example_strategy == "random"
    assert by_id["C5"].example_strategy == "dynamic"


def test_arm_rejects_examples_with_none_strategy():
    with pytest.raises(ValidationError, match="nao aceita n_examples"):
        ExperimentArm(id="X", label="x", use_rag=True, example_strategy="none", n_examples=2)


def test_arm_requires_examples_for_other_strategies():
    with pytest.raises(ValidationError, match="exige n_examples"):
        ExperimentArm(id="X", label="x", use_rag=True, example_strategy="dynamic", n_examples=0)


def test_arm_rejects_unknown_strategy():
    with pytest.raises(ValidationError):
        ExperimentArm(id="X", label="x", use_rag=True, example_strategy="magica", n_examples=1)


def test_retrieval_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        RetrievalConfig(chunk_size=100, chunk_overlap=100)


def test_data_config_rejects_same_split_for_eval_and_examples():
    with pytest.raises(ValidationError, match="vazamento"):
        DataConfig(eval_split="val", example_split="val")


def test_duplicate_arm_ids_rejected():
    arm = ExperimentArm(id="C1", label="a", use_rag=False, example_strategy="none")
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
