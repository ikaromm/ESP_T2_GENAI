"""Regressoes dos controles experimentais, sem pesos remotos ou GPU."""

import copy

import numpy as np
import pytest

from findsum_rag.analyze import HYPOTHESES, compare_hypotheses
from findsum_rag.context import matched_contexts, prefix, token_count
from findsum_rag.factual import score_review
from findsum_rag.metrics import numeric_prf
from findsum_rag.retrieval import SentenceTransformerEncoder


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "tail": 4, "other": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


def test_real_tokenizer_equal_budget_different_lengths(tokenizer):
    a, b = matched_contexts(
        tokenizer,
        "alpha " * 40,
        "beta other " * 100,
        12,
        lambda text: token_count(tokenizer, "alpha\n" + text + "\ntail"),
    )
    assert token_count(tokenizer, a) == token_count(tokenizer, b) == 12
    assert len(a) != len(b)


def test_table_budget_never_returns_partial_cell_or_orphan_header(tokenizer):
    source = "alpha beta\n[tabela test]\ncash | 12.6 & 12,500 (2020)\ndebt | 42 (2019)"
    for budget in range(1, token_count(tokenizer, source)):
        result = prefix(tokenizer, source, budget, preserve_table_rows=True)
        assert source.startswith(result)
        assert token_count(tokenizer, result) <= budget
        if "cash" in result:
            assert "cash | 12.6 & 12,500 (2020)" in result
        if "debt" in result:
            assert "debt | 42 (2019)" in result
        assert not result.endswith("[tabela test]")


def test_embedding_windows_can_split_long_table_like_prose(tokenizer):
    text = "alpha " * 30 + " | " + "beta " * 30
    part = prefix(tokenizer, text, 10)
    assert part and token_count(tokenizer, part) == 10


def test_equal_budget_survives_table_boundary(tokenizer):
    retrieved = "[tabela test]\ncash | 12.6 & 12,500 (2020)\ndebt | 42 (2019)"
    a, b = matched_contexts(
        tokenizer, "alpha " * 100, retrieved, 20, lambda text: token_count(tokenizer, text)
    )
    assert token_count(tokenizer, a) == token_count(tokenizer, b)
    assert b.endswith("(2020)") or b.endswith("(2019)")


def test_encoder_includes_tail_and_normalizes(tokenizer):
    import torch

    class Model:
        max_seq_length = 4
        device = "cpu"

        def eval(self):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def __call__(self, batch):
            ids = batch["input_ids"]
            assert ids.shape[1] <= self.max_seq_length
            mask = batch["attention_mask"]
            return {
                "sentence_embedding": torch.stack(
                    ((ids * mask).sum(1).float(), mask.sum(1).float()), dim=1
                )
            }

    encoder = SentenceTransformerEncoder.__new__(SentenceTransformerEncoder)
    encoder.model = Model()
    encoder.model.tokenizer = tokenizer
    encoder.batch_size = 2
    vectors = encoder.encode_texts(["alpha " * 20 + "tail", "alpha " * 20 + "other"])
    assert not np.allclose(vectors[0], vectors[1])
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1, 1])
    assert encoder.encode_texts([]).shape == (0, 2)


def complete_scores():
    return {
        arm: {f"d{i}": {"bertscore": 0.8, "rougeL": 0.5} for i in range(4)}
        for arm in ("C1", "C1t", "C2", "C3", "C4", "C5")
    }


def test_all_hypotheses_and_ties_remain_in_holm_family():
    comparisons = compare_hypotheses(complete_scores())
    assert {c.metric for c in comparisons} == {"bertscore", "rougeL"}
    assert len(comparisons) == 10
    assert {(c.hypothesis, c.arm_a, c.arm_b) for c in comparisons} == set(HYPOTHESES)
    assert all(c.p_adjusted == 1 for c in comparisons)


@pytest.mark.parametrize("defect", ["arm", "document", "metric", "nan"])
def test_incomplete_confirmatory_analysis_fails(defect):
    scores = complete_scores()
    if defect == "arm":
        del scores["C1t"]
    elif defect == "document":
        del scores["C3"]["d0"]
    elif defect == "metric":
        del scores["C2"]["d0"]["bertscore"]
    else:
        scores["C4"]["d0"]["rougeL"] = float("nan")
    with pytest.raises(ValueError):
        compare_hypotheses(scores)


def reviewed_case():
    return {
        "case_id": "x",
        "prediction": "revenue was 20 and profit was 100",
        "source": "revenue was 100 and profit was 20",
        "context": "revenue was 100",
        "reviewer": "reviewer-1",
        "segmentation_complete": True,
        "claims": [
            {
                "text": "revenue was 20",
                "entity": "company",
                "period": "unspecified",
                "relation": "revenue",
                "value": "20",
                "source_label": "contradicted",
                "context_label": "contradicted",
                "source_evidence": "revenue was 100",
                "context_evidence": "revenue was 100",
                "rationale": "swapped values",
            },
            {
                "text": "profit was 100",
                "entity": "company",
                "period": "unspecified",
                "relation": "profit",
                "value": "100",
                "source_label": "contradicted",
                "context_label": "insufficient_evidence",
                "source_evidence": "profit was 20",
                "context_evidence": "",
                "rationale": "swapped values; omitted context",
            },
        ],
    }


def test_swapped_numbers_not_certified_as_factual():
    case = reviewed_case()
    assert numeric_prf(case["prediction"], case["source"]).f1 == 1
    score = score_review([case])[0]
    assert score["source_supported_rate"] == 0
    assert score["source_contradicted_rate"] == 1
    assert score["context_insufficient_evidence_rate"] == 0.5


@pytest.mark.parametrize("defect", ["unfinished", "evidence", "empty", "label"])
def test_factual_review_cannot_score_unreviewed_or_untraceable_claims(defect):
    case = copy.deepcopy(reviewed_case())
    if defect == "unfinished":
        case["segmentation_complete"] = False
    elif defect == "evidence":
        case["claims"][0]["source_evidence"] = "fabricated evidence"
    elif defect == "empty":
        case["claims"] = []
    else:
        case["claims"][0]["source_label"] = ""
    with pytest.raises(ValueError):
        score_review([case])


def test_encoder_with_local_transformer_and_special_tokens(tmp_path, tokenizer):
    """Forward real do SentenceTransformer, pesos aleatorios locais e CPU."""
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel

    tokenizer.add_special_tokens({"cls_token": "[CLS]", "sep_token": "[SEP]"})
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tokenizer.cls_token_id), ("[SEP]", tokenizer.sep_token_id)],
    )
    tokenizer.save_pretrained(tmp_path)
    model = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            max_position_embeddings=16,
        )
    )
    model.save_pretrained(tmp_path)
    encoder = SentenceTransformerEncoder(str(tmp_path), device="cpu", batch_size=2)
    encoder.model.max_seq_length = 8
    vectors = encoder.encode_texts(["alpha " * 20 + "tail", "alpha " * 20 + "other"])
    assert vectors.shape == (2, 16)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1, 1], abs=1e-6)
    assert not np.allclose(vectors[0], vectors[1])


def test_cli_confirmatory_default_and_factual_roundtrip(tmp_path):
    import csv
    import json

    from typer.testing import CliRunner

    from findsum_rag.analyze import ANALYSIS_PLAN
    from findsum_rag.cli import app
    from findsum_rag.factual import evidence_digest

    runner = CliRunner()
    for arm, scores in complete_scores().items():
        path = tmp_path / arm / "scores.csv"
        path.parent.mkdir()
        with path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["doc_id", "bertscore", "rougeL"])
            writer.writeheader()
            writer.writerows({"doc_id": key, **row} for key, row in scores.items())
    (tmp_path / "analysis_plan.json").write_text(json.dumps(ANALYSIS_PLAN))
    result = runner.invoke(app, ["compare", "--run-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert len(json.loads((tmp_path / "comparisons.json").read_text())) == 10
    assert "nao demonstra equivalencia" in result.output

    review = tmp_path / "review.json"
    output = tmp_path / "facts.json"
    case = reviewed_case()
    review.write_text(json.dumps([case]))
    review.with_suffix(".map.json").write_text(
        json.dumps(
            {
                case["case_id"]: {
                    "arm": "C5",
                    "doc_id": "d0",
                    "evidence_sha256": evidence_digest(case),
                }
            }
        )
    )
    result = runner.invoke(app, ["review-score", "--review", str(review), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())[0]["source_contradicted_rate"] == 1
    case["prediction"] += " tampered"
    review.write_text(json.dumps([case]))
    result = runner.invoke(
        app, ["review-score", "--review", str(review), "--output", str(tmp_path / "tampered.json")]
    )
    assert result.exit_code != 0
    assert "alterados" in result.output


def test_prose_pipe_is_not_a_serialized_table(tokenizer):
    source = "alpha " * 40 + " | 37 " + "beta " * 40
    a, b = matched_contexts(
        tokenizer, source, "beta " * 100, 12, lambda text: token_count(tokenizer, text)
    )
    assert source.startswith(a)
    assert token_count(tokenizer, a) == token_count(tokenizer, b) == 12


def test_numbered_table_header_protects_rows_but_not_next_prose(tokenizer):
    source = "[1] [tabela test]\ncash | 12 (2020)\n\n[2] " + "alpha " * 20 + " | page 37"
    budget = token_count(tokenizer, "[1] [tabela test]\ncash | 12 (2020)\n\n[2] alpha alpha")
    result = prefix(tokenizer, source, budget, preserve_table_rows=True)
    assert "cash | 12 (2020)" in result and "[2] alpha" in result
