"""Testes das estrategias de selecao de exemplos few-shot.

As invariantes testadas aqui sao as que protegem a validade do experimento:
nenhum documento serve de exemplo para si mesmo, e o sorteio de C4 e
reproduzivel entre execucoes.
"""

from __future__ import annotations

import pytest

from findsum_rag.examples import (
    DynamicExamples,
    Example,
    ExampleStore,
    FixedExamples,
    NoExamples,
    RandomExamples,
    make_selector,
)


@pytest.fixture
def store() -> ExampleStore:
    return ExampleStore(
        [
            Example("d1", "cash flow operating activities debt repayment", "resumo de caixa"),
            Example("d2", "revenue growth gross margin products", "resumo de receita"),
            Example("d3", "cash flow financing activities credit facility", "resumo de liquidez"),
            Example("d4", "revenue segments international sales", "resumo de vendas"),
        ]
    )


def test_no_examples():
    selector = NoExamples()
    assert selector.select("d1") == []
    assert selector.n_examples == 0


def test_fixed_examples_are_stable(store: ExampleStore):
    selector = FixedExamples(store, n_examples=2)
    first = [e.doc_id for e in selector.select("dX")]
    second = [e.doc_id for e in selector.select("dY")]
    assert first == second == ["d1", "d2"]


def test_fixed_examples_exclude_self(store: ExampleStore):
    selector = FixedExamples(store, n_examples=2)
    assert "d1" not in [e.doc_id for e in selector.select("d1")]


def test_random_examples_are_reproducible(store: ExampleStore):
    a = RandomExamples(store, n_examples=2, seed=7)
    b = RandomExamples(store, n_examples=2, seed=7)
    assert [e.doc_id for e in a.select("dX")] == [e.doc_id for e in b.select("dX")]


def test_random_examples_vary_by_document(store: ExampleStore):
    selector = RandomExamples(store, n_examples=2, seed=7)
    picks = {tuple(e.doc_id for e in selector.select(f"d{i}")) for i in range(20, 40)}
    assert len(picks) > 1, "o sorteio deve variar entre documentos"


def test_random_examples_exclude_self(store: ExampleStore):
    selector = RandomExamples(store, n_examples=3, seed=1)
    for doc_id in ("d1", "d2", "d3", "d4"):
        assert doc_id not in [e.doc_id for e in selector.select(doc_id)]


def test_random_examples_handles_small_pool():
    store = ExampleStore([Example("only", "text", "summary")])
    assert RandomExamples(store, n_examples=3).select("only") == []


def test_dynamic_examples_pick_semantically_close(store: ExampleStore, stub_encoder):
    store.build_index(stub_encoder)
    selector = DynamicExamples(store, n_examples=1)
    query = stub_encoder.encode_texts(["cash flow operating activities debt repayment"])[0]
    # O mais proximo seria d1; excluido, deve cair no outro documento de caixa.
    assert [e.doc_id for e in selector.select("d1", query)] == ["d3"]


def test_dynamic_examples_require_query_vector(store: ExampleStore, stub_encoder):
    store.build_index(stub_encoder)
    with pytest.raises(ValueError, match="query_vector"):
        DynamicExamples(store, n_examples=1).select("d1")


def test_dynamic_examples_require_index(store: ExampleStore, stub_encoder):
    query = stub_encoder.encode_texts(["cash"])[0]
    with pytest.raises(RuntimeError, match="indice de exemplos"):
        DynamicExamples(store, n_examples=1).select("dX", query)


def test_example_truncation():
    example = Example("d", "one two three four five", "s")
    assert example.truncated(3).document == "one two three"
    # Nao trunca o que ja cabe, e devolve o mesmo objeto.
    assert example.truncated(99) is example


def test_make_selector_builds_each_strategy(store: ExampleStore):
    assert make_selector("none").name == "none"
    assert make_selector("fixed", store=store, n_examples=1).name == "fixed"
    assert make_selector("random", store=store, n_examples=1).name == "random"
    assert make_selector("dynamic", store=store, n_examples=1).name == "dynamic"


def test_make_selector_rejects_unknown():
    with pytest.raises(ValueError, match="estrategia desconhecida"):
        make_selector("magica", store=None)


def test_make_selector_requires_store():
    with pytest.raises(ValueError, match="ExampleStore"):
        make_selector("dynamic", store=None, n_examples=2)
