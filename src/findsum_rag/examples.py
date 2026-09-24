"""Estrategias de selecao de exemplos few-shot (C1 a C5).

Cada estrategia responde a mesma pergunta -- quais demonstracoes entram no
prompt -- de um modo diferente, e a comparacao entre elas e o objeto do estudo:

* `none`    (C1, C2) nenhum exemplo;
* `fixed`   (C3) os mesmos exemplos para todos os documentos;
* `random`  (C4) exemplos sorteados, com semente fixa por documento;
* `dynamic` (C5) exemplos mais similares semanticamente ao documento de entrada.

Todas as estrategias respeitam duas invariantes: os exemplos vem apenas do split
de treino e nunca incluem o documento que esta sendo resumido.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .retrieval import Encoder, VectorIndex, build_index


@dataclass(frozen=True)
class Example:
    """Um par (relatorio, resumo) usado como demonstracao."""

    doc_id: str
    document: str
    summary: str

    def truncated(self, max_words: int) -> Example:
        """Versao com o documento limitado a `max_words` palavras."""
        words = self.document.split()
        if len(words) <= max_words:
            return self
        return Example(self.doc_id, " ".join(words[:max_words]), self.summary)


class ExampleStore:
    """Base de exemplos few-shot construida a partir do split de treino."""

    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples
        self._index: VectorIndex | None = None

    def build_index(self, encoder: Encoder, *, query_words: int = 400) -> None:
        """Indexa os exemplos para a selecao dinamica.

        Apenas as primeiras `query_words` palavras de cada documento sao
        codificadas: os codificadores de sentenca truncam entradas longas, logo
        alimentar o documento inteiro daria a ilusao de usar todo o conteudo.
        """
        texts = [" ".join(e.document.split()[:query_words]) for e in self.examples]
        self._index = build_index(encoder, texts, list(self.examples))

    @property
    def index(self) -> VectorIndex:
        if self._index is None:
            raise RuntimeError("indice de exemplos nao construido; chame build_index()")
        return self._index

    def __len__(self) -> int:
        return len(self.examples)


class ExampleSelector(ABC):
    """Contrato comum das estrategias de selecao."""

    name: str

    def __init__(self, n_examples: int = 0) -> None:
        self.n_examples = n_examples

    @abstractmethod
    def select(self, doc_id: str, query_vector: np.ndarray | None = None) -> list[Example]:
        """Devolve os exemplos para um documento, excluindo o proprio documento."""


class NoExamples(ExampleSelector):
    """C1/C2: geracao sem demonstracoes."""

    name = "none"

    def __init__(self) -> None:
        super().__init__(n_examples=0)

    def select(self, doc_id: str, query_vector: np.ndarray | None = None) -> list[Example]:
        return []


class FixedExamples(ExampleSelector):
    """C3: os mesmos exemplos para todos os documentos.

    Os exemplos sao os `n_examples` primeiros da base apos ordenacao por
    `doc_id`, o que torna a escolha reproduzivel e independente da ordem de
    leitura dos arquivos.
    """

    name = "fixed"

    def __init__(self, store: ExampleStore, n_examples: int = 2) -> None:
        super().__init__(n_examples)
        self.store = store
        self._pool = sorted(store.examples, key=lambda e: e.doc_id)

    def select(self, doc_id: str, query_vector: np.ndarray | None = None) -> list[Example]:
        chosen = [e for e in self._pool if e.doc_id != doc_id]
        return chosen[: self.n_examples]


class RandomExamples(ExampleSelector):
    """C4: exemplos sorteados.

    A semente combina uma semente global com o `doc_id`, de modo que o sorteio
    e diferente entre documentos mas identico entre execucoes -- condicao para
    que a comparacao com C5 seja reproduzivel.
    """

    name = "random"

    def __init__(self, store: ExampleStore, n_examples: int = 2, seed: int = 42) -> None:
        super().__init__(n_examples)
        self.store = store
        self.seed = seed

    def select(self, doc_id: str, query_vector: np.ndarray | None = None) -> list[Example]:
        pool = [e for e in self.store.examples if e.doc_id != doc_id]
        if not pool:
            return []
        rng = random.Random(f"{self.seed}:{doc_id}")
        k = min(self.n_examples, len(pool))
        return rng.sample(pool, k)


class DynamicExamples(ExampleSelector):
    """C5: exemplos semanticamente mais similares ao documento de entrada.

    Esta e a estrategia proposta pela pesquisa. Requer `query_vector`, o
    embedding do documento a ser resumido, e busca sobre a base de exemplos
    indexada.
    """

    name = "dynamic"

    def __init__(self, store: ExampleStore, n_examples: int = 2) -> None:
        super().__init__(n_examples)
        self.store = store

    def select(self, doc_id: str, query_vector: np.ndarray | None = None) -> list[Example]:
        if query_vector is None:
            raise ValueError("DynamicExamples exige query_vector")
        # Busca folgada: o proprio documento pode estar na base (quando o split
        # avaliado e o de treino) e precisa ser descartado sem reduzir o k final.
        hits = self.store.index.search(query_vector, self.n_examples + 5)
        chosen: list[Example] = []
        for hit in hits:
            example = hit.payload
            assert isinstance(example, Example)
            if example.doc_id == doc_id:
                continue
            chosen.append(example)
            if len(chosen) == self.n_examples:
                break
        return chosen


SELECTORS = {
    NoExamples.name: NoExamples,
    FixedExamples.name: FixedExamples,
    RandomExamples.name: RandomExamples,
    DynamicExamples.name: DynamicExamples,
}


def make_selector(
    strategy: str,
    *,
    store: ExampleStore | None = None,
    n_examples: int = 2,
    seed: int = 42,
) -> ExampleSelector:
    """Instancia uma estrategia pelo nome.

    Raises:
        ValueError: se a estrategia for desconhecida ou faltar a base de exemplos.
    """
    if strategy == NoExamples.name:
        return NoExamples()
    if strategy not in SELECTORS:
        raise ValueError(f"estrategia desconhecida: {strategy!r}; use {sorted(SELECTORS)}")
    if store is None:
        raise ValueError(f"a estrategia {strategy!r} exige uma ExampleStore")
    if strategy == RandomExamples.name:
        return RandomExamples(store, n_examples=n_examples, seed=seed)
    return SELECTORS[strategy](store, n_examples=n_examples)
