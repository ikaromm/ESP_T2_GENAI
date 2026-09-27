"""Recuperacao densa: embeddings + indice FAISS.

Duas bases distintas sao usadas no estudo e nao devem ser confundidas:

1. **Base de contexto (RAG)** -- trechos do PROPRIO relatorio a ser resumido.
   Serve para escolher que partes do documento entram no prompt.
2. **Base de exemplos (few-shot)** -- pares (relatorio, resumo) de OUTROS
   documentos, sempre do split de treino. Serve para escolher demonstracoes.

Manter as duas separadas evita vazamento: um exemplo few-shot nunca pode ser o
proprio documento avaliado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .context import prefix, token_count

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Interface minima de um codificador de texto."""

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Codificador baseado em sentence-transformers, com vetores normalizados.

    Os vetores sao normalizados em L2 para que o produto interno do FAISS
    (`IndexFlatIP`) corresponda a similaridade de cosseno.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        # Token IDs sao passados ao forward: decode/reencode poderia alterar o
        # tamanho das janelas e voltar a provocar truncamento silencioso.
        import torch

        tokenizer = self.model.tokenizer
        if not tokenizer.is_fast:
            raise ValueError("encoder exige tokenizer fast para preservar todas as janelas")
        special = tokenizer.num_special_tokens_to_add(False)
        if self.model.max_seq_length <= special:
            raise ValueError("janela do encoder nao comporta tokens de conteudo")
        windows, owners, weights = [], [], []
        for owner, text in enumerate(texts):
            remaining = text
            while True:
                part = prefix(tokenizer, remaining, self.model.max_seq_length - special)
                if remaining and not part:
                    raise ValueError("tokenizer nao produziu uma janela com conteudo")
                encoded = tokenizer(part, truncation=False, return_attention_mask=True)
                if len(encoded["input_ids"]) > self.model.max_seq_length:
                    raise ValueError("janela ultrapassou o limite do encoder")
                windows.append(
                    {key: encoded[key] for key in tokenizer.model_input_names if key in encoded}
                )
                owners.append(owner)
                weights.append(max(1, token_count(tokenizer, part)))
                remaining = remaining[len(part) :]
                if not remaining:
                    break
        totals = np.zeros((len(texts), self.dimension), dtype=np.float32)
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                batch = tokenizer.pad(
                    windows[start : start + self.batch_size], padding=True, return_tensors="pt"
                )
                features = {key: value.to(self.model.device) for key, value in batch.items()}
                vectors = self.model(features)["sentence_embedding"].cpu().numpy()
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)
                for offset, vector in enumerate(vectors):
                    j = start + offset
                    totals[owners[j]] += vector * weights[j]
        norms = np.linalg.norm(totals, axis=1, keepdims=True)
        return totals / np.maximum(norms, 1e-12)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


@dataclass(frozen=True)
class Hit:
    """Um resultado de busca."""

    position: int
    score: float
    payload: object


class VectorIndex:
    """Indice denso de similaridade de cosseno sobre `IndexFlatIP` do FAISS.

    `IndexFlatIP` faz busca exata; para os volumes deste estudo (milhares de
    trechos por documento, dezenas de milhares de exemplos) o custo e aceitavel
    e evita a variabilidade de indices aproximados nos resultados do experimento.
    """

    def __init__(self, dimension: int) -> None:
        import faiss

        self._faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.payloads: list[object] = []

    def add(self, vectors: np.ndarray, payloads: list[object]) -> None:
        """Adiciona vetores e seus objetos associados.

        Raises:
            ValueError: se a quantidade de vetores e de payloads nao coincidir.
        """
        if len(vectors) != len(payloads):
            raise ValueError(f"{len(vectors)} vetores para {len(payloads)} payloads")
        if not len(vectors):
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"dimensao incompativel: indice={self.dimension}, vetores={vectors.shape[1]}"
            )
        self.index.add(vectors)
        self.payloads.extend(payloads)

    def search(self, query: np.ndarray, top_k: int) -> list[Hit]:
        """Busca os `top_k` itens mais similares a um unico vetor de consulta."""
        if not len(self.payloads) or top_k <= 0:
            return []
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        k = min(top_k, len(self.payloads))
        scores, positions = self.index.search(query, k)
        return [
            Hit(position=int(p), score=float(s), payload=self.payloads[int(p)])
            for s, p in zip(scores[0], positions[0], strict=True)
            if p >= 0
        ]

    def __len__(self) -> int:
        return len(self.payloads)

    def save(self, path: Path | str) -> None:
        """Persiste apenas os vetores; os payloads sao responsabilidade do chamador."""
        self._faiss.write_index(self.index, str(path))


def build_index(encoder: Encoder, texts: list[str], payloads: list[object]) -> VectorIndex:
    """Codifica os textos e devolve um indice povoado.

    Raises:
        ValueError: se textos e payloads tiverem tamanhos diferentes.
    """
    if len(texts) != len(payloads):
        raise ValueError(f"{len(texts)} textos para {len(payloads)} payloads")
    vectors = encoder.encode_texts(texts)
    dimension = vectors.shape[1] if len(vectors) else getattr(encoder, "dimension", 0)
    index = VectorIndex(dimension)
    index.add(vectors, payloads)
    return index
