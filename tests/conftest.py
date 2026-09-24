"""Fixtures compartilhadas, incluindo um FINDSum sintetico em disco.

Os testes unitarios nao devem depender dos 6.4 GB do dataset real, por isso as
fixtures escrevem uma copia minima com a MESMA estrutura de arquivos e os mesmos
marcadores. Os testes que exigem os dados reais ficam em `test_real_data.py` e
sao pulados quando o download nao esta presente.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from findsum_rag.data import Task

# Dois documentos, tres segmentos (tarefa Liquidity). O resumo de cada documento
# e a concatenacao das porcoes, exatamente como no dataset real.
DOC_SEGMENTS = {
    "doc0": [
        (
            "net cash used in operating activities was $ 50.0 million "
            "replace_table_token_1_th",
            "cash flow net cash used in operating activities was $ 50.0 million",
        ),
        (
            "story_separator_special_tag the company repaid $ 12.5 million of debt",
            "the company repaid $ 12.5 million of debt during 2016",
        ),
        (
            "capital expenditures totalled $ 3.4 million replace_table_token_2_th",
            "capital expenditures totalled $ 3.4 million",
        ),
    ],
    "doc1": [
        (
            "revenues increased 12.5 % to $ 1,250.0 million in fiscal 2020",
            "revenues increased 12.5 % to $ 1,250.0 million",
        ),
        (
            "gross margin was 41.2 % compared with 39.8 % replace_table_token_0_th",
            "gross margin was 41.2 % compared with 39.8 %",
        ),
        (
            "operating expenses rose to $ 410.0 million",
            "operating expenses rose to $ 410.0 million",
        ),
    ],
}

TABLE_RECORDS = [
    {
        "stock_name": "AAA",
        "report_id": "0000000000-16-000001",
        "mda_liquidity_tables": [
            [["cash and cash equivalents", "", "50.0", "2016", 2, 2]],
            [["total debt", "", "12.5", "2016", 3, 2]],
            [["capital expenditures", "", "3.4", "2016", 4, 2]],
        ],
    },
    {
        "stock_name": "BBB",
        "report_id": "0000000000-20-000002",
        "mda_liquidity_tables": [
            [["total revenue", "", "1,250.0", "2020", 2, 2]],
        ],
    },
]


def _write_split(root: Path, task: Task, split: str) -> None:
    text_dir = root / "text" / task.dir_name / f"{task.value}_input_2000"
    text_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = list(DOC_SEGMENTS)

    for segment in range(task.n_segments):
        frame = pd.DataFrame(
            {
                "document": [DOC_SEGMENTS[d][segment][0] for d in doc_ids],
                "summary": [DOC_SEGMENTS[d][segment][1] for d in doc_ids],
            }
        )
        name = f"{split}_{task.value}_segment_{segment}_input_2_1000.csv"
        frame.to_csv(text_dir / name, index=False)

    table_dir = root / "table" / task.dir_name
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / f"{split}_{task.value}_all_tuples_diff_sec.txt"
    with table_path.open("w", encoding="utf-8") as handle:
        for record in TABLE_RECORDS:
            handle.write(json.dumps(record) + "\n")


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """Raiz de um FINDSum sintetico com os splits `train`, `val` e `test`."""
    root = tmp_path / "findsum"
    for split in ("train", "val", "test"):
        _write_split(root, Task.LIQUIDITY, split)
    return root


class StubEncoder:
    """Codificador deterministico sem modelo, para testar indice e selecao.

    Projeta o texto num espaco de `dimension` dimensoes por hashing de palavras e
    normaliza em L2, de modo que textos com vocabulario parecido fiquem proximos.
    Usa `crc32` em vez de `hash()`: o hash de strings do Python e aleatorizado por
    processo, o que tornaria os testes intermitentes.
    """

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def encode_texts(self, texts: list[str]) -> object:
        import zlib

        import numpy as np

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                bucket = zlib.crc32(word.encode("utf-8")) % self.dimension
                vectors[i, bucket] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


@pytest.fixture
def stub_encoder() -> StubEncoder:
    return StubEncoder()
