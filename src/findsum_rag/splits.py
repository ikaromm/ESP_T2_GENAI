"""Construcao e carga dos conjuntos experimentais.

Os splits originais do FINDSum (`train`/`val`/`test`) foram feitos para treinar
modelos seq2seq. Neste estudo nada e treinado, e os conjuntos necessarios sao
outros tres:

* **exemplos**  -- pares (relatorio, resumo) que alimentam o few-shot de C3 a C5;
* **dev**       -- onde se ajusta prompt, `top_k` e numero de exemplos;
* **avaliacao** -- aberto uma unica vez, no fim; e o numero que vai no trabalho.

Por isso os tres splits originais sao fundidos num pool unico (20.672 documentos
da tarefa Liquidity) e reparticionados. O preco e perder comparabilidade com
resultados publicados sobre o split oficial; como o estudo compara C1 a C5 entre
si, e um preco baixo -- mas precisa constar na redacao.

Duas invariantes, ambas exigidas pelo desenho:

1. **Uma empresa aparece em um unico conjunto, com um unico relatorio.** Sem
   isso, um exemplo few-shot poderia ser outro exercicio da mesma empresa
   avaliada, e o ganho da selecao dinamica viria de quase-duplicacao em vez de
   relevancia semantica.
2. **A particao e congelada em disco.** Ampliar a avaliacao depois de olhar
   resultados so nao e cherry-picking se o conjunto ja estava definido antes.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .data import SPLITS, Document, FindSumPaths, Task, load_documents

# Accession number da SEC: 10 digitos do filer, 2 do ano, 6 sequenciais.
ACCESSION_RE = re.compile(r"^\d{10}-(\d{2})-(\d{6})$")

# Resumos muito curtos indicam registro truncado na origem. O plano preve
# descartar por "integridade do conteudo" e "existencia de resumo de referencia".
MIN_SUMMARY_WORDS = 100


class DocumentRef(BaseModel):
    """Referencia a um documento no pool, por split original e linha."""

    split: str
    row: int = Field(ge=0)
    doc_id: str
    stock_name: str
    report_id: str

    @property
    def year(self) -> int | None:
        """Ano do arquivamento, extraido do accession number."""
        match = ACCESSION_RE.match(self.report_id)
        if not match:
            return None
        two = int(match.group(1))
        return 2000 + two if two < 80 else 1900 + two


class SplitManifest(BaseModel):
    """Particao congelada dos conjuntos experimentais."""

    task: Task
    seed: int
    created_at: str
    min_summary_words: int
    sets: dict[str, list[DocumentRef]]

    @model_validator(mode="after")
    def _check_disjoint(self) -> SplitManifest:
        # A checagem de documento vem primeiro por ser a mais especifica: um
        # mesmo doc_id repetido tambem viola a regra de empresa, e reportar
        # "empresa X em eval e em eval" esconderia a causa real.
        seen_doc: dict[str, str] = {}
        for name, refs in self.sets.items():
            for ref in refs:
                if ref.doc_id in seen_doc:
                    raise ValueError(
                        f"documento {ref.doc_id} repetido em {name} e em {seen_doc[ref.doc_id]}"
                    )
                seen_doc[ref.doc_id] = name

        seen_company: dict[str, str] = {}
        for name, refs in self.sets.items():
            for ref in refs:
                if ref.stock_name in seen_company:
                    raise ValueError(
                        f"empresa {ref.stock_name} em {name} e em "
                        f"{seen_company[ref.stock_name]}"
                    )
                seen_company[ref.stock_name] = name
        return self

    def sizes(self) -> dict[str, int]:
        return {name: len(refs) for name, refs in self.sets.items()}

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> SplitManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def scan_pool(
    root: Path | str,
    task: Task,
    *,
    min_summary_words: int = MIN_SUMMARY_WORDS,
) -> list[DocumentRef]:
    """Varre os tres splits originais e devolve os documentos elegiveis.

    Descarta registros sem resumo utilizavel. Le os CSVs de texto apenas para
    medir o tamanho do resumo, e os arquivos de tabela apenas para obter
    `stock_name` e `report_id`.
    """
    paths = FindSumPaths(root)
    refs: list[DocumentRef] = []

    for split in SPLITS:
        missing = paths.missing(task, split)
        if missing:
            raise FileNotFoundError(
                f"arquivos ausentes para {task.value}/{split}: {missing}; "
                "rode python scripts/fetch_findsum.py"
            )
        documents = load_documents(root, task, split)
        for document in documents:
            if document.n_summary_words < min_summary_words:
                continue
            if not (document.stock_name and document.report_id):
                continue
            refs.append(
                DocumentRef(
                    split=split,
                    row=document.row,
                    doc_id=document.doc_id,
                    stock_name=document.stock_name,
                    report_id=document.report_id,
                )
            )
    return refs


def pick_one_per_company(refs: list[DocumentRef]) -> list[DocumentRef]:
    """Mantem um relatorio por empresa: o mais recente.

    O criterio e deterministico e nao depende de semente. Preferir o arquivamento
    mais recente tambem mantem os dados o mais atuais que o dataset permite.
    """
    by_company: dict[str, list[DocumentRef]] = defaultdict(list)
    for ref in refs:
        by_company[ref.stock_name].append(ref)

    chosen: list[DocumentRef] = []
    for company in sorted(by_company):
        candidates = by_company[company]
        chosen.append(
            max(candidates, key=lambda r: (r.year or 0, r.report_id, r.split, r.row))
        )
    return chosen


def build_manifest(
    root: Path | str,
    task: Task,
    *,
    sizes: dict[str, int],
    seed: int = 42,
    min_summary_words: int = MIN_SUMMARY_WORDS,
) -> SplitManifest:
    """Monta a particao congelada.

    Args:
        sizes: tamanho de cada conjunto, por exemplo
            `{"examples": 1000, "dev": 500, "eval": 1000}`.
        seed: semente do sorteio das empresas entre os conjuntos.

    Raises:
        ValueError: se o pool nao tiver empresas suficientes.
    """
    pool = pick_one_per_company(scan_pool(root, task, min_summary_words=min_summary_words))
    needed = sum(sizes.values())
    if needed > len(pool):
        raise ValueError(
            f"a particao pede {needed} documentos, mas o pool tem {len(pool)} "
            f"empresas elegiveis na tarefa {task.value}"
        )

    rng = random.Random(seed)
    shuffled = sorted(pool, key=lambda r: r.stock_name)
    rng.shuffle(shuffled)

    sets: dict[str, list[DocumentRef]] = {}
    cursor = 0
    for name, size in sizes.items():
        sets[name] = sorted(shuffled[cursor : cursor + size], key=lambda r: r.doc_id)
        cursor += size

    return SplitManifest(
        task=task,
        seed=seed,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        min_summary_words=min_summary_words,
        sets=sets,
    )


def load_set(
    root: Path | str,
    manifest: SplitManifest,
    name: str,
    *,
    limit: int | None = None,
    with_tables: bool = True,
) -> list[Document]:
    """Carrega os documentos de um conjunto do manifesto.

    As referencias apontam para linhas nos splits originais, por isso cada split
    e lido uma unica vez e apenas as linhas necessarias sao retidas.

    Raises:
        KeyError: se o conjunto nao existir no manifesto.
    """
    if name not in manifest.sets:
        raise KeyError(
            f"conjunto {name!r} ausente; disponiveis: {sorted(manifest.sets)}"
        )

    refs = manifest.sets[name]
    if limit is not None:
        refs = refs[:limit]

    wanted: dict[str, set[int]] = defaultdict(set)
    for ref in refs:
        wanted[ref.split].add(ref.row)

    by_key: dict[tuple[str, int], Document] = {}
    for split, rows in wanted.items():
        documents = load_documents(root, manifest.task, split, with_tables=with_tables)
        for document in documents:
            if document.row in rows:
                by_key[(split, document.row)] = document

    ordered: list[Document] = []
    for ref in refs:
        document = by_key.get((ref.split, ref.row))
        if document is None:
            raise ValueError(f"documento {ref.doc_id} do manifesto nao encontrado")
        ordered.append(document)
    return ordered
