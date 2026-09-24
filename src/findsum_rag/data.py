"""Carga e remontagem do FINDSum.

Formato dos arquivos distribuidos (verificado empiricamente, ver `docs/dataset.md`):

* `text/FINDSum-<Task>/<task>_input_2000/<split>_<task>_segment_<i>_input_2_1000.csv`
  CSV com colunas `document,summary`. Existem N arquivos de segmento por split
  (ROO: N=2, Liquidity: N=3), todos com o mesmo numero de linhas. A linha `r` de
  cada arquivo pertence ao MESMO relatorio: `segment_i` contem a i-esima porcao
  (~2000 palavras) do conteudo selecionado e a porcao correspondente do resumo
  de referencia. Concatenar os segmentos em ordem reconstroi entrada e resumo.

* `table/FINDSum-<Task>/<split>_<task>_all_tuples_diff_sec.txt`
  Um objeto JSON por linha, alinhado por indice de linha com os CSVs de texto.
  Traz `stock_name`, `report_id` (accession da SEC) e listas de tabelas; cada
  tabela e uma lista de tuplas `[rowname, colname, value, date, row_i, col_i]`.

Marcadores presentes no texto:

* `story_separator_special_tag` separa os trechos selecionados.
* `replace_table_token_<i>_th` indica onde a i-esima tabela do relatorio
  aparecia. Os indices sao globais ao relatorio e podem repetir entre segmentos,
  isto e, os segmentos sao selecoes de conteudo e nao janelas contiguas.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd

SEGMENT_SEPARATOR = "story_separator_special_tag"
TABLE_TOKEN_RE = re.compile(r"replace_table_token_(\d+)_th")

SPLITS = ("train", "val", "test")


class Task(StrEnum):
    """As duas tarefas do FINDSum."""

    ROO = "roo"
    LIQUIDITY = "liquidity"

    @property
    def n_segments(self) -> int:
        return 2 if self is Task.ROO else 3

    @property
    def dir_name(self) -> str:
        return "FINDSum-ROO" if self is Task.ROO else "FINDSum-Liquidity"

    @property
    def table_key(self) -> str:
        """Lista de tabelas mais pertinente a tarefa.

        Os nomes das chaves nao seguem o nome da tarefa: a ROO usa `result`, nao
        `roo`. Verificado nos arquivos reais (ver `TABLE_KEYS`).
        """
        return "mda_result_tables" if self is Task.ROO else "mda_liquidity_tables"


# Chaves de tabela presentes nos arquivos de tuplas, por tarefa, na ordem em que
# este projeto as numera: a chave especifica da tarefa primeiro, depois as de
# contexto. Confirmado lendo `val_roo` e `val_liquidity`.
TABLE_KEYS: dict[str, tuple[str, ...]] = {
    Task.ROO: (
        "mda_result_tables",
        "mda_before_result_tables",
        "mda_after_result_tables",
        "before_mda_tables",
        "after_mda_tables",
    ),
    Task.LIQUIDITY: (
        "mda_liquidity_tables",
        "mda_before_liquidity_tables",
        "mda_after_liquidity_tables",
        "before_mda_tables",
        "after_mda_tables",
    ),
}


@dataclass
class Table:
    """Uma tabela do relatorio, representada pelas suas celulas."""

    index: int
    cells: list[list]

    def to_text(self, max_cells: int = 120) -> str:
        """Lineariza a tabela para uso em prompt.

        Cada celula vira `linha | coluna | valor (data)`. As celulas do FINDSum
        ja vem sem cabecalho de coluna em muitos casos, por isso campos vazios
        sao omitidos em vez de virarem separadores soltos.
        """
        lines: list[str] = []
        for cell in self.cells[:max_cells]:
            row_name = str(cell[0]).strip() if len(cell) > 0 else ""
            col_name = str(cell[1]).strip() if len(cell) > 1 else ""
            value = str(cell[2]).strip() if len(cell) > 2 else ""
            date = str(cell[3]).strip() if len(cell) > 3 else ""
            parts = [p for p in (row_name, col_name, value) if p]
            if not parts:
                continue
            line = " | ".join(parts)
            if date:
                line += f" ({date})"
            lines.append(line)
        if len(self.cells) > max_cells:
            lines.append(f"[... {len(self.cells) - max_cells} celulas omitidas]")
        return "\n".join(lines)


@dataclass
class Segment:
    """Um par (trecho selecionado do relatorio, trecho do resumo)."""

    index: int
    document: str
    summary: str

    @property
    def passages(self) -> list[str]:
        """Os trechos selecionados, ja separados pela tag do dataset."""
        return [p.strip() for p in self.document.split(SEGMENT_SEPARATOR) if p.strip()]


@dataclass
class Document:
    """Um relatorio do FINDSum com entrada e resumo de referencia remontados."""

    doc_id: str
    task: Task
    split: str
    row: int
    segments: list[Segment]
    stock_name: str | None = None
    report_id: str | None = None
    tables: list[Table] = field(default_factory=list)

    @property
    def document(self) -> str:
        """Entrada completa: os segmentos concatenados na ordem original."""
        return " ".join(s.document for s in self.segments)

    @property
    def summary(self) -> str:
        """Resumo de referencia completo: as porcoes concatenadas em ordem."""
        return " ".join(s.summary for s in self.segments)

    @property
    def passages(self) -> list[str]:
        """Todos os trechos selecionados do relatorio, em ordem."""
        return [p for s in self.segments for p in s.passages]

    @property
    def table_indices(self) -> list[int]:
        """Indices de tabela referenciados pelos marcadores, sem repeticao."""
        seen: dict[int, None] = {}
        for match in TABLE_TOKEN_RE.finditer(self.document):
            seen.setdefault(int(match.group(1)), None)
        return list(seen)

    def referenced_tables(self) -> list[Table]:
        """As tabelas efetivamente citadas no texto de entrada."""
        by_index = {t.index: t for t in self.tables}
        return [by_index[i] for i in self.table_indices if i in by_index]

    @property
    def n_words(self) -> int:
        return len(self.document.split())

    @property
    def n_summary_words(self) -> int:
        return len(self.summary.split())


class FindSumPaths:
    """Resolve os caminhos dos arquivos brutos do FINDSum."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def text_dir(self, task: Task) -> Path:
        return self.root / "text" / task.dir_name / f"{task.value}_input_2000"

    def segment_file(self, task: Task, split: str, segment: int) -> Path:
        name = f"{split}_{task.value}_segment_{segment}_input_2_1000.csv"
        return self.text_dir(task) / name

    def table_file(self, task: Task, split: str) -> Path:
        name = f"{split}_{task.value}_all_tuples_diff_sec.txt"
        return self.root / "table" / task.dir_name / name

    def segment_files(self, task: Task, split: str) -> list[Path]:
        return [self.segment_file(task, split, i) for i in range(task.n_segments)]

    def missing(self, task: Task, split: str, *, need_tables: bool = True) -> list[Path]:
        paths = self.segment_files(task, split)
        if need_tables:
            paths.append(self.table_file(task, split))
        return [p for p in paths if not p.exists()]


def _iter_table_records(path: Path, limit: int | None = None) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def _tables_from_record(record: dict, task: Task) -> list[Table]:
    """Achata as listas de tabelas do registro em `Table` numeradas.

    A ordem e a de `TABLE_KEYS[task]`: a chave especifica da tarefa primeiro,
    depois as de contexto. O dataset nao documenta o mapeamento exato entre
    `replace_table_token_<i>_th` e essas listas, por isso a numeracao aqui e uma
    convencao do projeto -- consistente entre execucoes, mas nao garantidamente
    igual a do artigo original.

    Chaves desconhecidas que aparecam no registro sao anexadas ao fim em ordem
    alfabetica, para que uma revisao futura do dataset nao perca tabelas em
    silencio.
    """
    keys = list(TABLE_KEYS[task])
    keys += sorted(
        k
        for k, v in record.items()
        if k not in keys and isinstance(v, list) and k not in ("stock_name", "report_id")
    )
    tables: list[Table] = []
    for key in keys:
        for cells in record.get(key) or []:
            tables.append(Table(index=len(tables), cells=cells))
    return tables


def load_documents(
    root: Path | str,
    task: Task,
    split: str,
    *,
    limit: int | None = None,
    with_tables: bool = True,
) -> list[Document]:
    """Carrega e remonta os documentos de um split.

    Args:
        root: diretorio com `text/` e `table/` (por padrao `data/raw/findsum`).
        task: ROO ou Liquidity.
        split: `train`, `val` ou `test`.
        limit: le apenas as primeiras N linhas (util para testes rapidos).
        with_tables: tambem carrega o arquivo de tuplas de tabelas.

    Raises:
        FileNotFoundError: se algum arquivo necessario nao foi baixado.
        ValueError: se os arquivos de segmento tiverem numeros de linha distintos.
    """
    if split not in SPLITS:
        raise ValueError(f"split invalido: {split!r}; use um de {SPLITS}")

    paths = FindSumPaths(root)
    missing = paths.missing(task, split, need_tables=with_tables)
    if missing:
        listed = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"arquivos do FINDSum ausentes:\n  {listed}\n"
            "rode: python scripts/fetch_findsum.py"
        )

    frames = [
        pd.read_csv(p, sep=",", nrows=limit, usecols=["document", "summary"])
        for p in paths.segment_files(task, split)
    ]
    sizes = {len(f) for f in frames}
    if len(sizes) != 1:
        raise ValueError(
            f"arquivos de segmento de {task.value}/{split} tem tamanhos diferentes: {sizes}"
        )
    n_rows = sizes.pop()

    meta: list[dict] = []
    if with_tables:
        meta = list(_iter_table_records(paths.table_file(task, split), limit=limit))
        if len(meta) < n_rows:
            raise ValueError(
                f"arquivo de tabelas de {task.value}/{split} tem {len(meta)} linhas, "
                f"mas os CSVs tem {n_rows}"
            )

    documents: list[Document] = []
    for row in range(n_rows):
        segments = [
            Segment(
                index=i,
                document=_as_text(frame["document"].iloc[row]),
                summary=_as_text(frame["summary"].iloc[row]),
            )
            for i, frame in enumerate(frames)
        ]
        record = meta[row] if row < len(meta) else {}
        stock = record.get("stock_name")
        report = record.get("report_id")
        doc_id = f"{stock}-{report}" if stock and report else f"{task.value}-{split}-{row:05d}"
        documents.append(
            Document(
                doc_id=doc_id,
                task=task,
                split=split,
                row=row,
                segments=segments,
                stock_name=stock,
                report_id=report,
                tables=_tables_from_record(record, task) if record else [],
            )
        )
    return documents


def _as_text(value: object) -> str:
    """Converte celula do CSV em texto, tratando NaN de campos vazios."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def strip_table_tokens(text: str, replacement: str = " ") -> str:
    """Remove os marcadores `replace_table_token_<i>_th` do texto."""
    return re.sub(r"\s*replace_table_token_\d+_th\s*", replacement, text).strip()


def clean_text(text: str) -> str:
    """Remove marcadores do dataset e normaliza espacos em branco."""
    text = text.replace(SEGMENT_SEPARATOR, " ")
    text = strip_table_tokens(text)
    return re.sub(r"\s+", " ", text).strip()
