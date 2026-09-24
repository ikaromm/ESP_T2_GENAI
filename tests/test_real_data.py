"""Testes contra o FINDSum real, pulados quando o download nao esta presente.

Estes testes existem para travar as suposicoes sobre o formato do dataset. Se uma
versao futura dos arquivos mudar a estrutura de segmentos, o alinhamento com as
tabelas ou a escala dos resumos, e aqui que isso aparece -- e nao no meio de uma
rodada experimental de horas.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from findsum_rag.chunking import chunk_document
from findsum_rag.data import FindSumPaths, Task, clean_text, load_documents

ROOT = Path("data/raw/findsum")
TASK = Task.LIQUIDITY
SPLIT = "val"

pytestmark = pytest.mark.skipif(
    bool(FindSumPaths(ROOT).missing(TASK, SPLIT)),
    reason="FINDSum nao baixado; rode python scripts/fetch_findsum.py",
)


@pytest.fixture(scope="module")
def documents():
    return load_documents(ROOT, TASK, SPLIT, limit=50)


def test_loads_expected_number_of_segments(documents):
    assert len(documents) == 50
    assert all(len(d.segments) == TASK.n_segments for d in documents)


def test_doc_ids_are_unique_and_use_sec_identifiers(documents):
    ids = [d.doc_id for d in documents]
    assert len(ids) == len(set(ids))
    for document in documents:
        assert document.stock_name, "stock_name deve vir do arquivo de tabelas"
        assert document.report_id, "report_id deve vir do arquivo de tabelas"
        # report_id e um accession number da SEC: 10-2-6 digitos.
        assert len(document.report_id.split("-")) == 3
        assert document.doc_id == f"{document.stock_name}-{document.report_id}"


def test_input_length_matches_segment_budget(documents):
    # Cada segmento traz ~2000 palavras; 3 segmentos => ~6000 palavras.
    words = [d.n_words for d in documents]
    assert min(words) > 4500, f"entrada curta demais: {min(words)}"
    assert max(words) < 7000, f"entrada longa demais: {max(words)}"


def test_reference_summaries_are_long(documents):
    # Justifica max_new_tokens >= ~1300 tokens na GenerationConfig.
    words = [d.n_summary_words for d in documents]
    assert min(words) > 200
    assert sum(words) / len(words) > 500


def test_summary_is_concatenation_of_segment_parts(documents):
    document = documents[0]
    parts = [s.summary for s in document.segments]
    assert document.summary == " ".join(parts)
    # Cada segmento contribui com conteudo proprio.
    assert all(p.strip() for p in parts)
    assert len(set(parts)) == len(parts)


def test_table_metadata_aligns_with_text_content(documents):
    """O `stock_name` do arquivo de tabelas deve descrever o texto da mesma linha.

    Verificacao indireta do alinhamento por indice de linha entre os CSVs de
    texto e o arquivo de tuplas: se estivessem desalinhados, os nomes de linha
    das tabelas citadas nao teriam relacao com o vocabulario do relatorio.
    """
    document = next(d for d in documents if d.referenced_tables())
    text = clean_text(document.document).lower()
    row_names = {
        str(cell[0]).lower()
        for table in document.referenced_tables()
        for cell in table.cells
        if cell and cell[0]
    }
    assert row_names, "as tabelas citadas devem ter nomes de linha"
    hits = sum(1 for name in row_names if name and name in text)
    assert hits > 0, "nenhum nome de linha da tabela aparece no texto do relatorio"


def test_table_indices_are_document_global(documents):
    """Os marcadores nao reiniciam a numeracao por segmento.

    E essa propriedade que revela que os segmentos sao selecoes de conteudo, e
    nao janelas contiguas do relatorio.
    """
    overlapping = 0
    for document in documents:
        per_segment = []
        for segment in document.segments:
            from findsum_rag.data import TABLE_TOKEN_RE

            per_segment.append({int(m.group(1)) for m in TABLE_TOKEN_RE.finditer(segment.document)})
        for i in range(len(per_segment)):
            for j in range(i + 1, len(per_segment)):
                if per_segment[i] & per_segment[j]:
                    overlapping += 1
    assert overlapping > 0, (
        "esperava indices de tabela repetidos entre segmentos; se isso mudou, "
        "revise a documentacao do formato em docs/dataset.md"
    )


def test_reference_summaries_are_not_fully_grounded(documents):
    """O resumo de referencia NAO esta inteiramente contido no texto distribuido.

    Trava o achado que calibra as metricas de ancoragem: parte relevante dos
    numeros do resumo humano nao aparece no extrato de ~6000 palavras, porque o
    FINDSum aplica selecao de conteudo a montante. Isso impoe um teto pratico ao
    `numeric_grounding` de qualquer modelo que use apenas este texto.
    """
    from findsum_rag.metrics import reference_baseline

    baseline = reference_baseline(
        [clean_text(d.summary) for d in documents],
        [clean_text(d.document) for d in documents],
    )
    numeric = baseline["reference_numeric_grounding_mean"]
    assert 0.3 < numeric < 0.9, (
        f"ancoragem numerica da referencia = {numeric:.3f}; se isso mudou muito, "
        "revise a calibracao descrita em docs/dataset.md"
    )
    # Resumos abstrativos: sobreposicao de 4-gramas baixa por construcao.
    ngram = baseline["reference_ngram_grounding_mean"]
    assert ngram < 0.5, (
        f"extratividade da referencia = {ngram:.3f}; um valor alto indicaria "
        "resumos extrativos, o que mudaria a leitura da metrica"
    )


def test_table_keys_match_the_real_files():
    """As chaves declaradas em `TABLE_KEYS` existem de fato nos arquivos.

    Este teste existe porque a suposicao obvia estava errada: a tarefa ROO usa
    `mda_result_tables`, nao `mda_roo_tables`. Sem esta verificacao a chave
    especifica da tarefa era ignorada em silencio e a numeracao das tabelas da
    ROO saia trocada.
    """
    import json

    from findsum_rag.data import TABLE_KEYS

    for task in (Task.ROO, Task.LIQUIDITY):
        path = FindSumPaths(ROOT).table_file(task, "val")
        if not path.exists():
            pytest.skip(f"arquivo de tabelas ausente: {path}")
        with path.open(encoding="utf-8") as handle:
            record = json.loads(handle.readline())

        present = {k for k, v in record.items() if isinstance(v, list)}
        declared = set(TABLE_KEYS[task])
        assert declared == present, (
            f"{task.value}: chaves declaradas {sorted(declared)} != "
            f"chaves do arquivo {sorted(present)}"
        )
        assert task.table_key in present, f"{task.value}: table_key ausente do arquivo"


def test_roo_task_loads_with_two_segments():
    """A ROO tem 2 segmentos e resumos mais curtos que a Liquidity."""
    if FindSumPaths(ROOT).missing(Task.ROO, "val"):
        pytest.skip("split val da ROO nao baixado")
    documents = load_documents(ROOT, Task.ROO, "val", limit=20)
    assert all(len(d.segments) == 2 for d in documents)
    words = [d.n_words for d in documents]
    assert min(words) > 3000 and max(words) < 5000
    assert all(d.stock_name and d.report_id for d in documents)
    # As tabelas especificas da tarefa sao encontradas.
    assert any(d.referenced_tables() for d in documents)


def test_chunking_real_document_is_clean(documents):
    chunks = chunk_document(documents[0], chunk_size=220, overlap=40)
    assert len(chunks) > 10
    text_chunks = [c for c in chunks if c.source == "text"]
    assert text_chunks
    assert all("replace_table_token" not in c.text for c in text_chunks)
    assert all("story_separator_special_tag" not in c.text for c in text_chunks)
    assert all(c.n_words <= 220 for c in text_chunks)
    assert any(c.source == "table" for c in chunks)


def test_train_split_available_for_example_store():
    """A base de exemplos precisa do split de treino, que e o arquivo maior."""
    missing = FindSumPaths(ROOT).missing(TASK, "train", need_tables=False)
    if missing:
        pytest.skip(f"split de treino ainda nao baixado: {missing}")
    documents = load_documents(ROOT, TASK, "train", limit=5, with_tables=False)
    assert len(documents) == 5
    assert all(d.summary.strip() for d in documents)
