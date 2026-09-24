"""Orquestracao do experimento: prepara dados, roda as configuracoes, avalia.

O fluxo de uma rodada e:

1. carregar os documentos de avaliacao e a base de exemplos (splits distintos);
2. embutir cada documento uma unica vez (vetor de consulta reaproveitado por
   todas as configuracoes);
3. para cada configuracao, montar o contexto, gerar e pontuar;
4. agregar e gravar previsoes, metricas por documento e resumo por configuracao.

Os embeddings e os trechos sao calculados uma vez e compartilhados entre as
configuracoes: alem de economizar tempo, isso garante que C2 a C5 vejam
exatamente o mesmo contexto recuperado, isolando o efeito dos exemplos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_document
from .config import ExperimentArm, ExperimentConfig
from .data import Document, clean_text
from .examples import Example, ExampleStore, make_selector
from .generate import Generation, Summarizer
from .metrics import (
    DocumentScores,
    RougeScorer,
    aggregate,
    bertscore,
    reference_baseline,
    score_document,
)
from .prompts import build_prompt
from .retrieval import Encoder, SentenceTransformerEncoder, VectorIndex
from .splits import SplitManifest, load_set


@dataclass
class PreparedDocument:
    """Um documento com tudo o que as configuracoes precisam, calculado uma vez."""

    document: Document
    chunks: list[Chunk]
    query_vector: np.ndarray
    chunk_vectors: np.ndarray

    @property
    def doc_id(self) -> str:
        return self.document.doc_id

    @property
    def source_text(self) -> str:
        """Texto limpo do documento, usado como referencia de fidelidade."""
        return clean_text(self.document.document)


def prepare_documents(
    documents: list[Document],
    encoder: Encoder,
    *,
    chunk_size: int,
    chunk_overlap: int,
    query_words: int,
    include_tables: bool,
) -> list[PreparedDocument]:
    """Fatia e codifica os documentos de avaliacao."""
    prepared: list[PreparedDocument] = []
    for document in documents:
        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            include_tables=include_tables,
        )
        if not chunks:
            continue
        chunk_vectors = encoder.encode_texts([c.text for c in chunks])
        query_text = " ".join(clean_text(document.document).split()[:query_words])
        query_vector = encoder.encode_texts([query_text])[0]
        prepared.append(
            PreparedDocument(
                document=document,
                chunks=chunks,
                query_vector=query_vector,
                chunk_vectors=chunk_vectors,
            )
        )
    return prepared


def select_context(prepared: PreparedDocument, arm: ExperimentArm, top_k: int) -> list[Chunk]:
    """Escolhe os trechos que entram no prompt.

    Tres modos, e a diferenca entre eles e o fator experimental:

    * `full`      -- o documento inteiro. Possivel porque ele cabe na janela
      (~7,3 mil tokens contra 262 mil), logo e o baseline realista.
    * `retrieved` -- os `top_k` trechos mais similares ao documento (RAG).
    * `truncated` -- os `top_k` primeiros trechos na ordem original. Mesmo
      orcamento de `retrieved`, sem recuperacao: isola quanto do efeito vem de
      recuperar e quanto vem so de reduzir o contexto.

    Nos dois modos limitados os trechos voltam em ordem de leitura do relatorio,
    nao por score: a sequencia carrega informacao (fluxo narrativo, ordem
    temporal) que a ordenacao por similaridade destruiria.
    """
    if arm.uses_full_document:
        return list(prepared.chunks)

    if arm.context_mode == "truncated":
        return prepared.chunks[:top_k]

    index = VectorIndex(int(prepared.chunk_vectors.shape[1]))
    index.add(prepared.chunk_vectors, list(prepared.chunks))
    hits = index.search(prepared.query_vector, top_k)
    chosen: list[Chunk] = [h.payload for h in hits]  # type: ignore[misc]
    chosen.sort(key=lambda c: c.index)
    return chosen


def build_example_store(
    root: Path | str,
    manifest: SplitManifest,
    name: str,
    *,
    limit: int | None = None,
    max_words: int = 350,
    max_summary_words: int | None = None,
) -> ExampleStore:
    """Monta a base de exemplos few-shot a partir de um conjunto do manifesto.

    `with_tables=False` na carga: o arquivo de tuplas do split de treino tem
    ~2 GB e aqui so interessam texto e resumo -- os `doc_id` vem do manifesto.

    `max_summary_words=None` (padrao) mantem o resumo integral. A extensao-alvo
    e parte do que a demonstracao ensina: truncar o resumo do exemplo ensinaria o
    modelo a parar cedo. O custo e de contexto, nao de fidelidade.

    O `doc_id` vem do MANIFESTO, nao do documento carregado: sem as tabelas o
    `Document` nao tem `stock_name`/`report_id` e cairia num id sintetico por
    linha. Usar o id real mantem a exclusao do proprio documento efetiva e torna
    os `example_ids` gravados em `predictions.jsonl` rastreaveis.
    """
    refs = manifest.sets[name][:limit] if limit is not None else manifest.sets[name]
    documents = load_set(root, manifest, name, limit=limit, with_tables=False)
    examples: list[Example] = []
    for ref, document in zip(refs, documents, strict=True):
        summary = clean_text(document.summary)
        if not summary:
            continue
        if max_summary_words:
            summary = " ".join(summary.split()[:max_summary_words])
        examples.append(
            Example(
                doc_id=ref.doc_id,
                document=" ".join(clean_text(document.document).split()[:max_words]),
                summary=summary,
            )
        )
    return ExampleStore(examples)


@dataclass
class ArmResult:
    """Resultado de uma configuracao sobre o conjunto de avaliacao."""

    arm: ExperimentArm
    scores: list[DocumentScores]
    predictions: list[dict]
    summary: dict[str, float]


def run_arm(
    arm: ExperimentArm,
    prepared: list[PreparedDocument],
    *,
    config: ExperimentConfig,
    summarizer: Summarizer,
    store: ExampleStore | None,
    rouge: RougeScorer,
) -> ArmResult:
    """Roda uma configuracao sobre todos os documentos preparados."""
    selector = make_selector(
        arm.example_strategy,
        store=store,
        n_examples=arm.n_examples,
        seed=config.seed,
    )

    predictions: list[dict] = []
    scores: list[DocumentScores] = []

    for item in prepared:
        context = select_context(item, arm, config.retrieval.top_k)
        examples = selector.select(item.doc_id, item.query_vector)
        prompt = build_prompt(
            task=item.document.task,
            arm=arm,
            context_chunks=context,
            examples=examples,
            example_max_words=config.data.example_max_words,
        )
        generation: Generation = summarizer.generate(prompt)
        reference = clean_text(item.document.summary)

        scores.append(
            score_document(
                item.doc_id,
                generation.text,
                reference,
                item.source_text,
                rouge=rouge,
            )
        )
        predictions.append(
            {
                "arm": arm.id,
                "doc_id": item.doc_id,
                "stock_name": item.document.stock_name,
                "report_id": item.document.report_id,
                "prediction": generation.text,
                "reference": reference,
                "example_ids": [e.doc_id for e in examples],
                "n_context_chunks": prompt.n_context_chunks,
                "context_words": prompt.context_words,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "truncated_prompt": generation.truncated_prompt,
            }
        )

    return ArmResult(arm=arm, scores=scores, predictions=predictions, summary=aggregate(scores))


def truncation_report(result: ArmResult) -> str | None:
    """Aviso quando prompts foram truncados, ou `None` se nenhum foi.

    Truncamento silencioso invalidaria a comparacao: a configuracao afetada veria
    menos contexto que as outras, e a diferenca de desempenho seria atribuida a
    tecnica quando veio do orcamento de tokens. As configuracoes com few-shot sao
    as mais expostas, porque os exemplos competem com o contexto pelo mesmo
    espaco.
    """
    affected = [p["doc_id"] for p in result.predictions if p["truncated_prompt"]]
    if not affected:
        return None
    total = len(result.predictions)
    return (
        f"ATENCAO {result.arm.id}: {len(affected)}/{total} prompts truncados por "
        f"max_input_tokens. Aumente generation.max_input_tokens, reduza "
        f"retrieval.top_k ou data.example_max_words antes de interpretar os "
        f"resultados desta configuracao."
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    arms: list[str] | None = None,
    with_bertscore: bool = True,
) -> dict[str, ArmResult]:
    """Executa a rodada completa e grava os artefatos em `config.output_dir`."""
    out = Path(config.output_dir) / config.name
    out.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out / "config.yaml")

    encoder = SentenceTransformerEncoder(config.retrieval.embedding_model)

    manifest = SplitManifest.load(config.data.manifest)
    if manifest.task != config.data.task:
        raise ValueError(
            f"o manifesto e da tarefa {manifest.task.value}, mas a configuracao "
            f"pede {config.data.task.value}"
        )

    documents = load_set(
        config.data.root,
        manifest,
        config.data.eval_set,
        limit=config.data.n_eval_docs,
    )
    prepared = prepare_documents(
        documents,
        encoder,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
        query_words=config.retrieval.query_words,
        include_tables=config.retrieval.include_tables,
    )

    selected = [a for a in config.arms if arms is None or a.id in arms]
    needs_examples = any(a.example_strategy != "none" for a in selected)
    store: ExampleStore | None = None
    if needs_examples:
        store = build_example_store(
            config.data.root,
            manifest,
            config.data.example_set,
            limit=config.data.n_example_docs,
            max_words=config.data.example_max_words,
            max_summary_words=config.data.example_max_summary_words,
        )
        if any(a.example_strategy == "dynamic" for a in selected):
            store.build_index(encoder, query_words=config.retrieval.query_words)

    summarizer = Summarizer(config.generation)
    rouge = RougeScorer()

    # Calibracao: sem o patamar da propria referencia as metricas de ancoragem
    # nao sao interpretaveis (ver metrics.reference_baseline).
    baseline = reference_baseline(
        [clean_text(p.document.summary) for p in prepared],
        [p.source_text for p in prepared],
    )
    (out / "reference_baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )

    results: dict[str, ArmResult] = {}
    warnings: list[str] = []
    for arm in selected:
        result = run_arm(
            arm,
            prepared,
            config=config,
            summarizer=summarizer,
            store=store,
            rouge=rouge,
        )

        warning = truncation_report(result)
        if warning:
            warnings.append(warning)
            print(warning, flush=True)

        if with_bertscore and result.predictions:
            f1s = bertscore(
                [p["prediction"] for p in result.predictions],
                [p["reference"] for p in result.predictions],
            )
            for score, f1 in zip(result.scores, f1s, strict=True):
                score.bertscore = f1
            result.summary = aggregate(result.scores)

        _write_arm(out, result)
        results[arm.id] = result

    (out / "summary.json").write_text(
        json.dumps(
            {
                "reference_baseline": baseline,
                "warnings": warnings,
                "arms": {k: v.summary for k, v in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def _write_arm(out: Path, result: ArmResult) -> None:
    """Grava previsoes e metricas por documento de uma configuracao."""
    import csv

    arm_dir = out / result.arm.id
    arm_dir.mkdir(parents=True, exist_ok=True)

    with (arm_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in result.predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = [s.flat() for s in result.scores]
    if rows:
        with (arm_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    (arm_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )
