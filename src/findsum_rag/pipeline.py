"""Orquestracao do experimento: prepara dados, roda as configuracoes, avalia.

O fluxo de uma rodada e:

1. carregar os documentos de avaliacao e a base de exemplos (splits distintos);
2. embutir cada documento para selecionar exemplos e a instrucao da tarefa
   para recuperar evidencia, compartilhando ambos entre configuracoes;
3. para cada configuracao, montar o contexto, gerar e pontuar;
4. agregar e gravar previsoes, metricas por documento e resumo por configuracao.

Os embeddings e os trechos sao calculados uma vez e compartilhados entre as
configuracoes: alem de economizar tempo, isso garante que C2 a C5 vejam
exatamente o mesmo contexto recuperado, isolando o efeito dos exemplos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analyze import ANALYSIS_PLAN
from .chunking import Chunk, chunk_document
from .config import ExperimentArm, ExperimentConfig
from .context import matched_contexts, token_count
from .data import Document, clean_text
from .examples import Example, ExampleStore, make_selector
from .generate import Generation, PromptTokenizer
from .metrics import (
    DocumentScores,
    RougeScorer,
    aggregate,
    bertscore,
    score_document,
)
from .prompts import TASK_INSTRUCTIONS, build_prompt, format_context
from .retrieval import Encoder, SentenceTransformerEncoder, VectorIndex
from .splits import SplitManifest, load_set


@dataclass
class PreparedDocument:
    """Um documento com tudo o que as configuracoes precisam, calculado uma vez."""

    document: Document
    chunks: list[Chunk]
    query_vector: np.ndarray
    chunk_vectors: np.ndarray
    retrieval_vector: np.ndarray
    contexts: dict[str, str] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.document.doc_id

    @property
    def source_text(self) -> str:
        """Fonte disponivel: prosa limpa e tabelas efetivamente serializadas."""
        return "\n\n".join(
            [clean_text(self.document.document)]
            + [c.text for c in self.chunks if c.source == "table"]
        )


def prepare_documents(
    documents: list[Document],
    encoder: Encoder,
    *,
    chunk_size: int,
    chunk_overlap: int,
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
            raise ValueError(f"documento sem evidencia: {document.doc_id}")
        chunk_vectors = encoder.encode_texts([c.text for c in chunks])
        query_text = clean_text(document.document)
        query_vector = encoder.encode_texts([query_text])[0]
        prepared.append(
            PreparedDocument(
                document=document,
                chunks=chunks,
                query_vector=query_vector,
                chunk_vectors=chunk_vectors,
                retrieval_vector=encoder.encode_texts([TASK_INSTRUCTIONS[document.task]])[0],
            )
        )
    return prepared


def select_context(prepared: PreparedDocument, arm: ExperimentArm, top_k: int) -> list[Chunk]:
    """Seleciona candidatos; o orcamento em tokens e aplicado em prepare_contexts."""
    if arm.context_mode in {"full", "truncated"}:
        return [Chunk(prepared.doc_id, 0, prepared.source_text)]

    index = VectorIndex(int(prepared.chunk_vectors.shape[1]))
    index.add(prepared.chunk_vectors, list(prepared.chunks))
    hits = index.search(prepared.retrieval_vector, top_k)
    chosen: list[Chunk] = [h.payload for h in hits]  # type: ignore[misc]
    # Prioridade semantica antes do corte pelo orcamento; nao pela posicao na fonte.
    return chosen


def build_example_store(
    root: Path | str,
    manifest: SplitManifest,
    name: str,
    *,
    limit: int | None = None,
    max_summary_words: int | None = None,
) -> ExampleStore:
    """Monta a base de exemplos few-shot a partir de um conjunto do manifesto.

    `with_tables=False` na carga: o arquivo de tuplas do split de treino tem
    ~2 GB e aqui so interessam texto e resumo -- os `doc_id` vem do manifesto.

    `max_summary_words=None` (padrao) mantem o resumo integral. A extensao-alvo
    e parte do que a demonstracao ensina: truncar o resumo do exemplo ensinaria o
    modelo a parar cedo. O documento completo e preservado no indice; so sua
    apresentacao no prompt e abreviada, o que exige auditar suporte no piloto.

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
                document=clean_text(document.document),
                summary=summary,
            )
        )
    return ExampleStore(examples)


def prepare_contexts(item: PreparedDocument, config: ExperimentConfig, summarizer) -> None:
    """Calcula uma unica evidencia RAG compartilhada por C2--C5."""
    summarizer.load_tokenizer()
    retrieved_arm = ExperimentArm(
        id="budget", label="budget", context_mode="retrieved", example_strategy="none"
    )
    chunks = select_context(item, retrieved_arm, config.retrieval.top_k)

    def count(text):
        return summarizer.count_tokens(
            build_prompt(
                task=item.document.task,
                arm=retrieved_arm,
                context_chunks=[],
                context_text=text,
                examples=[],
            )
        )

    truncated, retrieved = matched_contexts(
        summarizer.tokenizer,
        item.source_text,
        format_context(chunks),
        config.retrieval.context_max_tokens,
        count,
    )
    item.contexts = {"full": item.source_text, "truncated": truncated, "retrieved": retrieved}


def arm_prompt(item, arm, config, summarizer, selector):
    if not item.contexts:
        prepare_contexts(item, config, summarizer)
    context = item.contexts[arm.context_mode]
    examples = selector.select(item.doc_id, item.query_vector)
    chunks = select_context(item, arm, config.retrieval.top_k)
    if arm.use_rag:
        included, offset = [], 0
        for i, chunk in enumerate(chunks, 1):
            if offset >= len(context):
                break
            included.append(chunk)
            offset += len(f"[{i}] {chunk.text}") + 2
        chunks = included
    prompt = build_prompt(
        task=item.document.task,
        arm=arm,
        context_chunks=chunks,
        context_text=context,
        examples=examples,
        example_max_words=config.data.example_max_words,
    )
    count = summarizer.count_tokens(prompt)
    if count > config.generation.max_input_tokens:
        raise ValueError(
            f"{arm.id}/{item.doc_id}: {count} tokens excedem max_input_tokens; "
            "ajuste a configuracao no dev antes de gerar"
        )
    return prompt, examples, context


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
    summarizer: PromptTokenizer,
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
        prompt, examples, context = arm_prompt(item, arm, config, summarizer, selector)
        generation: Generation = summarizer.generate(prompt)
        if generation.truncated_prompt:
            raise ValueError("gerador truncou um prompt; rodada invalida")
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
                "context": context,
                "source": item.source_text,
                "context_tokens": token_count(summarizer.tokenizer, context),
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "truncated_prompt": generation.truncated_prompt,
                "generation_metadata": generation.metadata,
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
    preflight_only: bool = False,
) -> dict[str, ArmResult]:
    """Executa a rodada completa e grava os artefatos em `config.output_dir`."""
    if config.data.eval_set == "eval":
        raise ValueError("full somente pelo executor OpenRouter congelado: findsum run")
    out = Path(config.output_dir) / config.name
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"diretorio de rodada nao vazio: {out}; use um nome novo")
    config.to_yaml(out / "config.yaml")
    (out / "analysis_plan.json").write_text(json.dumps(ANALYSIS_PLAN, indent=2))

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
            max_summary_words=config.data.example_max_summary_words,
        )
        if any(a.example_strategy == "dynamic" for a in selected):
            store.build_index(encoder)

    from .openrouter import OpenRouterSummarizer

    summarizer = OpenRouterSummarizer(config.generation, audit_dir=out / "api")
    # Valida TODOS os prompts antes da primeira geracao.
    for item in prepared:
        prepare_contexts(item, config, summarizer)
    preflight = []
    for arm in selected:
        selector = make_selector(
            arm.example_strategy, store=store, n_examples=arm.n_examples, seed=config.seed
        )
        for item in prepared:
            prompt, examples, context = arm_prompt(item, arm, config, summarizer, selector)
            preflight.append(
                {
                    "arm": arm.id,
                    "doc_id": item.doc_id,
                    "example_ids": [e.doc_id for e in examples],
                    "context": context,
                    "context_tokens": token_count(summarizer.tokenizer, context),
                    "prompt_tokens": summarizer.count_tokens(prompt),
                    "messages": prompt.as_messages(),
                }
            )
    (out / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if preflight_only:
        return {}
    rouge = RougeScorer()

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

    (arm_dir / "summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
