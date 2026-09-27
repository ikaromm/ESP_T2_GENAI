"""Preparacao de prompts OpenRouter; embeddings locais, nenhuma chamada de geracao."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from findsum_rag.analyze import ANALYSIS_PLAN
from findsum_rag.config import ExperimentConfig, GenerationConfig
from findsum_rag.context import token_count
from findsum_rag.examples import make_selector
from findsum_rag.generate import PromptTokenizer
from findsum_rag.pipeline import (
    arm_prompt,
    build_example_store,
    prepare_contexts,
    prepare_documents,
    select_context,
)
from findsum_rag.retrieval import SentenceTransformerEncoder
from findsum_rag.splits import DocumentRef, SplitManifest, load_set
from scripts.common.full_common import Limits, random_records
from scripts.experiments.screen_openrouter import save


class PreparationTokenizer(PromptTokenizer):
    """Somente tokenizer oficial; carregamento de pesos e geracao proibidos."""

    def __init__(self, config, tokenizer_name=None, revision=None):
        super().__init__(config)
        self.tokenizer_name = tokenizer_name or config.model_name
        self.revision = revision

    def load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, revision=self.revision, trust_remote_code=False
            )
            if not self._tokenizer.is_fast or not self._tokenizer.chat_template:
                raise ValueError("tokenizer fast/chat template obrigatorios")

    def render(self, prompt):
        return self.tokenizer.apply_chat_template(
            prompt.as_messages(),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def load(self):
        raise RuntimeError("preparacao nao carrega pesos de geracao")

    def generate(self, prompt):
        raise RuntimeError("preparacao nunca envia geracoes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dev_openrouter_ling.yaml"))
    parser.add_argument("--full", action="store_true", help="prepara candidatos eval, sem gerar")
    parser.add_argument("--models", nargs="+", choices=["ling-free", "qwen37", "gemma26"])
    args = parser.parse_args()
    frozen = None
    if args.full:
        from findsum_rag.full_lock import verify_full_lock

        frozen = verify_full_lock()
        if args.config.read_bytes() != Path("configs/full_openrouter.yaml").read_bytes():
            raise ValueError("full exige o perfil congelado")
    args.output.mkdir(parents=True, exist_ok=False)
    if frozen:
        save(args.output / "full-lock.json", frozen)
    plan = json.loads(Path("configs/openrouter_full.json").read_text())
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.to_yaml(args.output / "config.yaml")
    input_limit = cfg.generation.max_input_tokens
    split = "eval" if args.full else "dev"
    count = cfg.data.n_eval_docs
    output_limit = cfg.generation.max_new_tokens
    if cfg.data.eval_set != split or count is None or count <= 0:
        raise ValueError("configuracao e split de preparacao invalidos")
    if args.full and count != 1000:
        raise ValueError("full exige os 1000 documentos reservados")
    manifest = SplitManifest.load(cfg.data.manifest)
    if args.full and len(manifest.sets[split]) != 1000:
        raise ValueError("manifesto eval deve conter exatamente 1000 documentos")
    save(args.output / "analysis_plan.json", ANALYSIS_PLAN)
    manifest_hash = hashlib.sha256(cfg.data.manifest.read_bytes()).hexdigest()
    selected = random_records(
        [r.model_dump() for r in manifest.sets[split]], count, seed=cfg.seed, split=split
    )
    save(
        args.output / "selection.json",
        {
            "manifest_sha256": manifest_hash,
            "split": split,
            "manifest_path": str(cfg.data.manifest),
            "documents": selected,
            "sampling": {
                "method": "random_without_replacement",
                "seed": cfg.seed,
                "seed_namespace": f"cohort:{cfg.seed}:{split}",
                "population": len(manifest.sets[split]),
            },
            "source_policy": "prose_and_raw_task_tables_no_financial_filter",
            "examples": [r.model_dump() for r in manifest.sets["examples"]],
            "eval_documents_loaded": len(selected) if args.full else 0,
        },
    )
    with (args.output / "selected-cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["order", "doc_id", "stock_name", "report_id", "split", "row"]
        )
        writer.writeheader()
        writer.writerows({"order": i, **ref} for i, ref in enumerate(selected, 1))
    # Somente copia em memoria: o manifesto original em disco permanece intacto.
    sampled_manifest = manifest.model_copy(deep=True)
    sampled_manifest.sets[split] = [DocumentRef(**r) for r in selected]
    documents = load_set(cfg.data.root, sampled_manifest, split)
    encoder = SentenceTransformerEncoder(cfg.retrieval.embedding_model)
    print(f"Preparando {len(documents)} documentos de {split} e exemplos completos", flush=True)
    prepared = prepare_documents(
        documents,
        encoder,
        chunk_size=cfg.retrieval.chunk_size,
        chunk_overlap=cfg.retrieval.chunk_overlap,
        include_tables=cfg.retrieval.include_tables,
    )
    save(
        args.output / "table-audit.json",
        [
            {
                "doc_id": doc.doc_id,
                "tables": [
                    {
                        "section": table.section,
                        "section_index": table.section_index,
                        "cells": table.evidence_audit(),
                    }
                    for table in doc.tables
                    if table.section == doc.task.table_key
                ],
            }
            for doc in documents
        ],
    )
    store = build_example_store(cfg.data.root, manifest, "examples", limit=None)
    store.build_index(encoder)
    save(args.output / "examples.json", [asdict(e) for e in store.examples])
    selectors = {
        arm.id: make_selector(
            arm.example_strategy, store=store, n_examples=arm.n_examples, seed=cfg.seed
        )
        for arm in cfg.arms
    }
    rag_arm = next(a for a in cfg.arms if a.id == "C2")
    common = [
        {
            "doc_id": item.doc_id,
            "source": item.source_text,
            "reference": item.document.summary,
            "raw_document": asdict(item.document),
            "chunks": [asdict(c) for c in item.chunks],
            "retrieved_candidates": [
                asdict(c) for c in select_context(item, rag_arm, cfg.retrieval.top_k)
            ],
            "example_ids": {
                a.id: [e.doc_id for e in selectors[a.id].select(item.doc_id, item.query_vector)]
                for a in cfg.arms
            },
        }
        for item in prepared
    ]
    save(args.output / "documents.json", common)
    report = {
        "scope": "preparation_" + split,
        "documents": len(common),
        "examples": len(store),
        "generation_calls": 0,
        "models": {},
    }
    for name, target in plan["models"].items():
        if args.models and name not in args.models:
            continue
        folder = args.output / name
        folder.mkdir()
        save(
            folder / "endpoint-plan.json",
            target | plan["generation"] | {"max_input_tokens": input_limit},
        )
        if target["tokenizer"] is None:
            report["models"][name] = {
                "status": "blocked_tokenizer",
                "documents_prepared": len(documents),
                "prompts_validated": 0,
                "common_inputs": "../documents.json",
            }
            save(args.output / "report.json", report)
            continue
        # Teto de preparacao cabe na janela do endpoint, sem truncar C1.
        cfg.generation = GenerationConfig(
            model_name=target["model"],
            max_input_tokens=min(input_limit, target["context_length"] - output_limit - 256),
            max_new_tokens=output_limit,
            load_in_4bit=False,
        )
        tokenizer = PreparationTokenizer(
            cfg.generation, target["tokenizer"], target.get("tokenizer_revision")
        )
        rows, errors = [], []
        try:
            limits = Limits(cfg.generation.max_input_tokens, output_limit, target["context_length"])
            tokenizer.load_tokenizer()
            tokenizer.tokenizer.save_pretrained(folder / "tokenizer")
            for item in prepared:
                try:
                    prepare_contexts(item, cfg, tokenizer)
                    doc_rows = []
                    for arm in cfg.arms:
                        prompt, examples, context = arm_prompt(
                            item, arm, cfg, tokenizer, selectors[arm.id]
                        )
                        limits.check(tokenizer.count_tokens(prompt))
                        doc_rows.append(
                            {
                                "doc_id": item.doc_id,
                                "arm": arm.id,
                                "messages": prompt.as_messages(),
                                "context": context,
                                "example_ids": [e.doc_id for e in examples],
                                "context_tokens": token_count(tokenizer.tokenizer, context),
                                "prompt_tokens": tokenizer.count_tokens(prompt),
                                "max_output_tokens": output_limit,
                                "allow_length": True,
                            }
                        )
                    rows.extend(doc_rows)
                except ValueError as exc:
                    errors.append({"doc_id": item.doc_id, "error": str(exc)})
            save(folder / "preflight.json", rows)
            maximum = max((r["prompt_tokens"] for r in rows), default=0)
            input_total = sum(r["prompt_tokens"] for r in rows)
            worst = (
                input_total * target["input_per_million"]
                + len(rows) * output_limit * target["output_per_million"]
            ) / 1e6
            report["models"][name] = {
                "status": "preflight_passed"
                if len(rows) == len(documents) * len(cfg.arms) and not errors
                else "incomplete",
                "provider_contract_validated": False,
                "prompts_validated": len(rows),
                "max_prompt_tokens": maximum,
                "input_tokens_total": input_total,
                "cost_max_without_retries_usd": worst,
                "errors": errors,
            }
            print(name, report["models"][name], flush=True)
        except Exception as exc:
            report["models"][name] = {"status": "blocked", "error": str(exc)}
        save(args.output / "report.json", report)
    files = [p for p in args.output.rglob("*") if p.is_file()]
    save(
        args.output / "sha256.json",
        {
            str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    )
    if hashlib.sha256(cfg.data.manifest.read_bytes()).hexdigest() != manifest_hash:
        raise ValueError("manifesto mudou durante a preparacao")


if __name__ == "__main__":
    main()
