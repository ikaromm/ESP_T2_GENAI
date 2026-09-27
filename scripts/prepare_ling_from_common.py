"""Revalida prompts Ling a partir de insumos locais auditados; nenhuma API/embedding."""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from findsum_rag.chunking import Chunk
from findsum_rag.config import ExperimentConfig
from findsum_rag.context import matched_contexts, token_count
from findsum_rag.examples import Example
from findsum_rag.full_lock import digest, verify_full_lock
from findsum_rag.prompts import build_prompt, format_context
from full_common import Limits, selected_records, verify_hashes
from prepare_full_openrouter import PreparationTokenizer
from screen_openrouter import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = verify_full_lock()
    verify_hashes(args.source)
    split, records = selected_records(args.source)
    if split != "eval" or records != frozen["cohort"]:
        raise ValueError("coorte diferente")
    previous = json.loads((args.source / "full-lock.json").read_text())
    # Migracao mecanica: unica mudanca admitida nos arquivos preexistentes e
    # reconhecer marcadores de tabela, sem confundir pipes de prosa com celulas.
    for name, sha in previous["files"].items():
        if name != "src/findsum_rag/context.py" and frozen["files"].get(name) != sha:
            raise ValueError(f"insumo/codigo mudou alem da correcao de borda: {name}")
    cfg = ExperimentConfig.from_yaml(args.source / "config.yaml")
    if cfg != ExperimentConfig.from_yaml("configs/full_openrouter.yaml"):
        raise ValueError("perfil diferente")
    target = json.loads(Path("configs/openrouter_full.json").read_text())["models"]["ling-free"]
    tokenizer = PreparationTokenizer(cfg.generation, str(args.source / "ling-free/tokenizer"))
    tokenizer.load_tokenizer()
    documents = json.loads((args.source / "documents.json").read_text())
    examples = {
        r["doc_id"]: Example(**r) for r in json.loads((args.source / "examples.json").read_text())
    }
    if [r["doc_id"] for r in documents] != [r["doc_id"] for r in records]:
        raise ValueError("ordem diferente")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (
        "config.yaml",
        "selection.json",
        "selected-cases.csv",
        "analysis_plan.json",
        "documents.json",
        "examples.json",
        "table-audit.json",
    ):
        shutil.copyfile(args.source / name, args.output / name)
    folder = args.output / "ling-free"
    shutil.copytree(args.source / "ling-free/tokenizer", folder / "tokenizer")
    shutil.copyfile(args.source / "ling-free/endpoint-plan.json", folder / "endpoint-plan.json")
    save(args.output / "full-lock.json", frozen)
    limits = Limits(
        cfg.generation.max_input_tokens, cfg.generation.max_new_tokens, target["context_length"]
    )
    old = {
        (r["doc_id"], r["arm"]): r
        for r in json.loads((args.source / "ling-free/preflight.json").read_text())
    }
    rows, errors, changed = [], [], []
    for i, d in enumerate(documents, 1):
        try:

            def prompt(context, arm, selected):
                return build_prompt(
                    task=cfg.data.task,
                    arm=arm,
                    context_chunks=[],
                    context_text=context,
                    examples=selected,
                    example_max_words=cfg.data.example_max_words,
                )

            truncated, retrieved = matched_contexts(
                tokenizer.tokenizer,
                d["source"],
                format_context([Chunk(**c) for c in d["retrieved_candidates"]]),
                cfg.retrieval.context_max_tokens,
                lambda text: tokenizer.count_tokens(prompt(text, cfg.arm("C2"), [])),
            )
            contexts = {"full": d["source"], "truncated": truncated, "retrieved": retrieved}
            group = []
            for arm in cfg.arms:
                context = contexts[arm.context_mode]
                selected = [examples[key] for key in d["example_ids"][arm.id]]
                p = prompt(context, arm, selected)
                count = tokenizer.count_tokens(p)
                limits.check(count)
                row = {
                    "doc_id": d["doc_id"],
                    "arm": arm.id,
                    "messages": p.as_messages(),
                    "context": context,
                    "example_ids": [e.doc_id for e in selected],
                    "context_tokens": token_count(tokenizer.tokenizer, context),
                    "prompt_tokens": count,
                    "max_output_tokens": limits.max_output,
                    "allow_length": True,
                }
                if (d["doc_id"], arm.id) in old and row != old[(d["doc_id"], arm.id)]:
                    changed.append({"doc_id": d["doc_id"], "arm": arm.id})
                group.append(row)
            rows.extend(group)
        except ValueError as exc:
            errors.append({"doc_id": d["doc_id"], "error": str(exc)})
        if i % 25 == 0:
            print(f"Preflight Ling: {i}/1000 documentos; {len(errors)} erros", flush=True)
    save(folder / "preflight.json", rows)
    report = {
        "scope": "preparation_eval",
        "documents": 1000,
        "examples": len(examples),
        "generation_calls": 0,
        "models": {
            "ling-free": {
                "status": "preflight_passed" if len(rows) == 6000 and not errors else "incomplete",
                "prompts_validated": len(rows),
                "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
                "input_tokens_total": sum(r["prompt_tokens"] for r in rows),
                "cost_max_without_retries_usd": 0,
                "errors": errors,
            }
        },
        "migration": {
            "previous_lock_id": previous["lock_id"],
            "existing_prompts_changed": changed,
            "limits": asdict(limits),
        },
    }
    save(args.output / "report.json", report)
    save(
        args.output / "sha256.json",
        {str(p.relative_to(args.output)): digest(p) for p in args.output.rglob("*") if p.is_file()},
    )
    print(json.dumps(report), flush=True)
    if errors:
        raise ValueError("preparacao incompleta; nenhuma geracao autorizada")


if __name__ == "__main__":
    main()
