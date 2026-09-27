"""Registra quanto texto o BERTScore padrao efetivamente leu, sem recalcular escores."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from bert_score.utils import get_tokenizer, sent_encode

from screen_openrouter import save


def audit(text, tokenizer):
    full = tokenizer.encode(
        text.strip(), add_special_tokens=True, add_prefix_space=True, truncation=False
    )
    used = sent_encode(tokenizer, text)
    return {
        "full_tokens_with_specials": len(full),
        "used_tokens_with_specials": len(used),
        "truncated": len(full) > len(used),
        "evaluated_text": tokenizer.decode(used, skip_special_tokens=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    path = args.run / "trace.json"
    trace = json.loads(path.read_text())
    tokenizer = get_tokenizer("roberta-large", use_fast=False)
    trace["metric_token_audit"] = {
        "model": "roberta-large",
        "max_length": tokenizer.model_max_length,
        "reference": audit(trace["reference"], tokenizer),
        "predictions": {
            a["arm"]: audit(a["prediction"]["prediction"], tokenizer)
            for a in trace["arms"]
            if a["status"] == "accepted"
        },
        "interpretation": (
            "BERTScore padrao trunca textos na janela do encoder; ROUGE usa texto completo."
        ),
    }
    trace["versions"] = {
        p: importlib.metadata.version(p)
        for p in ["bert-score", "transformers", "torch", "rouge-score", "sentence-transformers"]
    }
    paths = [
        "src/findsum_rag/data.py",
        "src/findsum_rag/chunking.py",
        "src/findsum_rag/context.py",
        "src/findsum_rag/examples.py",
        "src/findsum_rag/metrics.py",
        "src/findsum_rag/pipeline.py",
        "src/findsum_rag/prompts.py",
        "src/findsum_rag/retrieval.py",
        "scripts/run_example_trace.py",
    ]
    trace["code_sha256"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    trace["artifact_sha256"] = {
        str(p.relative_to(args.run)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.run.rglob("*")
        if p.is_file() and p != path
    }
    save(path, trace)
    print(
        "Referencia:",
        {
            k: v
            for k, v in trace["metric_token_audit"]["reference"].items()
            if k != "evaluated_text"
        },
    )
    print(
        "Predicoes truncadas pela metrica:",
        [k for k, v in trace["metric_token_audit"]["predictions"].items() if v["truncated"]],
    )


if __name__ == "__main__":
    main()
