#!/usr/bin/env python3
"""Constroi a particao congelada dos conjuntos experimentais.

Funde os tres splits originais do FINDSum num pool unico e reparticiona em base
de exemplos, desenvolvimento e avaliacao, com uma empresa em um unico conjunto e
um unico relatorio por empresa. Ver `findsum_rag.splits` para o porque.

Uso:
    python scripts/build_splits.py
    python scripts/build_splits.py --examples 1000 --dev 500 --eval 1000
    python scripts/build_splits.py --task roo --out data/interim/splits-roo.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from findsum_rag.data import Task
from findsum_rag.splits import MIN_SUMMARY_WORDS, build_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "raw" / "findsum")
    parser.add_argument("--task", type=Task, default=Task.LIQUIDITY, choices=list(Task))
    parser.add_argument("--examples", type=int, default=1000, help="base de few-shot")
    parser.add_argument("--dev", type=int, default=500, help="ajuste de prompt e parametros")
    parser.add_argument("--eval", type=int, default=1000, help="avaliacao final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-summary-words", type=int, default=MIN_SUMMARY_WORDS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="sobrescreve manifesto existente")
    args = parser.parse_args(argv)

    out = args.out or REPO_ROOT / "data" / "interim" / f"splits-{args.task.value}.json"
    if out.exists() and not args.force:
        print(
            f"{out} ja existe. A particao e congelada de proposito: refaze-la "
            "depois de olhar resultados invalida a avaliacao. Use --force se "
            "tem certeza.",
            file=sys.stderr,
        )
        return 1

    sizes = {"examples": args.examples, "dev": args.dev, "eval": args.eval}
    print(f"tarefa: {args.task.value} | semente: {args.seed}")
    print(f"conjuntos pedidos: {sizes} (total {sum(sizes.values())})")
    print("varrendo o pool (le os tres splits originais)...", flush=True)

    manifest = build_manifest(
        args.root,
        args.task,
        sizes=sizes,
        seed=args.seed,
        min_summary_words=args.min_summary_words,
    )
    manifest.save(out)

    print(f"\nmanifesto gravado em {out}")
    print(f"tamanhos: {manifest.sizes()}")
    for name, refs in manifest.sets.items():
        years = Counter(r.year for r in refs)
        origem = Counter(r.split for r in refs)
        faixa = f"{min(y for y in years if y)}-{max(y for y in years if y)}" if years else "-"
        print(f"\n{name}: {len(refs)} documentos, {len({r.stock_name for r in refs})} empresas")
        print(f"  anos: {faixa}")
        print(f"  origem nos splits do FINDSum: {dict(origem)}")
        print(f"  primeiros: {[r.doc_id for r in refs[:3]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
