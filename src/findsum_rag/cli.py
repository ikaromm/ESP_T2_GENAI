"""Interface de linha de comando."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import DEFAULT_ARMS, ExperimentConfig
from .data import SPLITS, Task, load_documents

app = typer.Typer(add_completion=False, help="Experimentos de RAG + few-shot no FINDSum")


@app.command()
def inspect(
    root: Path = typer.Option(Path("data/raw/findsum"), help="raiz dos dados brutos"),
    task: Task = typer.Option(Task.LIQUIDITY, help="tarefa do FINDSum"),
    split: str = typer.Option("val", help=f"um de {SPLITS}"),
    limit: int = typer.Option(20, help="documentos a inspecionar"),
) -> None:
    """Mostra estatisticas dos documentos remontados, para validar a carga."""
    documents = load_documents(root, task, split, limit=limit)
    typer.echo(f"{len(documents)} documentos de {task.value}/{split}")
    doc_words = [d.n_words for d in documents]
    sum_words = [d.n_summary_words for d in documents]
    typer.echo(
        f"entrada: min={min(doc_words)} media={sum(doc_words) // len(doc_words)} "
        f"max={max(doc_words)} palavras"
    )
    typer.echo(
        f"resumo:  min={min(sum_words)} media={sum(sum_words) // len(sum_words)} "
        f"max={max(sum_words)} palavras"
    )
    typer.echo("\nprimeiros documentos:")
    for document in documents[:5]:
        typer.echo(
            f"  {document.doc_id:<32} segmentos={len(document.segments)} "
            f"palavras={document.n_words:>5} tabelas_citadas={len(document.referenced_tables())}"
        )


@app.command("init-config")
def init_config(
    path: Path = typer.Option(Path("configs/dev_openrouter.yaml"), help="destino"),
    force: bool = typer.Option(False, "--force", help="sobrescreve se existir"),
) -> None:
    """Grava um arquivo de configuracao com os valores padrao."""
    if path.exists() and not force:
        typer.echo(f"{path} ja existe; use --force para sobrescrever")
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    ExperimentConfig().to_yaml(path)
    typer.echo(f"configuracao gravada em {path}")


@app.command()
def splits(
    manifest: Path = typer.Option(
        Path("data/interim/splits-liquidity.json"), help="manifesto de particao"
    ),
    root: Path = typer.Option(Path("data/raw/findsum"), help="raiz dos dados brutos"),
    check: bool = typer.Option(False, "--check", help="carrega os documentos e valida"),
) -> None:
    """Mostra a particao congelada dos conjuntos experimentais."""
    from .splits import SplitManifest, load_set

    m = SplitManifest.load(manifest)
    typer.echo(f"tarefa: {m.task.value} | semente: {m.seed} | criado: {m.created_at}")
    typer.echo(f"piso de palavras no resumo: {m.min_summary_words}\n")

    for name, refs in m.sets.items():
        years = [r.year for r in refs if r.year]
        faixa = f"{min(years)}-{max(years)}" if years else "-"
        typer.echo(
            f"{name:<10} {len(refs):>5} documentos  "
            f"{len({r.stock_name for r in refs}):>5} empresas  anos {faixa}"
        )

    total = sum(len(r) for r in m.sets.values())
    companies = {r.stock_name for refs in m.sets.values() for r in refs}
    typer.echo(f"\ntotal: {total} documentos, {len(companies)} empresas distintas")
    if len(companies) != total:
        typer.echo("ATENCAO: ha empresa repetida entre conjuntos")
        raise typer.Exit(code=1)
    typer.echo("uma empresa por conjunto, um relatorio por empresa: OK")

    if check:
        for name in m.sets:
            documents = load_set(root, m, name, limit=5, with_tables=False)
            typer.echo(f"  {name}: carregou {len(documents)} documentos de amostra")


@app.command()
def arms() -> None:
    """Lista as configuracoes experimentais padrao."""
    for arm in DEFAULT_ARMS:
        typer.echo(
            f"{arm.id:<4} contexto={arm.context_mode:<10} "
            f"exemplos={arm.example_strategy:<8} n={arm.n_examples}  {arm.label}"
        )


def _script(name: str, arguments: list[str]) -> None:
    """Executa os scripts do checkout instalado, sem interpolacao de shell."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(repo / "scripts" / name), *arguments], cwd=repo)
    if result.returncode:
        raise typer.Exit(result.returncode)


@app.command("verify-full")
def verify_full() -> None:
    """Confere o lock e as versoes; nenhuma chamada API."""
    _script("freeze_full_openrouter.py", ["--verify"])


@app.command()
def prepare(
    output: Path = typer.Option(...),
    config: Path = typer.Option(Path("configs/full_openrouter.yaml")),
    full: bool = typer.Option(True, "--full/--dev"),
) -> None:
    """Prepara fonte, RAG, few-shot e tokenizers locais; nao gera resumos."""
    args = ["--output", str(output), "--config", str(config)]
    if full:
        args.append("--full")
    _script("prepare_full_openrouter.py", args)


@app.command("ling-batch")
def ling_batch(
    prepared: Path = typer.Option(Path("outputs/full-ling-prepared")),
    output: Path = typer.Option(Path("outputs/full-ling-batches")),
    batch: int | None = typer.Option(None, min=1, max=10),
    execute: bool = typer.Option(False, "--execute"),
) -> None:
    """Ling gratuito: valida 1000 casos e executa/retoma um lote fixo de 100."""
    args = ["--prepared", str(prepared), "--output", str(output)]
    if batch is not None:
        args.extend(["--batch", str(batch)])
    if execute:
        args.append("--execute")
    _script("run_ling_batches.py", args)


@app.command()
def calibrate(
    prepared: Path = typer.Option(...),
    output: Path = typer.Option(...),
    budget_usd: float = typer.Option(18.0, min=0.0),
    allow_eval: bool = typer.Option(False, "--allow-eval"),
    execute: bool = typer.Option(False, "--execute"),
) -> None:
    """Sondas Qwen PAGAS somente com --execute; nao sao resumos do experimento."""
    args = ["--prepared", str(prepared), "--output", str(output), "--budget-usd", str(budget_usd)]
    if allow_eval:
        args.append("--allow-eval")
    if execute:
        args.append("--execute")
    _script("prepare_qwen_remote.py", args)


@app.command()
def run(
    prepared: Path = typer.Option(...),
    qwen_prepared: Path = typer.Option(...),
    output: Path = typer.Option(...),
    qwen_budget: float = typer.Option(18.0, min=0.0),
    gemma_budget: float = typer.Option(9.0, min=0.0),
    allow_eval: bool = typer.Option(False, "--allow-eval"),
    execute: bool = typer.Option(False, "--execute"),
) -> None:
    """Unico executor principal: OpenRouter, tres modelos, seis bracos e retomada."""
    args = [
        "--prepared",
        str(prepared),
        "--qwen-prepared",
        str(qwen_prepared),
        "--output",
        str(output),
        "--qwen-budget",
        str(qwen_budget),
        "--gemma-budget",
        str(gemma_budget),
    ]
    if allow_eval:
        args.append("--allow-eval")
    if execute:
        args.append("--execute")
    _script("run_experiment_openrouter.py", args)


@app.command()
def compare(
    run_dir: Path = typer.Option(..., help="diretorio de uma rodada, ex: outputs/baseline-run"),
    reference: str | None = typer.Option(None, help="contraste exploratorio contra este braco"),
    alpha: float = typer.Option(0.05, help="nivel de significancia"),
) -> None:
    """Compara estatisticamente as configuracoes de uma rodada."""
    from dataclasses import asdict

    from .analyze import ANALYSIS_PLAN, compare_against, compare_hypotheses, format_table, load_run

    scores = load_run(run_dir)
    if not scores:
        typer.echo(f"nenhum scores.csv encontrado em {run_dir}")
        raise typer.Exit(code=1)
    typer.echo(f"configuracoes encontradas: {', '.join(sorted(scores))}\n")
    if reference:
        comparisons = compare_against(scores, reference)
        typer.echo("Analise exploratoria; nao substitui H1--H4.")
    else:
        plan_path = run_dir / "analysis_plan.json"
        if not plan_path.exists() or json.loads(plan_path.read_text()) != json.loads(
            json.dumps(ANALYSIS_PLAN)
        ):
            raise typer.BadParameter("plano predefinido ausente ou diferente nesta rodada")
        try:
            comparisons = compare_hypotheses(scores)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo("H1 C2/C1; H1b C2/C1t; H2 C3/C2; H3 C4/C3; H4 C5/C4.")
        typer.echo("H3: ausencia de significancia nao demonstra equivalencia.")
    filename = "exploratory_comparisons.json" if reference else "comparisons.json"
    (run_dir / filename).write_text(json.dumps([asdict(c) for c in comparisons], indent=2))
    typer.echo(format_table(comparisons, alpha=alpha))


@app.command("review-export")
def review_export(run_dir: Path = typer.Option(...), output: Path = typer.Option(...)) -> None:
    """Exporta resumos/evidencias para revisao humana de afirmacoes."""
    from .factual import export_review

    export_review(run_dir, output)
    typer.echo(f"Revisao: {output}; mantenha o mapa separado do revisor.")


@app.command("review-score")
def review_score(review: Path = typer.Option(...), output: Path = typer.Option(...)) -> None:
    """Pontua apenas revisoes completas; nao automatiza o julgamento factual."""
    from .factual import evidence_digest, score_review

    cases = json.loads(review.read_text())
    mapping = json.loads(review.with_suffix(".map.json").read_text())
    results = score_review(cases)
    if {row["case_id"] for row in results} != set(mapping):
        raise typer.BadParameter("revisao deve cobrir todos os casos exportados")
    for case in cases:
        if evidence_digest(case) != mapping[case["case_id"]]["evidence_sha256"]:
            raise typer.BadParameter("resumo ou evidencias foram alterados durante a revisao")
    for row in results:
        row.update(mapping[row["case_id"]])
    if output.exists():
        raise typer.BadParameter("arquivo de saida ja existe")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    typer.echo(f"Escores factuais revisados: {output}")


if __name__ == "__main__":
    app()
