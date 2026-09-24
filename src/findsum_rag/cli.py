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
    path: Path = typer.Option(Path("configs/experiment.yaml"), help="destino"),
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


@app.command()
def run(
    config: Path = typer.Option(..., help="arquivo YAML de configuracao"),
    arm: list[str] = typer.Option(None, "--arm", help="rodar apenas estas configuracoes"),
    no_bertscore: bool = typer.Option(False, "--no-bertscore", help="pula o BERTScore"),
) -> None:
    """Executa a rodada experimental."""
    from .pipeline import run_experiment

    cfg = ExperimentConfig.from_yaml(config)
    results = run_experiment(cfg, arms=arm or None, with_bertscore=not no_bertscore)
    typer.echo(json.dumps({k: v.summary for k, v in results.items()}, indent=2))


@app.command()
def compare(
    run_dir: Path = typer.Option(..., help="diretorio de uma rodada, ex: outputs/baseline-run"),
    reference: str = typer.Option("C5", help="configuracao de referencia da comparacao"),
    alpha: float = typer.Option(0.05, help="nivel de significancia"),
) -> None:
    """Compara estatisticamente as configuracoes de uma rodada."""
    from .analyze import compare_against, format_table, load_run

    scores = load_run(run_dir)
    if not scores:
        typer.echo(f"nenhum scores.csv encontrado em {run_dir}")
        raise typer.Exit(code=1)
    typer.echo(f"configuracoes encontradas: {', '.join(sorted(scores))}\n")
    typer.echo(format_table(compare_against(scores, reference), alpha=alpha))


if __name__ == "__main__":
    app()
