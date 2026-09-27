#!/usr/bin/env bash
# Uso: bash rodar_rodada.sh 100 [--dry-run] | --metrics-only
# Execucao atualiza metricas locais ao encerrar. --metrics-only nao gera respostas.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_dir"
exec uv run --locked python -u -m scripts.execution.run_full_round "$@"
