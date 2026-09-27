#!/usr/bin/env bash
# Uso: bash rodar_rodada.sh 100 [--dry-run]
# Sem --dry-run, executa APIs pagas e calibra Qwen se necessario.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_dir"
exec uv run --locked python -u scripts/run_full_round.py "$@"
