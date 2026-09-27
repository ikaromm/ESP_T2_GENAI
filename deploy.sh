#!/usr/bin/env bash
# Entrada unica OpenRouter. Sem geracao local, Docker ou publicacao automatica.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec uv run --locked findsum "$@"
