# Embeddings e metricas locais; toda geracao ocorre via OpenRouter.
# GPU e opcional para acelerar avaliacao, nao e requisito de geracao.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# As dependencias entram antes do codigo para que alterar um .py nao invalide a
# camada de ~3 GB do torch.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
RUN uv sync --locked --no-dev
RUN python -m nltk.downloader -d /usr/local/share/nltk_data wordnet

# O container roda com o UID do host (ver deploy.sh) para que os arquivos
# gravados nos volumes nao saiam pertencendo ao root. Esses diretorios precisam
# existir e ser graváveis por qualquer UID.
RUN mkdir -p /cache/huggingface /app/data /app/outputs /app/results /app/dist \
    && chmod -R 0777 /cache /app/outputs /app/results /app/dist

CMD ["findsum", "--help"]
