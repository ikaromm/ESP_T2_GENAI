# Ferramentas do experimento

Execute os módulos a partir da raiz do repositório. As entradas do dia a dia continuam sendo `bash rodar_rodada.sh 100` e `bash rodar_rodada.sh --metrics-only`.

| Pasta | Responsabilidade | Módulos principais |
|---|---|---|
| `data/` | Obter dados e separar conjuntos | `fetch_findsum`, `build_splits` |
| `preparation/` | Preparar fonte/RAG/exemplos, validar tokens e congelar | `prepare_full_openrouter`, `prepare_gemma_from_common`, `freeze_full_openrouter` |
| `execution/` | Executar e retomar chamadas | `run_full_round`, `run_ling_batches`, `run_paid_concurrent` |
| `evaluation/` | Métricas e auditoria | `update_progress_metrics`, `extend_reference_metrics`, `audit_example_metrics` |
| `reports/` | Exportar relatórios locais | `build_dev_review`, `build_example_guide`, `build_pipeline_guide` |
| `experiments/` | PoCs e testes auxiliares, fora do Bash normal | `poc_openrouter`, `benchmark_paid_rates`, `run_example_trace` |
| `common/` | Contratos compartilhados | `full_common` |

## Comandos

```bash
# Consulta os argumentos sem iniciar geração:
uv run --locked python -m scripts.data.fetch_findsum --help
uv run --locked python -m scripts.preparation.prepare_full_openrouter --help
uv run --locked python -m scripts.execution.run_full_round --help

# Verifica o lock existente; não o recria:
uv run --locked python -m scripts.preparation.freeze_full_openrouter --verify

# Preparação local antecipada do Qwen, reutilizando a preparação comum:
uv run --locked python -m scripts.preparation.prepare_gemma_from_common \
  --model qwen37 --source outputs/full-ling-prepared --output outputs/full-qwen-prepared

# Atualiza as métricas pela entrada que valida os artefatos:
bash rodar_rodada.sh --metrics-only
```

Preparações prontas não devem ser sobrescritas. Não execute `freeze_full_openrouter` sem `--verify` para contornar uma divergência: o congelamento novo exige revisão e preservação do lock anterior.

As funções de cálculo em `evaluation/update_progress_metrics.py` são chamadas pelo orquestrador; use `--metrics-only`, em vez de executar esse arquivo diretamente. Os comandos `findsum ling-batch`, `qwen-batch` e `gemma-batch` continuam disponíveis; o Bash mantém os três modelos alinhados e atualiza o painel.

Alguns módulos são históricos: `prepare_qwen_remote` faz calibração via API quando explicitamente executado, e os builders de HTML reconstroem demonstrações antigas a partir dos respectivos artefatos locais. Eles não fazem parte da preparação local atual nem substituem as métricas atuais. Os templates ficam em `reports/templates/`; os HTMLs gerados vão para `outputs/reports/`, fora do Git.
