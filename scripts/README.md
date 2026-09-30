# Ferramentas do experimento

Execute os módulos a partir da raiz do repositório. As entradas do dia a dia continuam sendo `bash rodar_rodada.sh 100` e `bash rodar_rodada.sh --metrics-only`.

| Pasta | Responsabilidade | Módulos principais |
|---|---|---|
| `data/` | Obter dados e separar conjuntos | `fetch_findsum`, `build_splits` |
| `preparation/` | Preparar fonte/RAG/exemplos, validar tokens e congelar | `prepare_full_openrouter`, `prepare_gemma_from_common`, `freeze_full_openrouter` |
| `execution/` | Executar e retomar chamadas | `run_full_round`, `run_ling_batches`, `run_paid_concurrent` |
| `evaluation/` | Métricas, auditoria e figuras legíveis | `update_progress_metrics`, `render_readable_progress`, `extend_reference_metrics`, `audit_example_metrics` |
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

# Atualiza somente as figuras e as tabelas descritivas dos READMEs:
uv run --locked python -m scripts.evaluation.render_readable_progress

# Ling pago no Novita somente quando a rodada selecionada coincide com um adendo
# registrado; 521–1000 processa os 480 restantes ou retoma essa mesma rodada:
bash rodar_rodada.sh 480 --ling-paid-this-round   # configs/ling-paid-round-521-1000.json
# As rodadas anteriores já estão concluídas:
bash rodar_rodada.sh 200 --ling-paid-this-round   # configs/ling-paid-round-321-520.json
bash rodar_rodada.sh 160 --ling-paid-this-round   # configs/ling-paid-round-161-320.json
```

Uma nova autorização de Ling pago exige outro arquivo `configs/ling-paid-round-<faixa>.json`, sem editar os anteriores. Registre o SHA-256 do arquivo em `PAID_LING_AMENDMENTS` (`execution/run_prepared_paid.py`), preserve o lock atual em `configs/lock-history/` e recongele com os três `--compatible-prepared-lock`. O plano da rodada é conferido contra o adendo antes de ser salvo; cada tentativa paga grava o hash do adendo no ledger. O adendo v2 também fixa tetos incrementais de custo retido por modelo, medidos apenas nos casos da rodada.

Preparações prontas não devem ser sobrescritas. Não execute `freeze_full_openrouter` sem `--verify` para contornar uma divergência: o congelamento novo exige revisão e preservação do lock anterior.

As funções de cálculo em `evaluation/update_progress_metrics.py` são chamadas pelo orquestrador; use `--metrics-only`, em vez de executar esse arquivo diretamente. Os comandos `findsum ling-batch`, `qwen-batch` e `gemma-batch` continuam disponíveis; o Bash mantém os três modelos alinhados e atualiza o painel.

A pontuação é centralizada no cache comum, em blocos de 120 pares, sem corte do texto. Executores individuais exportam a geração; finalize as métricas com `bash rodar_rodada.sh --metrics-only`. Quando a matriz contém os 1.000 documentos completos, `run_full_round` publica os 30 testes predefinidos a partir do CSV, com dez testes/Holm por modelo. Ele rejeita pares ausentes, duplicados ou não finitos e registra hashes do CSV e das comparações. A rotina histórica `run_experiment_openrouter.score_results` não é chamada pelo Bash normal.

Alguns módulos são históricos: `prepare_qwen_remote` faz calibração via API quando explicitamente executado, e os builders de HTML reconstroem demonstrações antigas a partir dos respectivos artefatos locais. Eles não fazem parte da preparação local atual nem substituem as métricas atuais. Os templates ficam em `reports/templates/`; os HTMLs gerados vão para `outputs/reports/`, fora do Git.
