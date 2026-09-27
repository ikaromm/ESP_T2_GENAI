# Auditoria técnica do primeiro lote Ling

Verificação local realizada em 27/09/2026, às 12h05 (America/Sao_Paulo), pelo caminho real do Bash em `--dry-run`. Nenhuma nova geração ou consulta OpenRouter foi enviada nesta auditoria.

| Verificação | Resultado |
|---|---:|
| Documentos da coorte congelada | 100 |
| Respostas aceitas | 600/600 |
| Respostas por braço C1, C1t, C2, C3, C4, C5 | 100 em cada |
| Tentativas registradas | 603 |
| Erros HTTP 429 no histórico | 3 |
| Pendências | 0 |
| Encerramento `stop` | 600 |
| Encerramento `length` | 0 |
| Tokens de entrada reportados | 13.313.594 |
| Tokens de saída reportados | 358.242 |
| Custo reportado | US$ 0 |

A auditoria verificou os hashes de código, configuração, dataset e recursos locais; recontou os 6.000 prompts preparados do Ling; conferiu a identidade da rodada; comparou os requests das 603 tentativas aos prompts congelados; validou modelo/provedor, custo zero, contagens de entrada e saída, texto não vazio e motivo de encerramento das 600 respostas aceitas. Também foram mantidas as verificações C1t/C2 e C2–C5 do preflight. Nenhum documento foi substituído.

O resultado aprova a integridade técnica do lote. As métricas científicas e os testes finais das hipóteses continuam programados para os 1.000 documentos de cada modelo.

Relatório auditável local: `outputs/full-ling-batches/audits/batch-01.json`, com IDs dos 100 documentos. Requests e respostas permanecem em `outputs/full-ling-batches/ling-free/`. Esses artefatos locais não são incluídos no Git.

Repetir somente a auditoria, sem API:

```bash
bash rodar_rodada.sh --audit-ling-batch 1
```

O plano real de `bash rodar_rodada.sh 100 --dry-run` selecionou esses mesmos 100 documentos e identificou: Ling 0 gerações pendentes, Gemma 600 e Qwen 600. A preparação Qwen ainda requer calibração paga integral. A conferência não criou uma rodada ativa nem avançou a coorte.
