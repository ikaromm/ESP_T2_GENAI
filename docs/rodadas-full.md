# Rodadas dos três modelos com um Bash

Na raiz do repositório:

```bash
bash rodar_rodada.sh 100
```

**Esse comando executa geração gratuita no Ling e geração paga no Gemma e Qwen. A preparação Qwen é local e não faz sondas pagas.** O número indica quantos documentos distintos da coorte serão tratados na rodada, com os mesmos IDs nos três modelos. Aceita de 1 a 1.000; o padrão, se omitido, é 100. Cada documento tem seis braços.

Para conferir o plano sem consultar nem gerar pela API:

```bash
bash rodar_rodada.sh 100 --dry-run
```

## Como escolhe e retoma

O script verifica os artefatos e as respostas aceitas, depois escolhe os primeiros documentos da ordem congelada que ainda tenham algum braço faltando em qualquer modelo. Não sorteia novos casos nem substitui documentos por causa de erros ou resultados.

Como o primeiro lote do Ling já concluiu os primeiros 100 documentos, a próxima rodada de 100 trata esses mesmos IDs: pula as 600 gerações aceitas do Ling e executa as pendências do Gemma e do Qwen. Só depois de concluir essa rodada nos três modelos a próxima invocação escolhe outros documentos.

A rodada é gravada em `outputs/full-rounds/active-round.json` antes de qualquer geração ou preparação. Se houver interrupção, repita o mesmo comando e a mesma quantidade. O script retoma os mesmos IDs e confere as respostas já salvas. Enquanto houver uma rodada pendente, outra quantidade é rejeitada para evitar mudar o grupo no meio da execução.

Os modelos rodam simultaneamente: Ling mantém seu executor gratuito de 20 RPM; Qwen e Gemma usam concorrência com teto de 500 RPM por modelo e retry adaptativo. Se um falhar, o script tenta os outros nos mesmos documentos, mantém a rodada pendente e termina com código diferente de zero. Uma interrupção do usuário bloqueia novos envios, recolhe as respostas já em voo e conserva o plano. Se uma chamada ficar com resultado desconhecido após interrupção ou falha de transporte, a repetição dessa chamada exige auditoria para evitar duplicação. Depois de concluir os três, arquiva a rodada em `outputs/full-rounds/history/` e encerra; não inicia outra rodada automaticamente.

## Preparação e limites

A preparação integral do Ling precisa existir. A do Gemma é criada localmente se estiver ausente; uma pasta Gemma incompleta exige auditoria, sem sobrescrever artefatos. A preparação Qwen também é local e criada se ausente; uma pasta incompleta exige auditoria, sem sobrescrita.

**A preparação local Qwen cobre os 1.000 documentos antes da geração, mesmo quando a rodada pede apenas 100.** Os prompts prontos são reutilizados. Nenhuma chamada de calibração é necessária; veja o comando de preparação antecipada no [guia Qwen/Gemma](lotes-qwen-gemma-full.md).

| Etapa | Teto cumulativo do experimento | Máximo por minuto, por executor |
|---|---:|---:|
| Ling gratuito | US$ 0 | 20 |
| Gemma, geração | US$ 9 | 500 (adaptativo) |
| Qwen, geração | US$ 18 | 500 (adaptativo) |
| Qwen, preparação local | US$ 0 | Sem API |

Tetos não são cobranças antecipadas nem garantia de concluir tudo com esse saldo. Nos modelos pagos, até 100 chamadas por modelo podem ficar em andamento, com redução automática do ritmo e da concorrência diante de falhas transitórias. Retries contam na janela de frequência e no controle de orçamento/cota. A cota gratuita é consultada antes das gerações Ling; insuficiência interrompe esse modelo. Não há fallback para modelo ou provedor diferente.

Cada rodada de 100 pode precisar de até 600 gerações por modelo, além de retries. Ao retomar, faz apenas as gerações faltantes. Não execute outro cliente na mesma conta durante a rodada se precisar manter esses limites agregados.

## Auditoria do primeiro lote Ling

Para executar somente a auditoria local, sem API:

```bash
bash rodar_rodada.sh --audit-ling-batch 1
```

Ela verifica congelamento, todos os prompts Ling, identidade do ledger, requests das tentativas, respostas aceitas, modelo/provedor, contagens de tokens, saída não vazia, motivo de encerramento e custo. Saídas `length`, se houver, são preservadas conforme o protocolo. O relatório fica em `outputs/full-ling-batches/audits/batch-01.json`.

Essa é uma auditoria técnica. As métricas descritivas são atualizadas pelo Bash após cada rodada. Testes das hipóteses finais continuam separados e exigem os 1.000 documentos de cada modelo.

## Arquivos e compatibilidade

- `outputs/full-rounds/preview.json`: documentos da rodada, preparação pendente e gerações faltantes por modelo.
- `outputs/full-rounds/active-round.json`: plano imutável da rodada em andamento; não apagar para tentar avançar.
- `outputs/full-rounds/last-result.json`: resultado da última tentativa da rodada.
- `outputs/full-rounds/history/`: rodadas concluídas nos três modelos.
- `outputs/full-{ling,gemma,qwen}-batches/`: mesmos ledgers, requests, respostas e orçamentos dos comandos individuais.

Os comandos `findsum ling-batch`, `gemma-batch` e `qwen-batch` continuam disponíveis e seus lotes canônicos continuam tendo 100 documentos. O Bash permite outra quantidade sem mudar a coorte nem a identidade dos ledgers. Use o Bash nas próximas rodadas para manter os três modelos alinhados.

O próprio Bash e o orquestrador Python são incluídos no congelamento. A atualização operacional aceita explicitamente as preparações anteriores do Ling e do Gemma, mantendo os hashes dos insumos científicos e as respostas já geradas.

## Teste concorrente pago — 27/09/2026

O [relatório de vazão](teste-vazao-openrouter-20260927.md) registra 185 respostas Qwen e 300 Gemma aceitas nos ledgers oficiais. A próxima rodada dos primeiros 100 documentos tem 415 gerações Qwen e 300 Gemma pendentes; Ling não tem pendências. 100 RPM passou nos dois modelos, enquanto 200 e 500 tiveram HTTP 429. O teste paralelo usou `scripts/benchmark_paid_rates.py`; após esse teste, o usuário escolheu teto 500 RPM e o Bash passou a usar [retry adaptativo](retry-adaptativo-openrouter.md). O teto é uma configuração operacional, não capacidade sustentada comprovada.

## Painel automático e estado atual

Os primeiros 100 documentos estão completos nos três modelos, conforme [auditoria da rodada](rodada100-adaptativa-20260927.md). As pendências citadas no teste de vazão acima eram o estado anterior à retomada.

Após cada execução do Bash, inclusive rodada com pendências, `results/progress/` recebe métricas e gráficos sobre o prefixo comum completo. O cache local evita recalcular respostas inalteradas. `--dry-run` não calcula métricas. Use `bash rodar_rodada.sh --metrics-only` para atualizar apenas o painel, sem iniciar outra rodada. Consulte [métricas automáticas](metricas-automaticas.md).
