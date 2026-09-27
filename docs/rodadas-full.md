# Rodadas dos três modelos com um Bash

Na raiz do repositório:

```bash
bash rodar_rodada.sh 100
```

**Esse comando executa geração gratuita no Ling e geração paga no Gemma e Qwen. Se faltar a calibração do Qwen, também executa sondas pagas.** O número indica quantos documentos distintos da coorte serão tratados na rodada, com os mesmos IDs nos três modelos. Aceita de 1 a 1.000; o padrão, se omitido, é 100. Cada documento tem seis braços.

Para conferir o plano sem consultar nem gerar pela API:

```bash
bash rodar_rodada.sh 100 --dry-run
```

## Como escolhe e retoma

O script verifica os artefatos e as respostas aceitas, depois escolhe os primeiros documentos da ordem congelada que ainda tenham algum braço faltando em qualquer modelo. Não sorteia novos casos nem substitui documentos por causa de erros ou resultados.

Como o primeiro lote do Ling já concluiu os primeiros 100 documentos, a próxima rodada de 100 trata esses mesmos IDs: pula as 600 gerações aceitas do Ling e executa as pendências do Gemma e do Qwen. Só depois de concluir essa rodada nos três modelos a próxima invocação escolhe outros documentos.

A rodada é gravada em `outputs/full-rounds/active-round.json` antes de qualquer geração ou calibração. Se houver interrupção, repita o mesmo comando e a mesma quantidade. O script retoma os mesmos IDs e confere as respostas já salvas. Enquanto houver uma rodada pendente, outra quantidade é rejeitada para evitar mudar o grupo no meio da execução.

Os modelos rodam um após o outro: Ling, Gemma, Qwen. Se um falhar, o script tenta os outros nos mesmos documentos, mantém a rodada pendente e termina com código diferente de zero. Uma interrupção do usuário encerra o processo e conserva o plano. Se uma chamada ficar com resultado desconhecido após interrupção ou falha de transporte, a repetição dessa chamada exige auditoria para evitar duplicação. Depois de concluir os três, arquiva a rodada em `outputs/full-rounds/history/` e encerra; não inicia outra rodada automaticamente.

## Preparação e limites

A preparação integral do Ling precisa existir. A do Gemma é criada localmente se estiver ausente; uma pasta Gemma incompleta exige auditoria, sem sobrescrever artefatos. A calibração Qwen é criada ou retomada automaticamente quando faltar sua preparação completa.

**A calibração Qwen cobre os 1.000 documentos antes da geração, mesmo quando a rodada pede apenas 100.** Essas sondas são chamadas pagas adicionais e podem demorar. O teto de calibração continua separado do teto de geração.

| Etapa | Teto cumulativo do experimento | Máximo por minuto, por executor |
|---|---:|---:|
| Ling gratuito | US$ 0 | 20 |
| Gemma, geração | US$ 9 | 100 |
| Qwen, geração | US$ 18 | 100 |
| Qwen, calibração | US$ 18 | 100 |

Tetos não são cobranças antecipadas nem garantia de concluir tudo com esse saldo. As chamadas permanecem sequenciais. Retries contam na janela de frequência e no controle de orçamento/cota. A cota gratuita é consultada antes das gerações Ling; insuficiência interrompe esse modelo. Não há fallback para modelo ou provedor diferente.

Cada rodada de 100 pode precisar de até 600 gerações por modelo, além de retries e da calibração Qwen inicial. Ao retomar, faz apenas as gerações faltantes. Não execute outro cliente na mesma conta durante a rodada se precisar manter esses limites agregados.

## Auditoria do primeiro lote Ling

Para executar somente a auditoria local, sem API:

```bash
bash rodar_rodada.sh --audit-ling-batch 1
```

Ela verifica congelamento, todos os prompts Ling, identidade do ledger, requests das tentativas, respostas aceitas, modelo/provedor, contagens de tokens, saída não vazia, motivo de encerramento e custo. Saídas `length`, se houver, são preservadas conforme o protocolo. O relatório fica em `outputs/full-ling-batches/audits/batch-01.json`.

Essa é uma auditoria técnica. Métricas e testes das hipóteses finais continuam sendo calculados após completar os 1.000 documentos de cada modelo.

## Arquivos e compatibilidade

- `outputs/full-rounds/preview.json`: documentos da rodada, preparação pendente e gerações faltantes por modelo.
- `outputs/full-rounds/active-round.json`: plano imutável da rodada em andamento; não apagar para tentar avançar.
- `outputs/full-rounds/last-result.json`: resultado da última tentativa da rodada.
- `outputs/full-rounds/history/`: rodadas concluídas nos três modelos.
- `outputs/full-{ling,gemma,qwen}-batches/`: mesmos ledgers, requests, respostas e orçamentos dos comandos individuais.

Os comandos `findsum ling-batch`, `gemma-batch` e `qwen-batch` continuam disponíveis e seus lotes canônicos continuam tendo 100 documentos. O Bash permite outra quantidade sem mudar a coorte nem a identidade dos ledgers. Use o Bash nas próximas rodadas para manter os três modelos alinhados.

O próprio Bash e o orquestrador Python são incluídos no congelamento. A atualização operacional aceita explicitamente as preparações anteriores do Ling e do Gemma, mantendo os hashes dos insumos científicos e as respostas já geradas.
