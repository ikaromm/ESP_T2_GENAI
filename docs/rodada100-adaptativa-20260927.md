# Primeiros 100 documentos: execução adaptativa real

Em 27/09/2026, concluímos os mesmos 100 documentos nos três modelos, com os seis braços: **1.800 respostas**, 600 por modelo. Nenhum documento do segundo lote foi enviado. O Ling já estava completo e não recebeu chamadas novas.

## Resultado operacional

| Modelo | Respostas novas | Tentativas novas | Falhas recuperadas nesta execução | Custo novo reportado | Custo acumulado dos 100 documentos |
|---|---:|---:|---|---:|---:|
| Ling gratuito | 0 | 0 | — | US$ 0 | US$ 0 |
| Qwen3.7 Flash / Alibaba | 415 | 435 | 20 HTTP 429 | US$ 0,707039 | US$ 1,022083 |
| Gemma 4 26B A4B / Darkbloom | 300 | 303 | 2 HTTP 502 e 1 erro 502 dentro da resposta | US$ 0,309947 | US$ 0,618157 |
| **Total** | **715** | **738** | **23** | **US$ 1,016986** | **US$ 1,640240** |

Custos são os valores reportados nas respostas salvas; não incluem taxas de compra de créditos nem calibrações/dev anteriores. O acumulado inclui as gerações do benchmark que já pertenciam a estes 100 documentos.

O comando começou às 13:51:47 e a última geração terminou às 14:03:17 (America/Sao_Paulo): **11min30s**, antes da auditoria final. A validação inicial levou aproximadamente 3min24s, sem geração via API. As chamadas dos dois modelos começaram em paralelo às 13:55:11.

| Modelo | Intervalo da primeira chamada nova à última conclusão | Vazão efetiva de respostas novas |
|---|---:|---:|
| Qwen | 8min06s | 51,2/min |
| Gemma | 6min35s | 45,5/min |

O intervalo do Gemma inclui a interrupção, auditoria, revalidação local e retomada. Esses números medem esta retomada parcial; não são um benchmark controlado entre modelos, nem o tempo de 600 chamadas novas por modelo.

## O que o teste revelou

- O teto de 500 RPM não se sustentou no Qwen. Os 429 fizeram o executor reduzir o ritmo, chegando a 50 RPM, com aumentos graduais após sucessos. Todos os 20 erros foram recuperados automaticamente.
- Dois HTTP 502 do Gemma entraram no retry automático. Outra resposta veio com `finish_reason=error`, erro interno 502 e `provider_unavailable`: o provedor desconectou após produzir apenas `during the year`. O executor marcou `audit_required` e interrompeu o modelo.
- Essa resposta incompleta também reportou 44.164 tokens de entrada, divergindo dos 38.641 esperados. Ela **não foi aceita**. A resposta original e seus dados foram preservados; a auditoria confirmou a desconexão explícita e liberou uma nova tentativa do mesmo prompt. A nova resposta passou pelo contrato integral, incluindo igualdade de tokens.
- Após a auditoria, `gemma-batch --batch 1 --execute` completou somente as 109 pendências. As 491 respostas já aceitas foram preservadas.
- **Pendência de automação:** tratar erros transitórios explícitos dentro de uma resposta HTTP bem-sucedida. Nesta execução esse caso exigiu auditoria; não foi implementada uma correção de código durante a run.

## Auditoria final

Conferimos os pedidos salvos contra os prompts congelados e revalidamos modelo, provedor, contagem de tokens, custo e saída de cada resposta aceita. Cada modelo tem exatamente os índices 0–599, os mesmos 100 documentos e 100 respostas de cada braço. Não há aceitações duplicadas, pendências ou casos fora do lote.

Ling e Qwen: 600 términos `stop` cada. Gemma: 598 `stop` e **2 `length`**, preservados conforme o protocolo: FCCY/C1 (já existente) e SFNC/C3 (novo). Portanto, completar o lote não significa que todas as saídas terminaram naturalmente nem aprovar sua qualidade factual.

BERTScore, ROUGE e hipóteses finais não foram calculados nesta execução. Este relatório avalia o processo de geração; a rotina final continua condicionada à base completa.

O processo principal registrou a interrupção inicial do Gemma e terminou com código 1. Depois da retomada independente, uma auditoria integral conciliou o estado, arquivou o plano e marcou a rodada como completa. O erro original foi preservado. Uma próxima execução de `bash rodar_rodada.sh 100` selecionará o próximo grupo; ele não foi iniciado aqui.

Evidências locais (não versionadas):

- `outputs/full-rounds/audit-first100-adaptive-20260927.json`
- `outputs/full-rounds/last-result.json` e `history/1c584d376e2d57bba371d6ed90d89c9de35d12b48d450c4eb82c1000412c9531.json`
- `outputs/full-rounds/first100-initial-interruption-result.json`
- `outputs/full-gemma-batches/gemma26/audit-0346-provider-disconnected.json`
- Ledgers e respostas em `outputs/full-{ling,qwen,gemma}-batches/`

O código executado tinha 303 testes aprovados na validação anterior, excluindo `slow` e `test_real_data.py`. Nenhum código de execução ou configuração científica foi alterado nesta rodada; não houve novo teste unitário nem commit/push.
