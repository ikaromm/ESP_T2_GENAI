# Teste real de concorrência OpenRouter — 27/09/2026

> Atualização posterior: o usuário escolheu teto de 500 RPM com retry adaptativo. Essa política foi integrada aos lotes/Bash, com modelos paralelos. As medições abaixo permanecem históricas; o comportamento atual está em [retry adaptativo](retry-adaptativo-openrouter.md).

**Resultado: 100 requisições/minuto por modelo foi o único patamar testado sem erros nos dois modelos.** 200 e 500 RPM produziram HTTP 429. Não avançamos para 300–400 ou 600–1.000, pois a subida dependia de concluir o patamar anterior sem erros. Isso identifica uma configuração promissora nesta amostra, não o limite oficial dos provedores.

## Método

Qwen3.7 Flash/Alibaba e Gemma 4 26B A4B/Darkbloom rodaram simultaneamente. Cada patamar previa 100 gerações de prompts reais ainda pendentes dos primeiros 100 documentos da coorte congelada. Um documento exige seis gerações. Nenhum resultado aceito foi repetido; não houve troca de modelos, provedores, prompts, temperatura ou limites de tokens.

O executor usou até 100 chamadas simultâneas por modelo, com espaçamento de envios e janela móvel de 60 segundos. Um coordenador exclusivo por modelo gravou requests, ledger, respostas e reservas de custo. Ao primeiro erro, cessaram novos envios naquele modelo; chamadas já enviadas terminaram e falhas transitórias foram recuperadas pelo executor sequencial de retries. O teste não ignorou limites do provedor nem alterou o saldo.

A primeira tentativa foi a 500 RPM. Como ambos falharam, usamos novos casos a 100 RPM; ambos passaram e avançaram a 200, onde ambos falharam. As duas etapas começaram em paralelo nos dois modelos. A validação de todos os prompts é local e anterior às chamadas.

## Medições antes dos retries

“Tempo” inclui envio e espera pela última resposta do patamar, mas exclui recuperação posterior. “Envio” é a taxa medida entre a primeira e a última submissão local, não throughput garantido pelo provedor. “Conclusão” inclui os tempos de espera.

| Modelo | Teto RPM | Envios | Aceitas | HTTP 429 | Tempo (s) | Envio RPM | Conclusão RPM | Pico simultâneo | Latência média / p95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma26 | 100 | 100 | 100 | 0 | 86.0 | 99.3 | 69.8 | 41 | 21.6 / 42.0 |
| gemma26 | 200 | 100 | 99 | 1 | 79.4 | 195.4 | 74.8 | 68 | 22.6 / 45.8 |
| gemma26 | 500 | 100 | 99 | 1 | 86.5 | 480.3 | 68.7 | 88 | 25.2 / 52.4 |
| qwen37 | 100 | 100 | 100 | 0 | 67.2 | 99.3 | 89.3 | 13 | 6.0 / 10.7 |
| qwen37 | 200 | 42 | 39 | 3 | 19.9 | 196.3 | 117.7 | 21 | 5.6 / 9.9 |
| qwen37 | 500 | 43 | 33 | 10 | 18.5 | 481.5 | 107.0 | 43 | 6.9 / 11.3 |

O Qwen interrompeu novos envios após 43 solicitações no patamar 500 e 42 no patamar 200. No Gemma, o primeiro erro chegou quando as 100 solicitações já tinham sido enviadas. Por isso, os denominadores diferem. As amostras de documentos também diferem entre patamares: não é uma comparação controlada de velocidade por texto idêntico.

## Diagnóstico e recuperação

Todos os 15 erros foram HTTP 429 com `limit_source=upstream_provider_shared_pool`: 13 na Alibaba e 2 na Darkbloom. O Qwen não trouxe Retry-After; os erros Gemma trouxeram Retry-After de 2 segundos. Todos foram recuperados, sem respostas pendentes ou rejeitadas. Não houve erro de contagem local/API. A capacidade do pool pode variar; não podemos afirmar que 200 RPM seja um limite fixo da conta.

O saldo disponível antes da primeira etapa era US$ 9,542300778; reserva conservadora para 1.200 gerações: US$ 7,235174400. Não ocorreu HTTP 402. A documentação distingue restrições de saldo, reserva de chamadas simultâneas e limites do provedor: [OpenRouter — Limits](https://openrouter.ai/docs/api_reference/limits).

## Resultados preservados no experimento

| Modelo | Tentativas HTTP | Gerações aceitas | Documentos com seis braços | Pendências no lote de 600 | Custo reportado US$ |
|---|---:|---:|---:|---:|---:|
| qwen37 | 198 | 185 | 30 | 415 | 0.315044 |
| gemma26 | 302 | 300 | 50 | 300 | 0.308210 |

**Total: 500 tentativas, 485 gerações válidas e custo reportado de US$ 0.623254042.** Custos incluem os retries com resposta; HTTP 429 não trouxe usage/custo. As duas janelas de execução, incluindo recuperação, somaram 288,6 segundos (4min49), além de pré-validação local e preparação do teste. O Ling permaneceu com suas 600 gerações anteriores.

O Qwen terminou 185 respostas por stop. O Gemma terminou 299 por stop e uma por length; essa saída foi preservada conforme o protocolo. Isso não é aprovação de qualidade factual. Não foram calculadas métricas nem hipóteses neste teste operacional.

## Decisão operacional sugerida

- Usar inicialmente 100 RPM por modelo com concorrência controlada e redução de carga diante de 429. O teste de 100 RPM atingiu aproximadamente 99 RPM de envio, com picos simultâneos de 13 no Qwen e 41 no Gemma.
- Não aumentar para 500–1.000 a partir desta evidência. Patamares com 100 solicitações são testes curtos, especialmente a 500 RPM, cujo envio dura cerca de 12 segundos; não comprovam vazão sustentada de um minuto ou mais.
- Extrapolação simples de seis grupos semelhantes ao patamar de 100: Qwen ~6min43 e Gemma ~8min36 para 600 respostas cada. Rodando os dois simultaneamente, algo como 10–15 minutos incluindo verificação local seria uma estimativa inicial, sujeita a variação de tamanho, filas e retries; não uma medição de um lote completo.
- O Bash normal `rodar_rodada.sh` ainda usa geração sequencial a 100 RPM. A concorrência foi aplicada no executor de teste `scripts/benchmark_paid_rates.py`; a projeção acima exige execução concorrente equivalente.

## Reprodutibilidade

Código: `scripts/benchmark_paid_rates.py`. Execuções feitas com `--execute` e depois `--lower-rates --execute`. Planos, índices dos casos, medições e relatórios JSON em `outputs/paid-rate-benchmark-20260927/` e `lower-rates/`. Requests/respostas ficam nos ledgers oficiais `outputs/full-qwen-batches/qwen37/` e `outputs/full-gemma-batches/gemma26/`. Não iniciar novas gerações para reproduzir esta tabela: os JSONs já contêm as medições.

294 testes passaram ao final, incluindo os cinco testes do benchmark (concorrência, orçamento em voo, auditoria, ritmo e seleção de pendências). Ruff e diff check passaram. Lock ativo: `5b14fd8a82480f12ff3765f487135e55ad5ee742ed7d14a3efda48f0a1086f65`. As preparações anteriores dos três modelos foram preservadas por compatibilidade explícita. Alterações locais, ainda sem commit/push.
