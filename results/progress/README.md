# Acompanhamento do experimento FINDSum

![Painel das métricas](dashboard.png)

**100/1000 documentos no prefixo comum completo.** Atualização local automática pelo Bash, sem chamadas de geração adicionais para as métricas.

Somente os documentos consecutivos da ordem congelada com seis respostas aceitas em todos os modelos entram na matriz. Pendências de um modelo não alteram a base de comparação. Progresso individual aparece nos cartões.

Resultados descritivos: sem p-valores ou confirmação de hipóteses. A rotina confirmatória final permanece separada. BERTScore XLNet-base-cased, camada 5, sem IDF/rescale, textos integrais; ROUGE-L F1; METEOR. Todos os eixos partem de zero; seus limites estão explícitos no painel.

| Modelo | Braço | BERT P | BERT R | BERT F1 | ROUGE-L | METEOR |
|---|---|---:|---:|---:|---:|---:|
| Ling Flash Fin | C1 | 0.7216 | 0.6064 | 0.6581 | 0.1615 | 0.1652 |
| Ling Flash Fin | C1t | 0.6601 | 0.5493 | 0.5979 | 0.1201 | 0.1176 |
| Ling Flash Fin | C2 | 0.6977 | 0.5689 | 0.6259 | 0.1439 | 0.1340 |
| Ling Flash Fin | C3 | 0.6798 | 0.5546 | 0.6100 | 0.1385 | 0.1321 |
| Ling Flash Fin | C4 | 0.6920 | 0.5674 | 0.6224 | 0.1456 | 0.1387 |
| Ling Flash Fin | C5 | 0.6953 | 0.5771 | 0.6299 | 0.1484 | 0.1479 |
| Qwen3.7 Flash | C1 | 0.7270 | 0.6164 | 0.6663 | 0.1654 | 0.1677 |
| Qwen3.7 Flash | C1t | 0.6830 | 0.5776 | 0.6244 | 0.1310 | 0.1356 |
| Qwen3.7 Flash | C2 | 0.7032 | 0.5770 | 0.6327 | 0.1415 | 0.1325 |
| Qwen3.7 Flash | C3 | 0.7204 | 0.5603 | 0.6288 | 0.1356 | 0.1081 |
| Qwen3.7 Flash | C4 | 0.7290 | 0.5745 | 0.6411 | 0.1426 | 0.1228 |
| Qwen3.7 Flash | C5 | 0.7369 | 0.5800 | 0.6473 | 0.1439 | 0.1202 |
| Gemma 4 26B A4B | C1 | 0.7156 | 0.5717 | 0.6341 | 0.1458 | 0.1203 |
| Gemma 4 26B A4B | C1t | 0.6802 | 0.5140 | 0.5819 | 0.1131 | 0.0933 |
| Gemma 4 26B A4B | C2 | 0.6846 | 0.5417 | 0.6036 | 0.1232 | 0.0987 |
| Gemma 4 26B A4B | C3 | 0.6958 | 0.5186 | 0.5931 | 0.1131 | 0.0778 |
| Gemma 4 26B A4B | C4 | 0.6994 | 0.5276 | 0.6000 | 0.1185 | 0.0874 |
| Gemma 4 26B A4B | C5 | 0.6980 | 0.5346 | 0.6040 | 0.1214 | 0.0936 |

[Scores por documento](scores.csv) · [Resumo e cobertura](summary.json) · [Histórico cumulativo](history.json)

C1: fonte inteira; C1t: prefixo equiparado a C2; C2: RAG sem exemplos; C3/C4/C5: mesmo RAG com quatro exemplos fixos/aleatórios/similares. Saídas `length` são mantidas. Precisão/recall/F1 são médias individuais, não F1 derivado das médias.

Execute `bash rodar_rodada.sh --metrics-only` para atualizar sem geração. O script não faz commit nem push automaticamente. O cálculo usa cache local por conteúdo e método; artefatos brutos e credenciais não são exportados.

Os pontos históricos são recalculados com os dados atuais e representam médias acumuladas, não réplicas independentes. Não ajustar o protocolo com base neste acompanhamento da coorte eval.
