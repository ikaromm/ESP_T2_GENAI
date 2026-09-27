# Acompanhamento do experimento FINDSum

![Painel das métricas](dashboard.png)

**160/1000 documentos no prefixo comum completo.** Atualização local automática pelo Bash, sem chamadas de geração adicionais para as métricas.

Somente os documentos consecutivos da ordem congelada com seis respostas aceitas em todos os modelos entram na matriz. Pendências de um modelo não alteram a base de comparação. Progresso individual aparece nos cartões.

Resultados descritivos: sem p-valores ou confirmação de hipóteses. A rotina confirmatória final permanece separada. BERTScore XLNet-base-cased, camada 5, sem IDF/rescale, textos integrais; ROUGE-L F1; METEOR. Todos os eixos partem de zero; seus limites estão explícitos no painel.

| Modelo | Braço | BERT P | BERT R | BERT F1 | ROUGE-L | METEOR |
|---|---|---:|---:|---:|---:|---:|
| Ling Flash Fin | C1 | 0.7279 | 0.6125 | 0.6642 | 0.1661 | 0.1693 |
| Ling Flash Fin | C1t | 0.6646 | 0.5560 | 0.6036 | 0.1233 | 0.1217 |
| Ling Flash Fin | C2 | 0.7022 | 0.5736 | 0.6305 | 0.1463 | 0.1364 |
| Ling Flash Fin | C3 | 0.6812 | 0.5568 | 0.6119 | 0.1388 | 0.1315 |
| Ling Flash Fin | C4 | 0.6952 | 0.5713 | 0.6263 | 0.1473 | 0.1390 |
| Ling Flash Fin | C5 | 0.7006 | 0.5756 | 0.6311 | 0.1481 | 0.1424 |
| Qwen3.7 Flash | C1 | 0.7324 | 0.6232 | 0.6725 | 0.1690 | 0.1696 |
| Qwen3.7 Flash | C1t | 0.6842 | 0.5831 | 0.6282 | 0.1351 | 0.1360 |
| Qwen3.7 Flash | C2 | 0.7060 | 0.5791 | 0.6352 | 0.1444 | 0.1355 |
| Qwen3.7 Flash | C3 | 0.7204 | 0.5602 | 0.6290 | 0.1372 | 0.1083 |
| Qwen3.7 Flash | C4 | 0.7317 | 0.5783 | 0.6446 | 0.1465 | 0.1250 |
| Qwen3.7 Flash | C5 | 0.7399 | 0.5832 | 0.6508 | 0.1457 | 0.1236 |
| Gemma 4 26B A4B | C1 | 0.7183 | 0.5794 | 0.6398 | 0.1496 | 0.1248 |
| Gemma 4 26B A4B | C1t | 0.6811 | 0.5192 | 0.5854 | 0.1159 | 0.0961 |
| Gemma 4 26B A4B | C2 | 0.6857 | 0.5431 | 0.6049 | 0.1249 | 0.1005 |
| Gemma 4 26B A4B | C3 | 0.6975 | 0.5222 | 0.5960 | 0.1175 | 0.0811 |
| Gemma 4 26B A4B | C4 | 0.7000 | 0.5316 | 0.6029 | 0.1232 | 0.0912 |
| Gemma 4 26B A4B | C5 | 0.7000 | 0.5396 | 0.6079 | 0.1251 | 0.0984 |

[Scores por documento](scores.csv) · [Resumo e cobertura](summary.json) · [Histórico cumulativo](history.json)

C1: fonte inteira; C1t: prefixo equiparado a C2; C2: RAG sem exemplos; C3/C4/C5: mesmo RAG com quatro exemplos fixos/aleatórios/similares. Saídas `length` são mantidas. Precisão/recall/F1 são médias individuais, não F1 derivado das médias.

Execute `bash rodar_rodada.sh --metrics-only` para atualizar sem geração. O script não faz commit nem push automaticamente. O cálculo usa cache local por conteúdo e método; artefatos brutos e credenciais não são exportados.

Os pontos históricos são recalculados com os dados atuais e representam médias acumuladas, não réplicas independentes. Não ajustar o protocolo com base neste acompanhamento da coorte eval.
