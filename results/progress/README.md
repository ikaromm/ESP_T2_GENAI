# Acompanhamento do experimento FINDSum

![Painel das métricas](dashboard.png)

**520/1000 documentos no prefixo comum completo.** Atualização local automática pelo Bash, sem chamadas de geração adicionais para as métricas.

Somente os documentos consecutivos da ordem congelada com seis respostas aceitas em todos os modelos entram na matriz. Pendências de um modelo não alteram a base de comparação. Progresso individual aparece nos cartões.

Resultados descritivos: sem p-valores ou confirmação de hipóteses. A rotina confirmatória final permanece separada. BERTScore XLNet-base-cased, camada 5, sem IDF/rescale, textos integrais; ROUGE-L F1; METEOR. Todos os eixos partem de zero; seus limites estão explícitos no painel.

| Modelo | Braço | BERT P | BERT R | BERT F1 | ROUGE-L | METEOR |
|---|---|---:|---:|---:|---:|---:|
| Ling Flash Fin | C1 | 0.7279 | 0.6070 | 0.6609 | 0.1657 | 0.1663 |
| Ling Flash Fin | C1t | 0.6549 | 0.5436 | 0.5920 | 0.1168 | 0.1126 |
| Ling Flash Fin | C2 | 0.7009 | 0.5644 | 0.6243 | 0.1443 | 0.1302 |
| Ling Flash Fin | C3 | 0.6775 | 0.5498 | 0.6061 | 0.1353 | 0.1258 |
| Ling Flash Fin | C4 | 0.6889 | 0.5620 | 0.6180 | 0.1434 | 0.1342 |
| Ling Flash Fin | C5 | 0.6956 | 0.5644 | 0.6220 | 0.1451 | 0.1363 |
| Qwen3.7 Flash | C1 | 0.7314 | 0.6167 | 0.6681 | 0.1672 | 0.1660 |
| Qwen3.7 Flash | C1t | 0.6792 | 0.5785 | 0.6232 | 0.1319 | 0.1313 |
| Qwen3.7 Flash | C2 | 0.7015 | 0.5722 | 0.6292 | 0.1443 | 0.1338 |
| Qwen3.7 Flash | C3 | 0.7206 | 0.5540 | 0.6249 | 0.1327 | 0.1030 |
| Qwen3.7 Flash | C4 | 0.7294 | 0.5704 | 0.6386 | 0.1439 | 0.1210 |
| Qwen3.7 Flash | C5 | 0.7398 | 0.5764 | 0.6463 | 0.1473 | 0.1240 |
| Gemma 4 26B A4B | C1 | 0.7143 | 0.5781 | 0.6374 | 0.1470 | 0.1249 |
| Gemma 4 26B A4B | C1t | 0.6700 | 0.5040 | 0.5706 | 0.1061 | 0.0881 |
| Gemma 4 26B A4B | C2 | 0.6826 | 0.5384 | 0.6009 | 0.1227 | 0.0976 |
| Gemma 4 26B A4B | C3 | 0.6928 | 0.5168 | 0.5908 | 0.1155 | 0.0804 |
| Gemma 4 26B A4B | C4 | 0.6984 | 0.5284 | 0.6003 | 0.1224 | 0.0914 |
| Gemma 4 26B A4B | C5 | 0.7027 | 0.5363 | 0.6070 | 0.1285 | 0.0996 |

[Scores por documento](scores.csv) · [Resumo e cobertura](summary.json) · [Histórico cumulativo](history.json)

C1: fonte inteira; C1t: prefixo equiparado a C2; C2: RAG sem exemplos; C3/C4/C5: mesmo RAG com quatro exemplos fixos/aleatórios/similares. Saídas `length` são mantidas. Precisão/recall/F1 são médias individuais, não F1 derivado das médias.

Execute `bash rodar_rodada.sh --metrics-only` para atualizar sem geração. O script não faz commit nem push automaticamente. O cálculo usa cache local por conteúdo e método; artefatos brutos e credenciais não são exportados.

Os pontos históricos são recalculados com os dados atuais e representam médias acumuladas, não réplicas independentes. Não ajustar o protocolo com base neste acompanhamento da coorte eval.
