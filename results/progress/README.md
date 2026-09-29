# Acompanhamento do experimento FINDSum

![Painel das métricas](dashboard.png)

**320/1000 documentos no prefixo comum completo.** Atualização local automática pelo Bash, sem chamadas de geração adicionais para as métricas.

Somente os documentos consecutivos da ordem congelada com seis respostas aceitas em todos os modelos entram na matriz. Pendências de um modelo não alteram a base de comparação. Progresso individual aparece nos cartões.

Resultados descritivos: sem p-valores ou confirmação de hipóteses. A rotina confirmatória final permanece separada. BERTScore XLNet-base-cased, camada 5, sem IDF/rescale, textos integrais; ROUGE-L F1; METEOR. Todos os eixos partem de zero; seus limites estão explícitos no painel.

| Modelo | Braço | BERT P | BERT R | BERT F1 | ROUGE-L | METEOR |
|---|---|---:|---:|---:|---:|---:|
| Ling Flash Fin | C1 | 0.7286 | 0.6109 | 0.6636 | 0.1671 | 0.1702 |
| Ling Flash Fin | C1t | 0.6569 | 0.5486 | 0.5959 | 0.1190 | 0.1165 |
| Ling Flash Fin | C2 | 0.7029 | 0.5704 | 0.6288 | 0.1470 | 0.1342 |
| Ling Flash Fin | C3 | 0.6821 | 0.5564 | 0.6120 | 0.1389 | 0.1299 |
| Ling Flash Fin | C4 | 0.6908 | 0.5663 | 0.6214 | 0.1453 | 0.1368 |
| Ling Flash Fin | C5 | 0.6984 | 0.5697 | 0.6265 | 0.1475 | 0.1390 |
| Qwen3.7 Flash | C1 | 0.7322 | 0.6174 | 0.6688 | 0.1671 | 0.1647 |
| Qwen3.7 Flash | C1t | 0.6800 | 0.5810 | 0.6250 | 0.1328 | 0.1328 |
| Qwen3.7 Flash | C2 | 0.7031 | 0.5768 | 0.6326 | 0.1454 | 0.1355 |
| Qwen3.7 Flash | C3 | 0.7220 | 0.5562 | 0.6268 | 0.1344 | 0.1039 |
| Qwen3.7 Flash | C4 | 0.7314 | 0.5766 | 0.6432 | 0.1454 | 0.1228 |
| Qwen3.7 Flash | C5 | 0.7405 | 0.5805 | 0.6491 | 0.1485 | 0.1257 |
| Gemma 4 26B A4B | C1 | 0.7138 | 0.5806 | 0.6387 | 0.1478 | 0.1259 |
| Gemma 4 26B A4B | C1t | 0.6710 | 0.5093 | 0.5746 | 0.1084 | 0.0904 |
| Gemma 4 26B A4B | C2 | 0.6820 | 0.5430 | 0.6034 | 0.1246 | 0.0999 |
| Gemma 4 26B A4B | C3 | 0.6945 | 0.5202 | 0.5937 | 0.1165 | 0.0810 |
| Gemma 4 26B A4B | C4 | 0.6983 | 0.5318 | 0.6024 | 0.1228 | 0.0920 |
| Gemma 4 26B A4B | C5 | 0.7032 | 0.5396 | 0.6092 | 0.1299 | 0.1019 |

[Scores por documento](scores.csv) · [Resumo e cobertura](summary.json) · [Histórico cumulativo](history.json)

C1: fonte inteira; C1t: prefixo equiparado a C2; C2: RAG sem exemplos; C3/C4/C5: mesmo RAG com quatro exemplos fixos/aleatórios/similares. Saídas `length` são mantidas. Precisão/recall/F1 são médias individuais, não F1 derivado das médias.

Execute `bash rodar_rodada.sh --metrics-only` para atualizar sem geração. O script não faz commit nem push automaticamente. O cálculo usa cache local por conteúdo e método; artefatos brutos e credenciais não são exportados.

Os pontos históricos são recalculados com os dados atuais e representam médias acumuladas, não réplicas independentes. Não ajustar o protocolo com base neste acompanhamento da coorte eval.
