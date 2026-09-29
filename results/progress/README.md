# Acompanhamento do experimento FINDSum

![Painel das métricas](dashboard.png)

<!-- readable-progress:start -->

**Gráficos ampliados:** os quatro painéis têm eixos verticais focados na faixa observada e limites visíveis. Abra os SVGs para ampliar sem perda de nitidez.

| Métrica | Figura detalhada | Versão vetorial |
|---|---|---|
| BERTScore F1 | [PNG](bertscore-f1.png) | [SVG](bertscore-f1.svg) |
| ROUGE-L F1 | [PNG](rouge-l-f1.png) | [SVG](rouge-l-f1.svg) |
| METEOR | [PNG](meteor.png) | [SVG](meteor.svg) |
| Evolução cumulativa | [PNG](evolucao.png) | [SVG](evolucao.svg) |

![BERTScore F1 ampliado](bertscore-f1.png)

![ROUGE-L F1 ampliado](rouge-l-f1.png)

![METEOR ampliado](meteor.png)

![Evolução cumulativa ampliada](evolucao.png)

## Hipóteses e observações parciais

Cada contraste usa os **mesmos documentos** em dois braços. Δ positivo significa maior similaridade média com a referência no primeiro braço; Δ negativo, menor. As diferenças abaixo são pontos da escala 0 a 1, não percentuais de acerto.

| Hipótese | Comparação | Pergunta |
|---|---|---|
| H1 | C2 - C1 | RAG versus fonte inteira |
| H1b | C2 - C1t | RAG versus prefixo de igual orçamento |
| H2 | C3 - C2 | Quatro exemplos fixos versus zero-shot RAG |
| H3 | C4 - C3 | Aleatórios versus fixos: controle de sensibilidade |
| H4 | C5 - C4 | Similares versus aleatórios |

H3 é um **controle de sensibilidade** à escolha dos exemplos: uma diferença não significativa ao final não provaria equivalência. As métricas primárias são BERTScore F1 e ROUGE-L F1; METEOR e BERTScore precisão/recall são descritivos.

**Recorte atual: 520 documentos por modelo.** Contrastes entre médias dos braços, arredondados a quatro casas:

| Modelo | Hipótese | Δ BERTScore F1 | Δ ROUGE-L F1 |
|---|---|---:|---:|
| Ling Flash Fin | H1 | -0,0366 | -0,0214 |
| Ling Flash Fin | H1b | +0,0322 | +0,0275 |
| Ling Flash Fin | H2 | -0,0182 | -0,0090 |
| Ling Flash Fin | H3 | +0,0119 | +0,0081 |
| Ling Flash Fin | H4 | +0,0040 | +0,0017 |
| Qwen3.7 Flash | H1 | -0,0388 | -0,0229 |
| Qwen3.7 Flash | H1b | +0,0060 | +0,0124 |
| Qwen3.7 Flash | H2 | -0,0043 | -0,0116 |
| Qwen3.7 Flash | H3 | +0,0137 | +0,0112 |
| Qwen3.7 Flash | H4 | +0,0077 | +0,0033 |
| Gemma 4 26B A4B | H1 | -0,0365 | -0,0243 |
| Gemma 4 26B A4B | H1b | +0,0302 | +0,0167 |
| Gemma 4 26B A4B | H2 | -0,0101 | -0,0072 |
| Gemma 4 26B A4B | H3 | +0,0094 | +0,0069 |
| Gemma 4 26B A4B | H4 | +0,0067 | +0,0061 |

Nos três modelos, **H1 é negativa** e **H1b é positiva**: o RAG ficou abaixo da fonte inteira, mas acima do prefixo com o mesmo orçamento de contexto. **H2 é negativa**: quatro exemplos fixos reduziram as duas métricas primárias frente ao RAG sem exemplos. **H3 e H4 são positivas**, com ganho menor em H4. Isso descreve este recorte e não estabelece eficácia causal ou qualidade factual.

**Nenhuma hipótese foi confirmada ou refutada aqui.** O teste predefinido usa Wilcoxon bilateral emparelhado e Holm sobre dez testes por modelo, somente depois dos 1.000 documentos. As saídas `length` permanecem na análise e podem afetar as médias. Os contrastes entre modelos não isolam arquitetura, tokenizador ou provedor.

<!-- readable-progress:end -->

**520/1000 documentos no prefixo comum completo.** Atualização local automática pelo Bash, sem chamadas de geração adicionais para as métricas.

Somente os documentos consecutivos da ordem congelada com seis respostas aceitas em todos os modelos entram na matriz. Pendências de um modelo não alteram a base de comparação. Progresso individual aparece nos cartões.

Resultados descritivos: sem p-valores ou confirmação de hipóteses. A rotina confirmatória final permanece separada. BERTScore XLNet-base-cased, camada 5, sem IDF/rescale, textos integrais; ROUGE-L F1; METEOR. Os eixos ampliados e seus limites estão explícitos em cada figura.

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
