# Métricas parciais dos primeiros 100 documentos — 27/09/2026

Calculadas localmente sobre 1.800 respostas: 100 documentos × seis braços × três modelos. Mesmos documentos e referências FINDSum. Nenhuma chamada de geração adicional. Análise descritiva da coorte eval parcial; nenhum teste de hipótese ou p-valor foi calculado.

Modelos: `ling-free` = Ling 3.0 Flash Fin gratuito / Novita; `qwen37` = Qwen3.7 Flash / Alibaba; `gemma26` = Gemma 4 26B A4B / Darkbloom. Os três usam os mesmos IDs da coorte congelada; não houve seleção dos casos pelo resultado das métricas.

Configurações: C1 = fonte inteira; C1t = prefixo com orçamento igual ao contexto de C2; C2 = RAG sem exemplos; C3 = mesmo RAG com quatro exemplos fixos; C4 = mesmo RAG com quatro exemplos aleatórios; C5 = mesmo RAG com quatro exemplos por similaridade.

BERTScore usa XLNet-base-cased, camada 5, sem IDF/rescale, textos integrais. Precisão (P), recall (R) e F1 são médias por documento, não o F1 calculado das médias de P/R. ROUGE-L é F1. METEOR usa o método existente do projeto. Escala 0–1; maior indica maior similaridade à referência, não garantia de exatidão factual.

| Modelo | Braço | BERT P | BERT R | BERT F1 | ROUGE-L | METEOR |
|---|---|---:|---:|---:|---:|---:|
| ling-free | C1 | 0.7216 | 0.6064 | 0.6581 | 0.1615 | 0.1652 |
| ling-free | C1t | 0.6601 | 0.5493 | 0.5979 | 0.1201 | 0.1176 |
| ling-free | C2 | 0.6977 | 0.5689 | 0.6259 | 0.1439 | 0.1340 |
| ling-free | C3 | 0.6798 | 0.5546 | 0.6100 | 0.1385 | 0.1321 |
| ling-free | C4 | 0.6920 | 0.5674 | 0.6224 | 0.1456 | 0.1387 |
| ling-free | C5 | 0.6953 | 0.5771 | 0.6299 | 0.1484 | 0.1479 |
| qwen37 | C1 | 0.7270 | 0.6164 | 0.6663 | 0.1654 | 0.1677 |
| qwen37 | C1t | 0.6830 | 0.5776 | 0.6244 | 0.1310 | 0.1356 |
| qwen37 | C2 | 0.7032 | 0.5770 | 0.6327 | 0.1415 | 0.1325 |
| qwen37 | C3 | 0.7204 | 0.5603 | 0.6288 | 0.1356 | 0.1081 |
| qwen37 | C4 | 0.7290 | 0.5745 | 0.6411 | 0.1426 | 0.1228 |
| qwen37 | C5 | 0.7369 | 0.5800 | 0.6473 | 0.1439 | 0.1202 |
| gemma26 | C1 | 0.7156 | 0.5717 | 0.6341 | 0.1458 | 0.1203 |
| gemma26 | C1t | 0.6802 | 0.5140 | 0.5819 | 0.1131 | 0.0933 |
| gemma26 | C2 | 0.6846 | 0.5417 | 0.6036 | 0.1232 | 0.0987 |
| gemma26 | C3 | 0.6958 | 0.5186 | 0.5931 | 0.1131 | 0.0778 |
| gemma26 | C4 | 0.6994 | 0.5276 | 0.6000 | 0.1185 | 0.0874 |
| gemma26 | C5 | 0.6980 | 0.5346 | 0.6040 | 0.1214 | 0.0936 |

## Diferenças descritivas dos contrastes

Variação percentual relativa: 100 × (média da configuração comparada / média da base − 1). Não são pontos percentuais nem resultados de significância. H3 continua sendo controle de sensibilidade.

| Comparação | Ling BERT F1 / ROUGE-L | Qwen BERT F1 / ROUGE-L | Gemma BERT F1 / ROUGE-L |
|---|---:|---:|---:|
| H1: C2 vs C1 | -4.89% / -10.87% | -5.04% / -14.50% | -4.82% / -15.51% |
| H1b: C2 vs C1t | +4.69% / +19.87% | +1.34% / +7.98% | +3.72% / +8.86% |
| H2: C3 vs C2 | -2.54% / -3.76% | -0.63% / -4.15% | -1.74% / -8.16% |
| H3: C4 vs C3 | +2.04% / +5.10% | +1.96% / +5.21% | +1.17% / +4.74% |
| H4: C5 vs C4 | +1.20% / +1.97% | +0.97% / +0.87% | +0.67% / +2.47% |

## Leitura dos resultados

- C1 lidera BERTScore F1, ROUGE-L e METEOR em todos os modelos. C2 supera C1t nas mesmas três métricas, mas fica abaixo de C1.
- Qwen tem maior BERTScore F1 em todos os seis braços. Isso não significa domínio em todas as métricas: Ling supera Qwen em ROUGE-L e METEOR de C2 a C5.
- C3 reduz F1 nos três modelos. C4 recupera parte da diferença e C5 aumenta novamente F1/ROUGE-L, em média. Os ganhos não são monotônicos em toda métrica: METEOR do Qwen cai de C4 para C5.
- Em Qwen e Gemma, C3 aumenta precisão BERT e reduz recall em relação a C2. Os exemplos fixos mudaram esse equilíbrio; esses resultados não identificam sozinhos a causa.
- Duas saídas Gemma terminaram em length e continuam incluídas, conforme protocolo: FCCY/C1 e SFNC/C3.

## Evidência e limites

Os 1.800 pedidos/respostas já tinham passado pela auditoria operacional. O cálculo verificou os hashes dos ledgers, os mesmos 100 IDs por modelo, 1.800 chaves únicas e 100 respostas por grupo. As médias foram reconciliadas separadamente a partir do CSV. As 3.600 entradas de auditoria de tokens confirmam avaliação integral, sem truncamento do BERTScore.

Arquivos versionados: [métricas por documento](../results/first100-20260927/scores.csv), [médias e contrastes](../results/first100-20260927/summary.json), [coorte](../results/first100-20260927/cohort.json) e [validação](../results/first100-20260927/validation.json).

Resultados locais: `outputs/first100-metrics-20260927/{calculate.py,scores.csv,summary.json,validation.json}`. Os artefatos preservam hashes das fontes, parâmetros e auditoria de tokens. Tempo de cálculo: 142.6 segundos. Não foram modificados os prompts, configurações, gerações ou o plano de análise final. Os testes confirmatórios continuam reservados à base completa de 1.000 documentos; este recorte não deve orientar ajustes que contaminem o experimento.
