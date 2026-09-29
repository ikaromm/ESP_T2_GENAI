# Auditoria dos primeiros 520 documentos

Os **520 primeiros documentos da coorte eval congelada** têm seis respostas aceitas em cada um dos três modelos: **9.360 respostas e 9.360 linhas de métricas**. Esta é uma análise parcial e descritiva; os testes confirmatórios H1, H1b, H2, H3 e H4 ficam para os 1.000 documentos completos.

A rodada 321–520 adicionou 200 documentos, ou 1.200 respostas por modelo. O custo abaixo soma `usage.cost` das respostas aceitas, em dólares americanos. “Retido” é o limite conservador usado pelo executor: considera o custo real das respostas aceitas e reserva o máximo permitido para cada tentativa falha, mesmo quando não há custo reportado para ela.

| Modelo e provedor fixo | Respostas | 429 refeitos | Custo reportado | Custo retido | Teto incremental |
|---|---:|---:|---:|---:|---:|
| Ling Flash Fin pago / Novita | 1.200 | 0 | US$ 0,9668 | US$ 0,9668 | US$ 1,70 |
| Qwen3.7 Flash / Alibaba | 1.200 | 67 | US$ 2,0183 | US$ 2,5671 | US$ 3,40 |
| Gemma 4 26B A4B / Darkbloom | 1.200 | 7 | US$ 1,2265 | US$ 1,2536 | US$ 1,90 |
| **Total** | **3.600** | **74** | **US$ 4,2116** | **US$ 4,7875** | **US$ 7,00** |

Houve 3 saídas `length` no Ling e 28 no Gemma nessa rodada; elas foram mantidas e marcadas. Nenhum dos 9.360 pares avaliados teve truncamento no BERTScore. O prefixo dos 320 documentos anteriores preservou scores e hashes dos arquivos brutos. O [painel](../progress/README.md) contém as médias por braço e modelo, sem p-valores.

O [relatório de auditoria](audit.json), o [snapshot dos scores](scores-520.csv), o [snapshot do resumo e auditoria de tokens](summary-520.json) e o [manifesto SHA-256 dos prompts e respostas](raw-artifact-manifest.csv) permitem verificar o recorte mesmo depois de novas rodadas. Os textos brutos, referências, prompts e respostas ficam em `outputs/`, fora do Git. Para refazer a auditoria local sem chamar a API:

```bash
uv run --locked python results/audit520/audit.py
```

O script confere coorte e splits, lock, parâmetros e prompts preparados, igualdade C1t/C2, RAG compartilhado em C2–C5, seleção de exemplos, identidade de modelo/provedor, adendos do Ling pago, tentativas e retries, custos, respostas, cache e cobertura de tokens. Ele cruza os arquivos brutos locais com os hashes versionados. Não recalcula embeddings/FAISS nem o BERTScore, e as métricas não avaliam correção factual.
