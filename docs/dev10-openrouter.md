# Dev10 integrado — 26/09/2026

Dez casos sorteados sem reposição no dev (semente 42), os mesmos seis braços nos
três modelos: Ling Flash Fin/Novita gratuito, Qwen3.7 Flash/Alibaba,
Gemma 4 26B A4B/Darkbloom. O manifesto original e as rodadas históricas são
preservados. A orientação de até 750 palavras permanece em todos os prompts.

## Avaliação integral

BERTScore passou a usar xlnet-base-cased, camada 5 (padrão da biblioteca
bert-score 0.3.13), sem IDF e sem reescala. Não é comparável diretamente aos
escores antigos com roberta-large. ROUGE-1/2/L permanece igual. Nenhuma
avaliação numérica/factual ou humana integra o fluxo.

XLNet usa posições relativas. Seu tokenizer declara um sentinela muito grande
para o limite, incompatível com o inteiro do backend em transformers 5.17.
O código ajusta esse sentinela ao número de tokens do MAIOR texto da rodada.
Antes de pontuar, compara os IDs completos com os IDs que o BERTScore usará;
qualquer divergência interrompe a avaliação. Não encurta a referência nem a
resposta. O limite é operacional e derivado dos dados, não um corte em 512.
Batch 1 limita memória de processamento, sem limitar comprimento textual.

O teste real com 810/814 tokens preservou todos os tokens. Modificar o trecho
posterior ao token 512 alterou o escore (1,0 versus 0,9959317), demonstrando que
aquela parte participa da avaliação. Isso valida o caminho técnico, não a
qualidade financeira da métrica.

## Execução e retomada

Arquivos do lote: `outputs/dev10-complete-20260926/`.
Config: `configs/dev10_openrouter_complete.yaml`.

```bash
uv run --no-sync python scripts/prepare_full_openrouter.py \
  --config configs/dev10_openrouter_complete.yaml \
  --output outputs/dev10-complete-20260926/prepared

# A credencial precisa estar no ambiente; nunca imprimi-la.
uv run --no-sync python scripts/prepare_qwen_remote.py \
  --prepared outputs/dev10-complete-20260926/prepared \
  --output outputs/dev10-complete-20260926/qwen-prepared --budget-usd 1

uv run --no-sync python scripts/run_experiment_openrouter.py \
  --prepared outputs/dev10-complete-20260926/prepared \
  --qwen-prepared outputs/dev10-complete-20260926/qwen-prepared \
  --output outputs/dev10-complete-20260926/run \
  --qwen-budget 1 --gemma-budget 0.5 --execute

uv run --no-sync python outputs/dev10-complete-20260926/audit_run.py

uv run --no-sync python scripts/build_dev_review.py \
  --root outputs/dev10-complete-20260926
```

A preparação local exige pasta nova; não rerodá-la sobre a pasta pronta.
Calibração e geração têm ledger e retomam chamadas aceitas sem repeti-las.
Sem `--execute`, o executor só valida. Eval exige permissão explícita pelo
parâmetro `--allow-eval`; não foi executado. A preparação eval também exige
`--full` e os 1.000 casos reservados. Não há filtro por qualidade de resposta.

Prompts integrais, exemplos, reservas de saída e margem são validados antes
da geração. C1t/C2 iguais em tokens de contexto e prompt; contexto idêntico
C2–C5 dentro de cada modelo. O Qwen mantém a contagem operacional por sondas
API, sem alegar tokenizer local. Reserva de saída 8.192; entrada 49.152.

Retries limitados para HTTP 408/429/500/502/503/504, esperas 1/2/4/8/16,
respeitando Retry-After e o circuit breaker existente. Falhas ambíguas de
transporte ficam para auditoria: não reenviar automaticamente algo que pode
ter sido cobrado. Orçamento reserva cada chamada antes de enviar.
Respostas `length` são preservadas, sinalizadas e pontuadas como produzidas;
respostas `stop` também. Não há nova geração para buscar melhor escore.

Tetos desta rodada: US$ 1 calibração Qwen + US$ 1 geração Qwen + US$ 0,50 Gemma.
Ling usa somente endpoint gratuito, max_price 0 e sem fallback pago automático.
O site de revisão apresenta custos efetivos, não a soma desses tetos.

A página `docs/dev10.html` permite escolher documento/modelo/braço, ler fonte,
contexto, pares demonstrativos e mensagens, comparar resposta com referência,
baixar escores e verificar cobertura de tokens por caso. Os textos são tratados
como dados, não como HTML executável.


## Resultado verificado

180 respostas em 180 chamadas de geração, sem erros HTTP ou retries. Ling e
Qwen: 60/60 `stop` cada. Gemma: 59 `stop` e 1 `length` (TNET/C3), preservada
na análise. Essa saída chegou a 8.192 tokens; sua extensão não foi corrigida.
A orientação de 750 palavras foi excedida em dois casos Ling e um Gemma.
Quatro respostas Gemma têm menos de 50 palavras; são observações, não exclusões.

ROUGE e BERTScore calculados em todos os 180 pares. Auditoria sem cortes:
maior referência 1.384 tokens XLNet, maior previsão 4.406. Os controles
C1t/C2, C2–C5, igualdade de contagem do prompt/API e hash do manifesto passaram.
255 testes locais passaram, excluindo slow e test_real_data.py. HTML conferido
nas 180 combinações no Chromium, sem erro JavaScript e sem overflow em 390px.

Custos reportados: Ling US$ 0; geração Qwen US$ 0,037401610; geração Gemma
US$ 0,061966446; calibração Qwen US$ 0,105871182 (332 sondas).
Total US$ 0,205239238, sem taxas de compra de créditos.

Página pública: https://findsum.206-189-224-216.nip.io/dev10.html.
Nenhum full/eval foi executado; nenhum resultado de hipótese foi confirmado.
