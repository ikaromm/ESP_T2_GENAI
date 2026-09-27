# Preparacao do dev e do experimento — 2026-09-26

Escolha atual: Ling 3.0 Flash Fin gratuito, Qwen3.7 Flash via Alibaba e Gemma 4
26B A4B via Darkbloom. Gemma 26B e diferente do Gemma 31B da tentativa anterior.
O usuario autorizou o Ling pago como contingencia quando a cota gratuita diaria
acabar. Isso nao autoriza abrir agora a avaliacao reservada.

Retomada apos a pausa: dev50 do Ling gratuito **concluido, 300/300 resumos**, em
`outputs/dev50-ling-20260926/`, com pool completo de 1.000 exemplos e perfil
de 20.480 tokens. `report.json` registra `complete` e `validation.json` confirma
os controles. Todos os campos dos 300 prompts preparados coincidem com os
executados; o preflight da execucao tambem armazena a fonte completa.
Esta rodada e distinta da regressao anterior de cinco documentos/pool32;
os artefatos anteriores foram preservados. A avaliacao final continua fechada.

## Resultado verificado desta etapa

- Dev50 Ling gratuito: **300/300 resumos**, 300 tentativas, nenhum erro HTTP,
  custo reportado **US$ 0**. Entrada: 2.591.401 tokens; saida: 176.145. Pico
  observado pelos horarios dos arquivos: 10 tentativas/60s. Todos com `stop`,
  sem reasoning, sem truncamento e contagem local/API coincidente. Manifesto e
  hashes preparados preservados; C1t/C2 iguais em tokens de contexto/prompt e
  C2-C5 com o mesmo contexto. Consulta de cota apos a rodada: 448 usadas, 552
  restantes, limite 1.000; e um snapshot do endpoint, nao a contabilidade local
  de chamadas, e deve ser revalidado. Nenhuma chamada paga nesta retomada.
- Regressao anterior Ling gratuito: **30/30 resumos**, 31 tentativas (um 429 recuperado apos 1s),
  custo reportado zero, pico de nove chamadas por minuto. Todos encerraram com
  `stop`, sem reasoning e com contagem local/API coincidente. Cota consultada
  ao final: 151 usadas, 849 restantes, limite 1.000; revalidar antes de usar.
- Dev50 preparado em `outputs/dev50-prepared-v2-20260926/`: 50 fontes, 1.000
  exemplos, candidatos RAG e exemplos selecionados. Ling free, Ling pago e
  Gemma26 tem 300 prompts cada com preflight local aprovado. A preparacao foi
  local; depois dela, somente o dev50 Ling gratuito foi gerado. Nao houve uso
  do eval. Qwen37 tem fontes/exemplos preparados, mas nenhum
  preflight de tokens aprovado enquanto o tokenizer estiver pendente.
- Maximo de entrada: Ling 19.580; Gemma26 19.404 tokens. Perfil remoto do Ling
  ajustado de 16.384 para **20.480**, mantendo C1 inteiro. Teto de saida 3.072.
  A preparacao permitiu a janela do endpoint para medir todas as fontes; a
  validacao independente confirmou que todos os prompts cabem em 20.480.
- Custo maximo das 300 saidas, todas chegando a 3.072 tokens, sem cache/retries:
  Ling pago **US$ 0,32137206**; Gemma26 **US$ 0,310976592**. Nao sao gastos
  executados. Qwen37 ainda nao tem estimativa com tokenizer validado.
- `validation.json` verifica C1 inteiro, prefixo C1t, igualdade C1t/C2 em
  tokens de contexto/prompt, contexto comum C2-C5, mesmos IDs de exemplos
  entre modelos, celulas completas, manifesto e hashes preservados. Os
  prompts Ling free/pago sao identicos, mas isso nao prova equivalencia dos
  provedores. Tokenizers salvos junto aos artefatos; contrato API dos pagos
  ainda nao testado nesta etapa.
- Suite local: 219 testes passaram, excluindo slow e `test_real_data.py`;
  Ruff e diff check aprovados. Nenhum commit ou publicacao.

Na regressao anterior, a inspecao exploratoria do assistente encontrou erros. ABEO/C4 afirma
que o caixa aumentou, mas informa 130,4 no inicio e 13,6 no fim. ACAD/C4 afirma
que gerou caixa operacional e na mesma frase informa caixa usado. ACET/C4
informa saldos 14,9 -> 88,9 e variacao 72,9, que nao fecha a diferenca de 74,0.
Isso nao e anotacao humana nem revisao exaustiva. As saidas completas estao em
`outputs/screen-safe-tables-v2-20260926/resumos-ling.md`; `human-review.json`
permanece sem julgamentos. Nao interpretar correcao do corte como aprovacao
factual. O contexto ABIO ainda contem fragmento CSS.

No dev50, a inspecao das seis saidas de AGRX encontrou inferencias de escalas,
valores comparativos e sinais que precisam de revisao. Detalhes e limites em
`outputs/dev50-ling-20260926/inspecao-exploratoria.md`. Os 300 resumos estao em
`outputs/dev50-ling-20260926/resumos-ling.md`; a ficha humana em
`outputs/dev50-ling-20260926/ling/human-review.json` tem 300 casos sem julgamentos.
BERTScore e testes de hipoteses nao foram executados nesta rodada. Nenhuma
hipotese ou qualidade factual foi aprovada. Revisar evidencia e tabelas antes
do experimento final; qualquer correcao exige novo dev com entradas congeladas.

## Etapas e artefatos

1. Regressao pequena do Ling: cinco primeiros documentos dev, seis bracos,
   32 exemplos. Artefatos em `outputs/screen-safe-tables-v2-20260926/`.
2. Preparacao do dev50: primeiros 50 documentos do manifesto preservado e
   todos os 1.000 exemplos; mesmos documentos e selecoes de exemplos para os
   candidatos. `scripts/prepare_dev_openrouter.py` prepara fontes, candidatos
   RAG, exemplos, prompts por tokenizer, contagens e hashes. Nao envia geracoes,
   nao carrega pesos de LLM e nao carrega documentos do conjunto eval.
3. Dev50 Ling gratuito concluido como rodada exploratoria. Os outros modelos
   aguardam validacao de contrato/tokenizer e execucao. Cinquenta documentos
   sao 300 geracoes por modelo, nao 50.
4. Congelar escolhas no dev, executar revisao humana e so entao autorizar a
   avaliacao final dos 1.000 documentos reservados: 6.000 geracoes por modelo.

Preparacao executavel, sem credencial de geracao:

```bash
uv run --no-sync python scripts/prepare_dev_openrouter.py \
  --output outputs/dev50-prepared-NOVA-RODADA
```

Configuracao usada no dev gratuito: `configs/dev_openrouter_ling.yaml`.
Comando para uma nova rodada apos tratar as pendencias:

```bash
uv run --no-sync --env-file ../.env python scripts/screen_openrouter.py \
  --config configs/dev_openrouter_ling.yaml --models ling --n-docs 50 \
  --output outputs/dev50-ling-NOVA-RODADA
```

Para retomar, usar o mesmo output e acrescentar `--resume`. Cada caso concluido
e salvo; config e prompts precisam ser identicos. Mudancas de metodologia ou
banco de exemplos exigem outra rodada. Nao combinar resultados dos pools 32 e
1.000 como se fossem a mesma configuracao.

## Orcamento e provedores

Catalogo de preparacao: `configs/openrouter_candidates.json`. Snapshots de
endpoints consultados no MCP OpenRouter: `outputs/preparacao-dev-20260926/`.

| Modelo | Provedor fixado | Entrada / saida por milhao | Janela do endpoint |
|---|---|---|---:|
| Ling gratuito | Novita | US$ 0 / 0 | 262.144 |
| Ling pago | DeepInfra FP4 | US$ 0,06 / 0,18 | 262.144 |
| Qwen3.7 Flash | Alibaba | US$ 0,03 / 0,13 ate a primeira faixa | 1.000.000 |
| Gemma 4 26B A4B | Darkbloom | US$ 0,042 / 0,22 | 131.072 |

Qwen tem faixas de preco: a partir de 32.000 tokens de prompt, 0,10/0,40;
a partir de 256.000, 0,20/0,80. Nao projetar todo o experimento pela tarifa
menor sem medir C1 e os prompts few-shot. Precos precisam ser revalidados antes
da execucao. Preparacao local nao e autorizacao automatica para gastar.

O tokenizer oficial do Gemma 26B esta identificado. Para Qwen3.7 Flash, nao
foi validado um tokenizer oficial nem metodo de contagem exata remoto. As
fontes e exemplos ficam preparados, mas nao se declara preflight C1t/C2 nem
se substitui silenciosamente pelo tokenizer de outro Qwen.

## Cota e contingencia

- O limite e de chamadas, nao documentos: cada documento completo consome
  seis chamadas, alem das tentativas. Janela de 20 chamadas/minuto, com retries
  limitados 1, 2, 4, 8, 16 segundos e respeito a Retry-After.
- A triagem verifica a cota atual e contabiliza cada tentativa. Nao exige
  antecipadamente 1.800 chamadas disponiveis para um lote de 300. Ao atingir
  o saldo conservador, interrompe e preserva os casos concluídos.
- Um 429 de pool compartilhado nao prova esgotamento diario. Revalidar `/key`
  antes de atribuir essa causa; o cliente nao troca de provedor automaticamente.
- Ling gratuito usa Novita; Ling pago usa DeepInfra FP4. Registrar modelo,
  provedor e precisao, validar contrato de tokens/geracao e estabelecer teto
  de gasto antes de rodar a contingencia. Nao alternar dentro do bloco de seis
  bracos de um documento. Se houver bloco parcial, termina-lo no gratuito em
  outro dia ou refazer os seis bracos no pago, em rodada separada.
- Nao juntar resultados entre provedores sem apresentar a composicao e
  verificar sensibilidade por provedor. A contingencia paga tem executor
  separado, descrito abaixo; o comando gratuito permanece restrito a custo zero.

## Executor dos prompts pagos preparados

`scripts/run_prepared_paid.py` aceita Gemma26/Darkbloom e Ling/DeepInfra FP4.
Sem `--execute`, confere hashes, manifesto dev50, tokenizers locais salvos,
limites e controles, e calcula a reserva conservadora. Nao exige chave nessa
modalidade e nao chama geracao. Qwen37 permanece excluido enquanto nao houver
preflight nativo validado: o cadastro OpenRouter informa `hugging_face_id: null`.
A documentacao publica de contagem localizada para Alibaba OpenSearch lista
qwen-turbo/plus/max, nao valida suporte a Qwen3.7 Flash. A contagem nativa de
OpenRouter e retornada apos a chamada, nao resolve sozinha o preflight local.
Fontes: [modelo](https://openrouter.ai/qwen/qwen3.7-flash),
[contagem OpenSearch](https://www.alibabacloud.com/help/en/open-search/search-platform/developer-reference/token-calculation),
[uso de tokens OpenRouter](https://openrouter.ai/docs/api_reference/overview).

```bash
uv run --no-sync python scripts/run_prepared_paid.py \
  --prepared outputs/dev50-prepared-v2-20260926 --model gemma26 --n-docs 50
```

Dry-run real concluido para os 300 prompts Gemma26 e os 300 prompts Ling pago.
Reservas: US$ 0,460800 e US$ 0,534528, respectivamente, sem retries. Sao maiores
que o custo maximo calculado acima porque cada tentativa reserva todo o teto de
entrada 20.480, dando margem para overhead inesperado. A contagem API deve ainda
coincidir exatamente com a local; uma diferenca de um token bloqueia a rodada.

Execucao exige explicitamente `--execute --output PASTA_NOVA --budget-usd VALOR`.
O ledger e gravado antes do envio; inclusive erros e timeouts conservam reserva.
So 429 e repetido, com no maximo seis tentativas. Respostas brutas sao gravadas
antes da validacao; custos observados sao registrados mesmo se o contrato falhar.
Uma resposta aceita e reutilizada na retomada, sem nova chamada; resultado
desconhecido ou invalido exige auditoria. Lock exclusivo impede duas execucoes
gastando simultaneamente na mesma pasta. Nao ha fallback de modelo/provedor.

Para Ling pago, o executor exige cota gratuita `remaining == 0`. O argumento
`--pending-from outputs/dev50-ling-20260926 --n-docs 50` restringe a contingencia
a documentos sem seis resultados completos na rodada gratuita, que precisa
estar interrompida. Um documento parcial e refeito inteiro no pago; resultados
pagos ficam em pasta separada, sem sobrescrever nem mesclar previsoes gratuitas.
Os prompts devem coincidir com os do gratuito. O executor nao alterna para pago
em resposta a qualquer 429 e nao autoriza gasto por si so.

Doze testes offline verificam orcamento, falha de rede, retomada, seis 429,
contrato, bloqueio concorrente e blocos de documentos pendentes. Nesta retomada
nenhuma chamada paga foi executada.

## Correcao e limitacoes

O corte final do contexto agora recua ate a celula completa e remove cabecalho
tabular orfao, inclusive quando o corte ocorre antes do primeiro separador `|`.
C1t/C2 continuam exigindo igualdade exata de contexto e prompt zero-shot;
o teto efetivo pode diminuir. C2-C5 compartilham a evidencia resultante.
Essa regra e exclusiva do orcamento do prompt: janelas de embedding continuam
cobrindo todo o texto, inclusive linhas longas com separadores.

Preservar celulas nao resolve unidades ausentes, strings ambiguas com `&`,
sinais, cabecalhos incompletos, recuperacao predominantemente tabular ou
demonstracoes cujo texto abreviado nao sustenta todo o resumo. Essas questoes
ainda exigem inspecao no dev. Nenhum modelo, hipotese ou qualidade factual
esta aprovado pelo sucesso de uma chamada ou por testes de codigo.
