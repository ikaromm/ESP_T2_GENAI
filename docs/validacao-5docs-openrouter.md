# Validação exploratória — cinco documentos, três modelos — 2026-09-26

**Precisa de revisão.** O Ling gerou os 30 resumos, mas há erros materiais de
escala, direção, período e classificação. Qwen e Gemma não produziram saídas:
continuaram bloqueados por 429 nos provedores. Não há comparação de qualidade
entre os três modelos nem confirmação de hipóteses.

## Escopo e execução

Mesmos cinco primeiros documentos do manifesto `dev`, sem seleção por resultado:
ABEO-0001493152-21-006705, ABIO-0001564590-21-014148, ABT-0001047469-19-000624,
ACAD-0001564590-21-008372 e ACET-0001564590-21-012664. Seis braços por documento;
pool reduzido de 32 exemplos, quatro demonstrações em C3–C5. É triagem, não o
piloto de 50 nem a avaliação final. Os três primeiros já foram usados para
ajustes; ACAD e ACET ampliam a inspeção. A amostra não é representativa aleatória.

Instrução final de até 750 palavras, teto de saída 3.072 tokens, entrada máxima
16.384, contexto RAG até 3.072 tokens, temperatura zero, reasoning desativado,
provedores fixados, preço máximo zero e sem fallback. O código metodológico não
foi alterado durante a rodada. O script ganhou apenas `--n-docs` (padrão 3).

| Modelo gratuito | Provedor | Saídas aceitas / previstas | Tentativas reais |
|---|---|---:|---|
| Ling 3.0 Flash Fin | Novita | 30/30 | 30 gerações, cinco documentos |
| Qwen3.8 27B | ModelRun | 0/30 | 6 tentativas em ABEO/C1 e 1 em C1 de cada outro documento |
| Gemma 4 31B IT | Google AI Studio | 0/30 | 6 tentativas em ABEO/C1 e 1 em C1 de cada outro documento |

Qwen e Gemma: todas as dez tentativas por modelo receberam HTTP 429 com
`limit_source=upstream_provider_shared_pool`, sem headers de limite. Os demais
braços ficaram pendentes após o bloqueio. São saídas **indisponíveis**, não escores
zero ou demonstração de inferioridade desses modelos.

Foram preparados e validados os 90 prompts: fonte C1 integral dentro da política
por seção, igualdade de tokens de contexto e prompt C1t/C2, contexto idêntico em
C2–C5 e mesmos IDs de exemplos entre modelos. Nenhum overflow. As 30 saídas do
Ling terminaram em `stop`, com 156–702 palavras e custo reportado total US$ 0.
O pico observado pelos horários dos pedidos foi 11 chamadas em 60 segundos,
abaixo do limitador de 20. Retentativas de 1,2,4,8,16 segundos foram limitadas.

Consulta às 14:35:28 UTC: 121 chamadas usadas, 879 restantes, teto diário 1.000.
Não inferir cobrança ou isenção de todas as falhas a partir desse contador.

## Leitura das saídas

Foram lidos os 30 resumos do Ling. A conferência das afirmações foi seletiva,
priorizando erros materiais; não é revisão exaustiva de todas as afirmações e
não substitui os dois revisores humanos. Os rótulos humanos continuam vazios.
Trechos exatos e seu tipo de evidência estão em `assistant-review.json`.

| Documento | Exemplo observado | Conclusão sustentada |
|---|---|---|
| ABEO | C1 e C4 dizem que o caixa aumentou de 130,4 para 13,6 milhões | Contradição aritmética: queda de 116,8 milhões |
| ABIO | C2 e C4 dizem investimento de 19 milhões em 2020 | A referência FINDSum informa US$ 19.000; escala 1.000 vezes maior |
| ABT | C3 atribui `6.3 & 6,300` a 2017 | A fonte atribui a 2018; o contexto entregue perdeu o período no corte |
| ACAD | C4 começa dizendo que gerou caixa operacional e logo informa caixa usado nos três anos | Contradição dentro do próprio resumo; fonte diz `used` |
| ACET | C4 apresenta saldo de 14,9 → 88,9 milhões e variação de 72,9 milhões | Os saldos implicam 74,0 milhões; nem o arredondamento explica a diferença |

Outros achados rastreados: ABIO/C3 chama uma captação de IPO, enquanto a referência
fala em oferta registrada direta e vendas at-the-market; ABIO/C5 substitui a
captação de ações por financiamento de fornecedor no cálculo de fôlego de caixa.
ACET/C2 coloca o caixa adquirido na fusão em financiamento, mas a referência o
inclui em investimento. ACET/C1 inverte o sinal do investimento de 2019 frente
à referência: entrada em vez de saída. Fonte entregue, referência FINDSum e
consistência interna são evidências distintas; não foram consultados os 10-K
originais para homologar todos os valores e unidades.

Há comportamentos úteis: ABIO/C1t preserva o horizonte até 2022 presente na prosa
e reconhece ausência de saldos; ABT/C1t explicita a falta de fluxo detalhado.
ACET/C1 e C1t recuperam termos do empréstimo presentes na prosa. Esses acertos
localizados não aprovam os resumos completos. Em ACAD e ACET, vários braços RAG
reconhecem falta de narrativa, mas ainda inferem escalas ou relações não sustentadas.

## Problemas na evidência entregue

1. **Corte residual de células.** Dividir tabelas por linhas não impediu que
   `matched_contexts` cortasse a última célula ao igualar os tokens. ABT/C2–C5
   termina em `net cash from operating activities | 6.3 &`; ABEO termina em
   `cash and cash equivalents | 12.6 & 12,5`. Isso remove valores/períodos e
   favorece complementação indevida. A correção anterior foi insuficiente nesse
   ponto; precisa haver corte seguro também na etapa final de orçamento.
2. **Contexto quase inteiramente tabular.** Em ABEO, ACAD e ACET, todos os blocos
   entregues pelo RAG são tabelas. ABIO tem oito blocos tabulares e um fragmento
   de CSS; ABT tem sete tabulares e um trecho contábil pouco ligado aos drivers
   de caixa. A consulta por tarefa melhorou o tema recuperado, mas não garantiu
   narrativa suficiente sobre causas, liquidez ou crédito.
3. **Ambiguidade de colunas, escalas e sinais.** Há várias entradas com mesma
   rubrica/período, cabeçalho vazio e números crus diferentes. Manter `&` e avisar
   o modelo não recompõe a semântica perdida. É necessário auditar a estrutura
   original antes de interpretar tais células como montantes independentes.

Não é correto atribuir tudo à capacidade do Ling: a evidência tem limitações
concretas. Também não é correto desculpar contradições internas como a de ABEO;
essas são detectáveis mesmo sem consultar o relatório original.

## Métricas descritivas do Ling

Média de cinco documentos, sem testes de hipóteses ou BERTScore:

| Braço | ROUGE-L | F1 numérico |
|---|---:|---:|
| C1 | 0,159 | 0,355 |
| C1t | 0,142 | 0,233 |
| C2 | 0,161 | 0,299 |
| C3 | 0,143 | 0,260 |
| C4 | 0,152 | 0,279 |
| C5 | 0,139 | 0,240 |

As médias foram recalculadas a partir dos CSVs e reconciliadas com o resumo da
execução. Coincidência de números e similaridade textual não medem factualidade.
Não escolher um vencedor, inferir equivalência ou confirmar H1–H4 com esta rodada.

## Artefatos e próximos passos

Diretório: `outputs/screen-5docs-20260926/` (ignorado pelo git).

- `resumos-ling.md`: as 30 saídas completas, organizadas por documento/braço.
- `assistant-review.json`: nove achados com trechos, notas para os 30 casos e finais de contexto.
- `validation.json` e `validate_artifacts.py`: verificações independentes reproduzíveis.
- `ling/human-review.json` e `.map.json`: 30 casos cegos, aguardando revisão humana.
- `*/preflight.json`, `*/api/`, `*/config.yaml`: prompts, respostas/erros e condições.
- `availability-probes/`: uma tentativa C1 nos outros quatro documentos por modelo bloqueado.
- `implementation-sha256.json`, `quota-after.json`: versão efetiva e cota observada.

```bash
uv run --no-sync --env-file ../.env python scripts/screen_openrouter.py \
  --output outputs/screen-5docs-NOVO --n-docs 5 --models ling qwen gemma
```

Para retomar esta mesma configuração, usar `--resume --n-docs 5` e o diretório
existente; o script confere preflight/configuração e não repete casos aceitos.
Mudar preparação ou prompt exige rodada nova.

Antes de ampliar para 50: corrigir o corte final de células, auditar colunas e
unidades, recuperar narrativa de caixa junto às tabelas e repetir a regressão
sem misturar rodadas. Completar a comparação de modelos depende da disponibilidade
de Qwen e Gemma. Nenhuma avaliação final, chamada paga ou publicação foi executada.
24 testes do cliente/retomada passaram, Ruff/diff check passaram; 26 arquivos
anteriores e o manifesto foram conferidos por hash e preservados.
