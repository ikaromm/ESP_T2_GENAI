# Formato do FINDSum

Notas sobre a estrutura real dos arquivos distribuidos, verificadas
empiricamente contra o download. Os testes em `tests/test_real_data.py`
travam cada uma destas afirmacoes.

Fonte: [StevenLau6/FINDSum](https://github.com/StevenLau6/FINDSum) (Liu et al.,
2022), distribuido via Google Drive sob licenca
[ODC-BY](https://opendatacommons.org/licenses/by/1-0/).

## Arvore de arquivos

21 arquivos, 6.4 GB no total, em duas ramificacoes:

```
text/FINDSum-Liquidity/liquidity_input_2000/{train,val,test}_liquidity_segment_{0,1,2}_input_2_1000.csv
text/FINDSum-ROO/roo_input_2000/{train,val,test}_roo_segment_{0,1}_input_2_1000.csv
table/FINDSum-Liquidity/{train,val,test}_liquidity_all_tuples_diff_sec.txt
table/FINDSum-ROO/{train,val,test}_roo_all_tuples_diff_sec.txt
```

Duas tarefas: **ROO** (results of operations, 2 segmentos) e **Liquidity**
(liquidity and capital resources, 3 segmentos).

## Os arquivos de segmento nao sao shards

Esta e a parte contraintuitiva. Os `segment_0`, `segment_1`, `segment_2` de um
mesmo split **nao** sao particoes do conjunto de documentos: todos tem o mesmo
numero de linhas e a linha `r` de cada arquivo pertence ao **mesmo relatorio**.

Cada arquivo traz, para aquele relatorio, uma porcao (~2000 palavras) do
conteudo selecionado e a **porcao correspondente do resumo de referencia**.
Concatenar os segmentos na ordem reconstroi entrada e resumo completos:

```python
document = " ".join(seg.document for seg in segments)   # ~6000 palavras
summary  = " ".join(seg.summary  for seg in segments)   # ~1000 palavras
```

A evidencia e a continuidade dos resumos. Em `val_liquidity`, linha 0 (Genco
Shipping):

| segmento | fim do trecho de resumo | inicio do seguinte |
|---|---|---|
| 0 → 1 | `...decrease in the purchase of vessels , including deposits .` | `the decrease is primarily due to the completion of the purchase of the three ultramax newbuilding vessels...` |
| 1 → 2 | `...$ 2.8 million repayment of debt` | `under the 2014 term loan facilities and $ 1.5 million payment of deferred financing costs .` |

Tratar os arquivos como shards independentes produziria 3x documentos, cada um
com um terco do resumo de referencia -- e resultados sem sentido.

## Os segmentos sao selecoes de conteudo, nao janelas contiguas

Os marcadores `replace_table_token_<i>_th` usam numeracao **global ao
relatorio** e os intervalos **se sobrepoem** entre segmentos. Em `val_liquidity`
linha 0: seg0 referencia as tabelas 9–15, seg1 as 14–15, seg2 as 14–15.

Se os segmentos fossem janelas contiguas, os intervalos seriam disjuntos e
crescentes. Portanto o pipeline original do FINDSum aplica **selecao de
conteudo** antes de montar cada segmento, e o texto distribuido ja e um extrato
do 10-K, nao o documento integral.

### Consequencia para a pesquisa

O que se tem em maos e ~6000 palavras de conteudo **ja selecionado**, nao um
10-K completo (que passa de 50.000 palavras). Duas leituras possiveis:

1. **Tratar o extrato concatenado como "o documento"** (o que este repositorio
   faz). O RAG recupera trechos de dentro dele. Autocontido e fiel ao dataset
   publicado, mas o ganho atribuivel ao RAG e menor, porque uma etapa de
   selecao ja foi aplicada a montante.
2. **Recuperar os 10-Ks originais na EDGAR.** Cada linha do arquivo de tabelas
   traz `stock_name` e `report_id`, e o `report_id` e o accession number da SEC
   (ex.: `0001558370-17-002200`), o que torna o documento integral recuperavel.
   Documentos genuinamente longos, ao custo de uma etapa de coleta a mais.

A escolha deve ser declarada explicitamente na redacao do TCC, porque muda o que
a afirmacao "sumarizacao de relatorios longos" significa.

## Marcadores no texto

| marcador | significado |
|---|---|
| `story_separator_special_tag` | separa os trechos selecionados dentro de um segmento |
| `replace_table_token_<i>_th` | posicao onde a i-esima tabela do relatorio aparecia |

`clean_text()` remove ambos; `Document.passages` usa o primeiro como fronteira
natural de trecho, preservando as unidades de conteudo da selecao original em
vez de cortar o texto em pontos arbitrarios.

## Arquivos de tabela

Um objeto JSON por linha, **alinhado por indice de linha** com os CSVs de texto.
Chaves:

| chave | conteudo |
|---|---|
| `stock_name` | ticker (ex.: `GNK`) |
| `report_id` | accession number da SEC |
| `mda_liquidity_tables` | tabelas da secao de liquidez do MD&A |
| `mda_before_liquidity_tables`, `mda_after_liquidity_tables` | tabelas vizinhas no MD&A |
| `before_mda_tables`, `after_mda_tables` | tabelas fora do MD&A |

Cada tabela e uma lista de celulas
`[rowname, colname, cell_value, date, cell_row_index, cell_col_index]`. O campo
`colname` vem vazio na maioria dos casos; `cell_value` as vezes carrega duas
formas do mesmo numero separadas por `&` (ex.: `'1.7 & 1,653,732'`, isto e, 1.7
bilhoes escrito tambem em milhares).

O alinhamento foi confirmado de duas formas: `stock_name=GNK` na linha 0 de
`val_liquidity` corresponde a um texto sobre navios e fretes (Genco Shipping), e
os `rowname` das tabelas citadas reaparecem no texto do mesmo relatorio.

A numeracao usada por `Table.index` neste projeto e uma **convencao local**
(chave da tarefa primeiro, depois as de contexto): o dataset nao documenta o
mapeamento exato entre `replace_table_token_<i>_th` e essas listas. E
deterministica entre execucoes, mas nao necessariamente identica a do artigo.

## Estatisticas medidas

Tarefa Liquidity, split `val`, 200 primeiros documentos:

| | min | media | max |
|---|---|---|---|
| entrada (palavras) | 5.751 | 5.931 | 6.003 |
| resumo de referencia (palavras) | 384 | 999 | 1.181 |

Resumos de ~1000 palavras equivalem a ~1.300 tokens: por isso
`GenerationConfig.max_new_tokens` e 1536 e nao um valor tipico de 512, que
truncaria a geracao e penalizaria a cobertura de todas as configuracoes por
igual.

Contagem de linhas por split (Liquidity): `val` tem 2.067 documentos. Os
arquivos de `train` sao ~8x maiores, coerente com os 21.125 relatorios de 3.794
empresas descritos no artigo.

## O resumo de referencia nao esta contido no texto distribuido

Consequencia direta da selecao de conteudo, e a medida mais importante para
interpretar os resultados. Comparando o **proprio resumo de referencia** com o
extrato de ~6000 palavras (Liquidity/val, 50 documentos):

| metrica da referencia contra o documento | media | minimo |
|---|---|---|
| `numeric_grounding` (numeros do resumo presentes no texto) | ~0,56 | 0,38 |
| `ngram_grounding` (4-gramas do resumo presentes no texto) | ~0,12 | 0,004 |

Ou seja: **cerca de 44% dos numeros do resumo humano nao aparecem no texto
distribuido**. O material de onde esses numeros vieram foi descartado pela
selecao de conteudo (provavelmente estava nas tabelas ou em secoes do 10-K fora
do extrato).

Duas implicacoes para a analise:

1. `numeric_grounding` tem **teto pratico de ~0,56**, nao 1,0. Avaliar uma
   configuracao contra 1,0 levaria a concluir que todas alucinam metade dos
   numeros, quando na verdade estao no mesmo regime da referencia.
2. `ngram_grounding` mede **extratividade, nao fidelidade**. Os resumos do
   FINDSum sao abstrativos (0,12 de sobreposicao de 4-gramas). Um modelo com 0,9
   estaria colando trechos do documento -- comportamento pior, nao melhor. A
   leitura util e a distancia ao patamar da referencia.

`metrics.reference_baseline()` calcula esses valores e toda rodada os grava em
`reference_baseline.json`. Se a opcao 2 da secao anterior (coletar os 10-Ks
integrais na EDGAR) for adotada, espera-se que o teto numerico suba
substancialmente -- e isso seria, por si, um argumento a favor dessa escolha.

## Como baixar

A listagem de pastas publicas do Google Drive nao esta exposta na API sem
credencial, e `gdown` falha silenciosamente nesta pasta (cria a arvore de
diretorios e baixa zero arquivos). `scripts/fetch_findsum.py` extrai o payload
`window['_DRIVE_ivd']` embutido no HTML da pagina da pasta, percorre a arvore,
grava `manifest.json` e baixa com retomada:

```bash
python scripts/fetch_findsum.py --manifest-only   # so lista
python scripts/fetch_findsum.py                   # baixa (pula o que existe)
```

A verificacao de integridade compara o tamanho do arquivo local com o do
manifesto; arquivos divergentes sao rebaixados.
