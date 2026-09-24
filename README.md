# ESP_T2_GENAI

Avaliação de RAG e few-shot prompting na sumarização de relatórios financeiros
longos, usando o dataset FINDSum e uma LLM de pesos abertos.

Projeto de pesquisa do TCC (AKCIT / Embrapii) — Diego Corrêa Bento e Ikaro
Moribayashi Mendonça.

## Pergunta de pesquisa

Em que medida a combinação de RAG com seleção dinâmica de exemplos few-shot
semanticamente similares melhora a qualidade textual, a fidelidade factual e a
preservação de informações numéricas de resumos de relatórios financeiros
longos, comparada a abordagens sem RAG e a estratégias convencionais de seleção
de exemplos?

## As cinco configurações

| id | RAG | exemplos few-shot |
|----|-----|-------------------|
| C1 | não | nenhum (baseline) |
| C2 | sim | nenhum |
| C3 | sim | fixos |
| C4 | sim | aleatórios |
| C5 | sim | **dinâmicos por similaridade semântica** ← a proposta |

O que muda entre elas é apenas o contexto e os exemplos. Modelo, instrução,
estrutura do prompt, parâmetros de geração e conjunto avaliado são idênticos —
sem isso, uma diferença de desempenho poderia vir da redação do prompt em vez da
técnica investigada.

## Instalação

Requer [uv](https://docs.astral.sh/uv/). O Python 3.12 é fixado no
`.python-version` porque o torch ainda não publica wheels para o 3.14.

```bash
uv sync --group dev
```

## Dados

O FINDSum (Liu et al., 2022, licença ODC-BY) é distribuído via Google Drive.

```bash
uv run python scripts/fetch_findsum.py --manifest-only   # lista os 21 arquivos (6.4 GB)
uv run python scripts/fetch_findsum.py                   # baixa, com retomada
```

O download é idempotente: arquivos já completos são pulados por comparação de
tamanho com o manifesto. Os dados ficam em `data/raw/findsum/`, fora do git.

**Leia [`docs/dataset.md`](docs/dataset.md) antes de mexer no pipeline de
dados.** O formato tem duas armadilhas que invalidariam os resultados em
silêncio:

1. Os arquivos `segment_0/1/2` **não** são shards do conjunto — são porções do
   *mesmo* documento, com as porções correspondentes do resumo. É preciso
   concatená-los.
2. O texto distribuído já passou por seleção de conteúdo; são ~6.000 palavras de
   extrato, não o 10-K integral. Isso limita o que a expressão "documentos
   longos" pode afirmar no TCC.

## Uso

```bash
uv run findsum arms                       # lista as configurações
uv run findsum inspect --split val        # valida a carga dos dados
uv run findsum init-config                # gera configs/experiment.yaml
uv run findsum run --config configs/experiment.yaml
uv run findsum run --config configs/experiment.yaml --arm C1 --arm C5
uv run findsum compare --run-dir outputs/baseline-run --reference C5
```

Artefatos por rodada, em `outputs/<name>/`:

```
config.yaml            configuração efetiva, para reprodução
summary.json           métricas agregadas de todas as configurações
<arm>/predictions.jsonl  resumo gerado, referência, exemplos usados, tokens
<arm>/scores.csv         métricas por documento
<arm>/summary.json       médias e medianas da configuração
reference_baseline.json  calibração das métricas de ancoragem (ver abaixo)
```

## Comparação estatística

Como todas as configurações rodam sobre os mesmos documentos, as amostras são
emparelhadas. `findsum compare` aplica Wilcoxon para postos sinalizados (não
supõe normalidade, e ROUGE/BERTScore são limitados em [0,1] e assimétricos),
compara tudo contra uma configuração de referência e corrige os p-valores por
Holm-Bonferroni — com 5 configurações e 8 métricas, testar todos os pares sem
correção infla o erro tipo I.

O tamanho de efeito reportado é a proporção de documentos em que a referência
supera a outra configuração, descontados os empates.

## Métricas

Três dimensões, conforme o projeto de pesquisa:

| dimensão | métricas |
|---|---|
| qualidade textual | ROUGE-1/2/L, BERTScore |
| preservação numérica | precisão / cobertura / F1 dos números contra a referência |
| fidelidade factual | ancoragem numérica contra o **documento** |
| (controle) extratividade | sobreposição de n-gramas com o documento |

A separação entre preservação e fidelidade é deliberada. Comparar com a
referência mede se o resumo **cobre** o que deveria; comparar com o documento
mede se o resumo **inventa**. Um número ausente da fonte é alucinação mesmo que
apareça na referência, e um resumo pode ir bem numa dimensão e mal na outra.

A extração numérica normaliza escalas textuais, então `$ 50.0 million` e
`50,000,000` contam como o mesmo valor. `numeric_grounding` vale 1.0 para
resumos sem números — interprete sempre junto de `numeric_recall`, que é quem
penaliza o resumo que simplesmente evita citar valores.

### As métricas de ancoragem precisam de calibração

Medido nos dados reais (Liquidity/val), o **próprio resumo de referência** marca:

| métrica | valor da referência |
|---|---|
| `numeric_grounding` | ~0,56 |
| `ngram_grounding` | ~0,12 |

Duas consequências, ambas contraintuitivas:

1. **~44% dos números do resumo humano não estão no texto distribuído.** É efeito
   da seleção de conteúdo do FINDSum (ver `docs/dataset.md`). Isso impõe um teto
   prático: nenhum modelo que use apenas esse texto deveria ser cobrado de
   superar ~0,56. Comparar contra 1,0 produziria a conclusão falsa de que todas
   as configurações alucinam massivamente.
2. **`ngram_grounding` mede extratividade, não fidelidade.** Os resumos do
   FINDSum são abstrativos. Um modelo com 0,9 aqui estaria colando trechos — pior,
   não melhor. O que interessa é a proximidade ao patamar da referência.

Toda rodada grava `reference_baseline.json` com esses valores, para que os
resultados sejam lidos na escala certa.

## Estrutura

```
scripts/fetch_findsum.py   download do Google Drive com retomada
src/findsum_rag/
  data.py        carga e remontagem dos segmentos, tabelas
  chunking.py    fatiamento em trechos recuperáveis
  retrieval.py   embeddings + índice FAISS (cosseno exato)
  examples.py    as quatro estratégias de seleção de exemplos
  prompts.py     montagem do prompt, idêntica entre configurações
  generate.py    execução da LLM, quantização 4-bit
  metrics.py     ROUGE, BERTScore, métricas numéricas e de fidelidade
  analyze.py     Wilcoxon emparelhado + Holm-Bonferroni entre configurações
  pipeline.py    orquestração das rodadas
  config.py      configuração validada (pydantic) + as 5 configurações
  cli.py         interface de linha de comando
docs/dataset.md  formato do FINDSum, verificado empiricamente
```

Duas bases vetoriais distintas, que não devem ser confundidas: a de **contexto**
(trechos do próprio documento a resumir, usada pelo RAG) e a de **exemplos**
(pares documento-resumo de outros documentos, sempre do split de treino, usada
pelo few-shot). Nenhum documento é exemplo de si mesmo.

## Testes

```bash
uv run pytest                  # tudo
uv run pytest -m "not slow"    # sem carregar LLM
uv run ruff check .
```

Os testes unitários usam um FINDSum sintético em `tmp_path` e não precisam do
download. `tests/test_real_data.py` roda contra os dados reais e é pulado se
eles não estiverem presentes — é lá que ficam travadas as suposições sobre o
formato do dataset. Os testes marcados `slow` carregam uma LLM minúscula para
exercitar a API do transformers de verdade.

## Hardware de referência

RTX 4070 (12 GB), onde um modelo de 7B em bf16 não cabe. O padrão é quantização
NF4 em 4 bits (~5 GB), deixando folga para o cache de atenção dos prompts de
~12k tokens deste estudo.

## Pendências

- [ ] Escolher e justificar a LLM de pesos abertos (o default é
      `Qwen/Qwen2.5-7B-Instruct`, ainda não comparado com alternativas)
- [ ] Rodada completa das cinco configurações e testes estatísticos entre elas
- [ ] Análise de erros por documento e por tipo de informação
- [ ] Teste de generalização em documentos fora do desenvolvimento
- [ ] Decidir entre o extrato do FINDSum e a coleta dos 10-Ks na EDGAR
      (ver `docs/dataset.md`)

## Citação do dataset

```bibtex
@inproceedings{liu-etal-2022-long,
    title = "Long Text and Multi-Table Summarization: Dataset and Method",
    author = "Liu, Shuaiqi and Cao, Jiannong and Yang, Ruosong and Wen, Zhiyuan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    pages = "1995--2010",
    url = "https://aclanthology.org/2022.findings-emnlp.145",
}
```
