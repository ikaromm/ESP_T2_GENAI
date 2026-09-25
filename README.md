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

## As configurações

| id | contexto do documento | exemplos few-shot |
|----|----|----|
| C1 | **documento inteiro** | nenhum |
| C1t | truncado no orçamento de C2 | nenhum |
| C2 | recuperado (`top_k` trechos) | nenhum |
| C3 | recuperado | 4 fixos |
| C4 | recuperado | 4 aleatórios |
| C5 | recuperado | **4 dinâmicos por similaridade semântica** ← a proposta |

O que muda entre elas é apenas o contexto e os exemplos. Modelo, instrução,
estrutura do prompt, parâmetros de geração e conjunto avaliado são idênticos —
sem isso, uma diferença de desempenho poderia vir da redação do prompt em vez da
técnica investigada.

C1 recebe o **documento inteiro** porque ele cabe: são ~7,3 mil tokens contra uma
janela de 262 mil. Truncar o baseline seria construir uma escassez de contexto
que não existe. Isso muda o que a hipótese do RAG afirma — deixa de ser "a
recuperação dá mais informação ao modelo" e passa a ser **"um recorte curado
supera o documento inteiro"**, testável pela degradação conhecida de atenção em
contexto longo.

C1t existe para separar os dois efeitos. Com o mesmo orçamento de C2 mas cortando
em vez de recuperar, ele isola quanto do resultado vem de *recuperar* e quanto
vem apenas de *reduzir* o contexto.

### A cadeia de ablação

Cada par adjacente muda exatamente um fator:

| | comparação | fator isolado |
|---|---|---|
| H1 | C1 → C2 | recorte curado vs documento inteiro |
| H1b | C1t → C2 | recuperar vs truncar, mesmo orçamento |
| H2 | C2 → C3 | presença de exemplos |
| H3 | C3 → C4 | identidade dos exemplos (controle) |
| H4 | C4 → C5 | **relevância semântica dos exemplos** |

H3 parece inútil e é o controle mais importante: se C3 ≈ C4, o que importa é ter
exemplos, não quais — e aí o ganho de C5 isola limpo o efeito da seleção
semântica. H4 é a hipótese central da pesquisa.

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

## Os conjuntos experimentais

Aqui **nada é treinado** — nenhum peso é atualizado. Os splits `train`/`val`/`test`
do FINDSum foram feitos para treinar modelos seq2seq e não servem a este estudo.
Os três splits são fundidos num pool único (20.672 documentos, 3.742 empresas) e
reparticionados em:

| conjunto | tamanho | para que serve | quantas vezes se olha |
|---|---|---|---|
| `examples` | 1.000 | demonstrações few-shot de C3–C5 | nunca é pontuado |
| `dev` | 500 | ajustar prompt, `top_k`, nº de exemplos | quantas vezes quiser |
| `eval` | 1.000 | o número que vai no trabalho | **uma vez, no fim** |

```bash
uv run python scripts/build_splits.py     # grava data/interim/splits-liquidity.json
uv run findsum splits                     # inspeciona a partição
```

Duas invariantes, ambas exigidas pelo desenho e validadas na carga do manifesto:

1. **Uma empresa em um único conjunto, com um único relatório** (o mais recente).
   Sem isso um exemplo few-shot poderia ser outro exercício da mesma empresa
   avaliada, e o ganho de C5 viria de quase-duplicação em vez de relevância
   semântica.
2. **A partição é congelada em disco.** Ampliar a avaliação depois de olhar
   resultados só não é cherry-picking se o conjunto já estava definido antes. O
   script se recusa a sobrescrever um manifesto existente sem `--force`.

O preço é perder comparabilidade com resultados publicados sobre o split oficial
do FINDSum. Como o estudo compara as configurações entre si, é um preço baixo —
mas precisa constar na redação.

## Uso

```bash
uv run findsum arms                       # lista as configurações
uv run findsum splits                     # inspeciona a partição
uv run findsum inspect --split val        # valida a carga dos dados brutos
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

## Executar num servidor remoto (Docker)

`deploy.sh` empacota tudo em container. A imagem não contém dados nem pesos de
modelo: o FINDSum (6,5 GB) e o modelo (~6 GB) são volumes, para que a imagem
continue reproduzível e alterar um `.py` não invalide a camada de 3 GB do torch.

```bash
git clone git@github.com:ikaromm/ESP_T2_GENAI.git && cd ESP_T2_GENAI
./deploy.sh doctor        # confere docker, GPU, dados, disco
./deploy.sh data          # baixa o FINDSum e gera a partição (uma vez)
./deploy.sh test_50       # roda o perfil configs/test_50.yaml (~6 h)
```

Perfis disponíveis são os arquivos em `configs/`. `./deploy.sh <perfil>` equivale
a `./deploy.sh run <perfil>`.

| perfil | conjunto | documentos | custo estimado |
|---|---|---|---|
| `test_50` | `dev` | 50 | ~6 h |
| `eval_1000` | `eval` | 1.000 | ~71 h |

### Onde fica a saída

Três destinos, e a diferença entre eles é deliberada:

| destino | conteúdo | tamanho | no git? |
|---|---|---|---|
| `results/<perfil>/` | métricas agregadas, `scores.csv` por documento, tabela estatística | ~50 KB | **sim** |
| `outputs/<perfil>/` | tudo, incluindo `predictions.jsonl` com cada resumo gerado | ~5 MB (50 docs) | não |
| `dist/<perfil>.tar.gz` | `outputs/` comprimido, para baixar | | não |

Para publicar os resultados:

```bash
git add results/test_50 && git commit -m "resultados: test_50" && git push
```

Para trazer tudo para a máquina local:

```bash
scp <servidor>:~/ESP_T2_GENAI/dist/test_50.tar.gz . && tar -xzf test_50.tar.gz
```

`predictions.jsonl` fica fora do git porque cresce com o número de documentos
(~100 MB em 1.000), mas é o arquivo que importa para análise de erros — cada
linha traz o resumo gerado, a referência, os exemplos usados no prompt e as
contagens de token. É por isso que existe o tarball.

### Pré-requisitos no servidor

GPU com ~12 GB, driver NVIDIA e **nvidia-container-toolkit**. `./deploy.sh doctor`
verifica e imprime as instruções de instalação se faltar. A geração exige GPU —
em CPU cada documento levaria horas em vez de ~45 s.

Se seu usuário não estiver no grupo `docker`, use `FINDSUM_DOCKER="sudo docker"
./deploy.sh ...` em vez de entrar no grupo (o grupo `docker` equivale a acesso
root na máquina).

Espaço em disco: ~7 GB de dataset, ~7 GB de imagem, ~6 GB de pesos.

## Métricas

A análise principal compara **previsão × resumo de referência**:

| dimensão | métricas |
|---|---|
| qualidade textual | ROUGE-1/2/L, BERTScore |
| preservação numérica | precisão / cobertura / F1 dos números |

A extração numérica normaliza escalas textuais, então `$ 50.0 million` e
`50,000,000` contam como o mesmo valor.

Como diagnóstico secundário, e sem custo de GPU, também são gravadas duas
métricas contra o **documento**: `numeric_grounding` e `ngram_grounding`. Elas
existem porque comparar apenas com a referência pune igualmente um número
inventado e um número que está no relatório mas que o resumidor humano não
selecionou. Se em algum momento for preciso separar os dois casos, o dado já está
em `scores.csv`.

Para interpretá-las é obrigatório olhar `reference_baseline.json`: medido nos
dados reais, o **próprio resumo de referência** marca `numeric_grounding` ~0,56 e
`ngram_grounding` ~0,12 contra o texto distribuído. Ou seja, ~44% dos números do
resumo humano não estão na fonte (efeito da seleção de conteúdo do FINDSum, ver
`docs/dataset.md`), e as referências são abstrativas — um modelo com 0,9 de
`ngram_grounding` estaria colando trechos, o que é pior e não melhor.

## Estrutura

```
deploy.sh                  execução em container, num servidor remoto
Dockerfile                 imagem sem dados nem pesos (ambos são volumes)
configs/test_50.yaml        perfil de iteração (50 documentos do dev)
configs/eval_1000.yaml      perfil de avaliação final (1.000 do eval)
scripts/fetch_findsum.py   download do Google Drive com retomada
scripts/build_splits.py    partição congelada dos conjuntos experimentais
src/findsum_rag/
  data.py        carga e remontagem dos segmentos, tabelas
  splits.py      partição por empresa, manifesto congelado
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

Duas recuperações distintas, que não devem ser confundidas:

1. **contexto** — o índice são os trechos do *próprio documento* a resumir. Não
   pode ser uma base separada: recuperar de outros relatórios faria o modelo
   citar números de outra empresa no resumo.
2. **exemplos** — o índice são os 1.000 documentos do conjunto `examples`, com
   seus resumos. É desta base que saem as demonstrações de C5.

Nenhum documento é exemplo de si mesmo, e nenhuma empresa aparece em dois
conjuntos.

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

## Hardware de referência e custo medido

RTX 4070 (12 GB). O padrão é quantização NF4 em 4 bits. Medições reais com
prompts deste estudo:

| configuração | prompt | taxa | por documento (1.374 tokens de saída) |
|---|---|---|---|
| C1 / C1t / C2 (0 exemplos) | ~3,2 mil tok | 38,6 tok/s | ~36 s |
| C3 / C4 / C5 (4 exemplos) | ~10,3 mil tok | 28,7 tok/s | ~48 s |

Pico de VRAM: 7,9 GB dos 12,3. Uma passada das 6 configurações custa ~4,3 min por
documento, logo **~6 h nos 50 do dev** e **~71 h nos 1.000 da avaliação**.

Por isso o `dev` de 500 não é o loop de iteração: a 30 h por passada, três
iterações custariam mais que a avaliação final inteira. Itere em 50
(`n_eval_docs: 50`, o padrão) e use os 500 uma única vez, para confirmar a
configuração antes de abrir a avaliação.

## Pendências

- [ ] Calibrar `top_k` no dev de 50 (padrão atual: 12, ou ~31% dos trechos)
- [ ] Pré-teste comparativo do modelo, se desejado (o padrão é `Qwen/Qwen3.5-9B`,
      medido e cabendo na GPU, mas não comparado com Llama-3.1-8B / Mistral-7B)
- [ ] Verificar a dispersão de C3 com 3 conjuntos fixos distintos no dev
- [ ] Rodada completa das 6 configurações e testes estatísticos
- [ ] Análise de erros por documento e por tipo de informação
- [ ] Teste de generalização no conjunto `eval`, aberto uma única vez

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
