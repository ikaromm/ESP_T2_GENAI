# FINDSum: sumarização financeira com RAG e few-shot

Projeto de pesquisa de pós-graduação que compara recuperação de contexto e seleção de exemplos na sumarização de relatórios financeiros do **FINDSum Liquidity**. A geração usa exclusivamente a API do **OpenRouter**. Preparação, embeddings, FAISS e métricas são executados localmente.

## Estado verificado em 27/09/2026

| Etapa | Estado |
|---|---|
| Dev10 | Concluído: 10 documentos × 6 braços × 3 modelos = 180 respostas |
| Coorte final | 1.000 documentos reservados, IDs e ordem congelados |
| Preparação do Ling | 6.000/6.000 prompts válidos; nenhum documento excluído |
| Geração final do Ling | Ainda não iniciada no último status verificado: 0/6.000 respostas |
| Qwen e Gemma no full | Preparação/validação completa e geração ainda pendentes |
| Testes | 262 aprovados, excluindo `slow` e `test_real_data.py`; Ruff e diff check aprovados |

O comando dos lotes foi validado sem chamadas à API. O dev é exploratório; nenhuma hipótese foi confirmada como resultado do experimento final. Consulte o status salvo para acompanhar a execução posterior à atualização deste README.

## Executar o Ling em lotes de 100

**Neste workspace, a preparação já está pronta em `outputs/full-ling-prepared`.** Execute os comandos a partir da raiz do repositório.

Conferir o congelamento e validar os prompts/progresso sem chamadas à API:

```bash
uv run --locked findsum verify-full
uv run --locked findsum ling-batch
```

Executar um lote ou retomar o primeiro lote incompleto:

```bash
uv run --locked findsum ling-batch --execute
```

Repita o mesmo comando nas próximas sessões. Cada invocação executa **no máximo um lote de 100 documentos**, com seis braços por documento: **600 gerações, sem contar retries**. Ao concluir um lote, o processo encerra; a próxima invocação seleciona o primeiro lote incompleto. Respostas já aceitas são conferidas e preservadas.

Para indicar um lote específico, use um número de 1 a 10:

```bash
uv run --locked findsum ling-batch --batch 1 --execute
```

Os dez lotes são partes da mesma coorte final, sem novo sorteio ou substituição de casos. Mantenha as mesmas pastas para retomar; não apague respostas nem abra uma nova pasta para repetir um lote. A conferência inicial reconta os 6.000 prompts e pode levar alguns minutos.

O executor consulta a cota ao iniciar, limita as tentativas ao saldo informado e aplica **20 chamadas por minuto**. Se houver 1.000 chamadas disponíveis, um lote de 600 deixa uma margem de 400 tentativas adicionais. Essa cota não garante disponibilidade do provedor; erros 429 do pool compartilhado ainda podem ocorrer. Evite outros clientes consumindo a mesma cota simultaneamente.

O Ling usa preço máximo zero, provedor fixo e **nenhum fallback pago**. Não é necessário calibrar o Qwen ou gerar com os outros modelos para iniciar os lotes do Ling. O guia completo está em [lotes do Ling](docs/lotes-ling-full.md).

## Dados e desenho experimental

O [manifesto de splits](data/interim/splits-liquidity.json) mantém um banco de 1.000 exemplos, 500 documentos de desenvolvimento e 1.000 de avaliação, com empresas separadas entre conjuntos. Os conjuntos são definidos dentro do FINDSum; não representam validação em um dataset externo.

A coorte final e sua ordem estão em [full-eval-cohort.csv](configs/full-eval-cohort.csv). A ordem foi sorteada com seed 42. Como todos os 1.000 documentos reservados entram no full, os lotes não selecionam um novo subconjunto com base em qualidade ou sucesso das respostas.

| Braço | Entrada entregue ao modelo |
|---|---|
| C1 | Fonte inteira, sem exemplos |
| C1t | Prefixo da fonte com orçamento igual ao contexto de C2 |
| C2 | Contexto recuperado por RAG, sem exemplos |
| C3 | RAG + quatro exemplos fixos |
| C4 | RAG + quatro exemplos aleatórios, com seleção reprodutível por documento |
| C5 | RAG + quatro exemplos selecionados por similaridade |

As cinco comparações são H1: C2/C1; H1b: C2/C1t; H2: C3/C2; H3: C4/C3; H4: C5/C4. **H3 é um controle de sensibilidade:** ausência de significância não prova equivalência. Ver [protocolo](docs/protocolo.md).

## Como a pipeline funciona

1. Reconstrói a prosa e as tabelas da tarefa, preservando os valores brutos e a referência FINDSum. Não aplica filtro financeiro de números.
2. Divide a fonte em chunks de 220 palavras, com sobreposição de 40. Os embeddings MiniLM-L6-v2 usam janelas que cobrem toda a prosa processada.
3. Recupera evidências com FAISS dentro do próprio relatório, usando uma consulta fixa da tarefa, top-12 e orçamento de até 3.072 tokens.
4. Seleciona os exemplos de um banco separado. C3 mantém quatro exemplos fixos; C4 usa sorteio reprodutível; C5 usa similaridade entre documentos. As demonstrações incluem documento e resumo de referência completos.
5. Monta os seis prompts e valida os limites antes de gerar. C1 não concatena chunks sobrepostos; C1t/C2 têm igualdade de tokens de contexto **e de prompt zero-shot**. C2–C5 recebem o mesmo contexto dentro de cada modelo.
6. Envia os prompts ao OpenRouter e salva requests, respostas, tentativas, consumo e metadados para retomada e auditoria.
7. Ao completar os 1.000 documentos do Ling, calcula as métricas locais e as comparações pareadas. Não produz testes de hipótese finais a cada lote de 100.

As instruções de geração são comuns aos braços, com orientação de **até 750 palavras**, temperatura 0, top-p 1 e reasoning desativado. Seed não é enviada aos endpoints sem suporte. Código dos prompts: [prompts.py](src/findsum_rag/prompts.py).

## Modelos e limites

| Modelo | Identificador OpenRouter | Provedor fixado | Uso planejado |
|---|---|---|---|
| Ling 3.0 Flash Fin | `inclusionai/ling-3.0-flash-fin:free` | Novita | Lotes gratuitos prontos para execução |
| Qwen3.7 Flash | `qwen/qwen3.7-flash` | Alibaba | Geração paga e contagem remota cobrada |
| Gemma 4 26B A4B | `google/gemma-4-26b-a4b-it` | Darkbloom | Geração paga; tokenizer local |

O perfil estabelece teto de **49.152 tokens de entrada**, reserva de **8.192 tokens de saída** e margem de contexto de **256 tokens**. A maior entrada validada do Ling tem **43.743 tokens**, dentro de sua janela configurada de 262.144. Overflow de entrada bloqueia a execução; não se corta silenciosamente a fonte de C1. O prefixo de C1t é o recorte intencional do desenho experimental.

A orientação de 750 palavras não é um corte rígido da resposta. Saídas encerradas por `finish_reason=length` são preservadas, sinalizadas e avaliadas como produzidas.

Antes da geração de cada modelo, seus 6.000 prompts devem passar pela pré-validação. A validação do Ling não substitui a contagem e o preflight dos outros modelos. Configurações: [full_openrouter.yaml](configs/full_openrouter.yaml) e [openrouter_full.json](configs/openrouter_full.json).

## Métricas e hipóteses

| Papel | Métricas |
|---|---|
| Primárias | BERTScore F1 e ROUGE-L F1 |
| Descritivas | BERTScore precisão/recall, METEOR e ROUGE-1/2 |

O BERTScore usa **XLNet, camada 5**, sem IDF ou rescale, com auditoria dos tokens efetivamente avaliados para rejeitar corte silencioso. METEOR usa NLTK/Treebank/Porter/WordNet em inglês. O plano aplica Holm aos dez testes — cinco contrastes × duas métricas primárias — **por modelo**. Dados incompletos não sustentam a análise final; não há uma conclusão inferencial conjunta dos três modelos.

Não há revisão humana, F1 numérico ou grounding factual como métricas do protocolo atual. Utilitários históricos de revisão permanecem no código, mas não são etapas obrigatórias. Os resultados do dev10 e as métricas complementares estão em [dev10](docs/dev10-openrouter.md) e [matriz de métricas](docs/dev10-metricas-complementares.md).

## Retomada e congelamento

Falhas HTTP transitórias recebem até seis tentativas, com esperas de 1, 2, 4, 8 e 16 segundos e tratamento de `Retry-After`. No executor de lotes, após esgotar as seis tentativas, uma nova rodada pode ser iniciada na retomada após uma hora. Cota ou orçamento insuficientes interrompem a etapa. Falhas de transporte com resultado desconhecido exigem auditoria para evitar duplicar uma chamada já processada.

Um lock de execução impede processos concorrentes na mesma rodada. O [contrato congelado](configs/full_openrouter.lock.json) verifica hashes de código, configuração, manifesto, coorte, dados, recursos locais e versões. Versões anteriores ficam em `configs/lock-history/`. Não altere esses insumos entre lotes; mudanças exigem revisão do congelamento. Aliases remotos e implementações dos provedores não são controlados por esse lock local.

Na validação real, foi corrigido um defeito que confundia ` | ` em prosa com linha de tabela e impedia C1t/C2 em 11 documentos. A correção reconhece os marcadores explícitos do serializador. Os 5.934 prompts inicialmente válidos ficaram idênticos e os 66 restantes foram completados, sem excluir casos. A preparação inicial permanece para auditoria.

## Arquivos para acompanhar a execução

Os caminhos em `outputs/` são artefatos locais; não acompanham necessariamente um clone do repositório.

| Caminho | Conteúdo |
|---|---|
| `outputs/full-ling-prepared/report.json` | Resultado da validação dos 6.000 prompts |
| `outputs/full-ling-prepared/selected-cases.csv` | IDs e ordem dos documentos |
| `outputs/full-ling-prepared/documents.json` | Fontes, referências, chunks recuperados e IDs dos exemplos |
| `outputs/full-ling-prepared/ling-free/preflight.json` | Prompts completos e contagens de tokens |
| `outputs/full-ling-batches/batch-status.json` | Progresso dos dez lotes |
| `outputs/full-ling-batches/ling-free/ledger.json` | Histórico de tentativas e respostas aceitas |
| `outputs/full-ling-batches/ling-free/*.request.json` | Entradas enviadas à API |
| `outputs/full-ling-batches/ling-free/*.response.json` | Saídas e metadados retornados |
| `outputs/full-ling-batches/ling-free/results/` | Métricas e comparações após os 1.000 documentos |

## Ambiente e demais comandos

Requer **Python 3.12** e `uv`. Para instalar as dependências congeladas e consultar a interface:

```bash
uv sync --locked
uv run --locked findsum --help
```

Os dados FINDSum, os artefatos preparados e os recursos locais de tokenização/avaliação precisam estar disponíveis. Instalar as dependências não reconstrói automaticamente a preparação. A credencial é fornecida por `OPEN_ROUTER_KEY` no ambiente ou no `.env` do workspace, um nível acima do repositório. Nunca inclua seu valor em código ou relatórios.

Além de `ling-batch`, estão disponíveis:

| Comando | Função |
|---|---|
| `findsum verify-full` | Verifica o congelamento offline |
| `findsum prepare` | Prepara fonte, recuperação, exemplos e prompts locais |
| `findsum calibrate` | Faz a contagem remota do Qwen; sondas cobradas somente com `--execute` |
| `findsum run` | Executor integrado dos três modelos, com preparação local e Qwen já calibrado |

`calibrate` e `run` exigem `--allow-eval` para o conjunto final e `--execute` para enviar chamadas. **`ling-batch --execute` é o caminho atual dos lotes gratuitos** e não exige `--allow-eval`. O executor integrado não é um comando exclusivo de Qwen/Gemma; não o use como continuação automática da rodada Ling já iniciada. Consulte [full congelado](docs/full-congelado.md) antes das etapas pagas.

A estimativa documentada em 26/09 para os três modelos, **incluindo todo o Ling como pago e a calibração Qwen**, foi de US$ 42,03, com reserva sugerida de US$ 60. É uma projeção baseada no dev10, não cobrança desta preparação nem garantia de custo final. Os lotes atuais do Ling continuam gratuitos. Premissas, faixas de preço e cenário de saídas longas: [orçamento](docs/orcamento-full.md).

## Verificação do código

```bash
uv run --locked pytest -q -m 'not slow' --ignore=tests/test_real_data.py
uv run --locked ruff check src scripts tests
```

O backend de geração local, seus perfis antigos e suas dependências específicas foram removidos. Torch/Transformers continuam necessários para embeddings, tokenização e métricas. PoCs e resultados históricos ficam preservados para rastreabilidade; não substituem a preparação atual. O snapshot antigo do Graphify está desatualizado.

## Documentação

- [Protocolo científico](docs/protocolo.md)
- [Dataset e splits](docs/dataset.md)
- [Operação dos lotes Ling](docs/lotes-ling-full.md)
- [Configuração congelada e etapas do full](docs/full-congelado.md)
- [Dev10 e limitações observadas](docs/dev10-openrouter.md)
- [BERTScore precisão/recall e METEOR](docs/dev10-metricas-complementares.md)
- [Leitura dos exemplos do dev10 em HTML](docs/dev10.html)
