# FINDSum: sumarização financeira com RAG e few-shot

Projeto de pesquisa de pós-graduação que compara recuperação de contexto e seleção de exemplos na sumarização de relatórios financeiros do **FINDSum Liquidity**. A geração usa exclusivamente a API do **OpenRouter**. Preparação, embeddings, FAISS e métricas são executados localmente.

## Painel atualizado por rodada

![Acompanhamento FINDSum](results/progress/dashboard.png)

Ao terminar `bash rodar_rodada.sh 100`, o script atualiza as métricas locais e este painel. A comparação usa o prefixo da coorte com seis braços completos nos três modelos. [Tabela completa e evolução](results/progress/README.md) · [Como funciona](docs/metricas-automaticas.md).

```bash
bash rodar_rodada.sh --metrics-only  # atualiza métricas e visual sem gerar respostas
```

Os arquivos ficam prontos para commit; o Bash não publica automaticamente no Git.

## Resultados parciais dos primeiros 100 documentos

Os três modelos concluíram os mesmos 100 documentos × seis braços, totalizando 1.800 respostas. A [matriz e interpretação das métricas](docs/metricas-primeiros100-20260927.md) e os [resultados por documento](results/first100-20260927/) estão versionados. C1 liderou as médias de BERTScore F1, ROUGE-L e METEOR em cada modelo; C2 superou C1t, mas ficou abaixo de C1. São resultados descritivos parciais, sem testes de significância ou confirmação de hipóteses. Este registro atualiza a situação da geração para este lote; o restante da coorte continua pendente.

## Estado verificado em 27/09/2026

| Etapa | Estado |
|---|---|
| Dev10 | Concluído: 10 documentos × 6 braços × 3 modelos = 180 respostas |
| Coorte final | 1.000 documentos reservados, IDs e ordem congelados |
| Preparação do Ling | 6.000/6.000 prompts válidos; nenhum documento excluído |
| Geração final do Ling | Primeiro lote concluído: 100 documentos, 600 respostas aceitas |
| Preparação do Gemma | 6.000/6.000 prompts válidos; maior entrada de 43.088 tokens |
| Qwen no full | Preparação local integral com tokenizer validado em 332 sondas do dev |
| Geração dos três modelos | Primeiro lote concluído: mesmos 100 documentos, 600 respostas por modelo |
| Métricas parciais | 1.800 pares avaliados; atualização incremental automática e painel no Git |
| Testes | 308 aprovados, excluindo `slow` e `test_real_data.py`; Ruff e diff check aprovados |

O comando dos lotes foi validado sem chamadas à API; depois foi executado um [teste pago de concorrência](docs/teste-vazao-openrouter-20260927.md), com respostas aproveitadas no full. O Bash agora executa os modelos em paralelo; Qwen/Gemma usam teto de 500 RPM por modelo com [retry adaptativo](docs/retry-adaptativo-openrouter.md). Ling mantém 20 RPM. O dev é exploratório; nenhuma hipótese foi confirmada como resultado do experimento final. Consulte o status salvo para acompanhar a execução posterior à atualização deste README.

## Rodar os três modelos com um Bash

```bash
bash rodar_rodada.sh 100 --dry-run  # confere o plano sem API
bash rodar_rodada.sh 100            # executa/retoma uma rodada nos três modelos
```

O número indica documentos da mesma coorte, com os mesmos IDs para os três modelos. Aceita de 1 a 1.000. O script preserva uma rodada incompleta e gera somente os braços faltantes. Os primeiros 100 documentos já estão concluídos nos três modelos; a próxima rodada selecionará os documentos 101–200, mantendo a ordem congelada.

**O comando de execução faz chamadas pagas de geração no Gemma/Qwen; `--dry-run`, `--metrics-only` e `--audit-ling-batch` não geram respostas.** A preparação Qwen é local, cobre os 1.000 documentos e não envia sondas à API. Tetos monetários e frequências permanecem os já definidos. A rodada encerra após o grupo escolhido; uma falha preserva o plano para retomada.

Auditar o primeiro lote Ling sem API:

```bash
bash rodar_rodada.sh --audit-ling-batch 1
```

Instruções, custos, arquivos de progresso e retomada: [guia do Bash](docs/rodadas-full.md). Resultado verificado: [auditoria do primeiro lote Ling](docs/auditoria-ling-lote1.md).

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

O contador do log, como `354/6000`, usa o índice global do experimento. No primeiro lote, o comando para ao completar `600/6000`; no segundo, `1200/6000`. São 600 gerações aceitas por lote; retries podem acrescentar chamadas à API.

Para indicar um lote específico, use um número de 1 a 10:

```bash
uv run --locked findsum ling-batch --batch 1 --execute
```

Os dez lotes são partes da mesma coorte final, sem novo sorteio ou substituição de casos. Mantenha as mesmas pastas para retomar; não apague respostas nem abra uma nova pasta para repetir um lote. A conferência inicial reconta os 6.000 prompts com quatro trabalhadores locais e mostra o avanço a cada 100 prompts. Operações bloqueantes mostram atividade a cada 15 segundos; a preparação ainda pode levar alguns minutos.

O executor consulta a cota ao iniciar, limita as tentativas ao saldo informado e aplica **20 chamadas por minuto**. Se houver 1.000 chamadas disponíveis, um lote de 600 deixa uma margem de 400 tentativas adicionais. Essa cota não garante disponibilidade do provedor; erros 429 do pool compartilhado ainda podem ocorrer. Evite outros clientes consumindo a mesma cota simultaneamente.

O Ling usa preço máximo zero, provedor fixo e **nenhum fallback pago**. Não é necessário calibrar o Qwen ou gerar com os outros modelos para iniciar os lotes do Ling. O guia completo está em [lotes do Ling](docs/lotes-ling-full.md).

## Qwen e Gemma em lotes de 100

Os dois modelos usam os mesmos grupos do Ling, com pastas e retomadas independentes. O Gemma pode ser preparado localmente com `uv run --locked findsum prepare-gemma`. O Qwen usa contagem local; prepare antecipadamente os 6.000 prompts com `uv run --locked python scripts/prepare_gemma_from_common.py --model qwen37 --source outputs/full-ling-prepared --output outputs/full-qwen-prepared`. Veja a sequência de preparação no [guia de lotes Qwen/Gemma](docs/lotes-qwen-gemma-full.md).

Depois de preparados, valide sem API com `findsum qwen-batch` ou `findsum gemma-batch`. Para executar um lote:

```bash
uv run --locked findsum qwen-batch --execute
uv run --locked findsum gemma-batch --execute
```

Execute o comando do modelo desejado e repita nas próximas sessões. Cada invocação processa no máximo um lote. Os tetos são cumulativos para os dez lotes: **US$ 18 de geração Qwen**, **US$ 9 de geração Gemma**; a preparação local Qwen não consome créditos. Nenhum desses comandos gera com o Ling.

O teto operacional dos modelos pagos é **100 chamadas por minuto por executor**, incluindo retries e sondas Qwen; o Ling gratuito mantém **20/minuto**. As requisições permanecem sequenciais: 100/minuto é um máximo, não uma velocidade garantida.

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
7. Ao completar os 1.000 documentos de cada modelo, calcula as métricas locais e as comparações pareadas. Não produz testes de hipótese finais a cada lote de 100.

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
| `findsum qwen-batch` / `findsum gemma-batch` | Um lote independente de 100, sem API por padrão |
| `findsum prepare-gemma` | Preflight local reutilizando os insumos comuns do Ling |
| `findsum verify-full` | Verifica o congelamento offline |
| `findsum prepare` | Prepara fonte, recuperação, exemplos e prompts locais |
| `findsum calibrate` | Comando legado de sondas remotas; não necessário nem acionado pelas rodadas atuais |
| `findsum run` | Executor integrado dos três modelos, com preparação local e Qwen já calibrado |

`calibrate` e `run` exigem `--allow-eval` para o conjunto final e `--execute` para enviar chamadas. Os três comandos `*-batch --execute` operam a coorte final congelada e não exigem `--allow-eval`. O executor integrado não é um comando exclusivo de Qwen/Gemma; não o use como continuação automática da rodada Ling já iniciada. Consulte [full congelado](docs/full-congelado.md) antes das etapas pagas.

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
- [Rodadas dos três modelos com Bash](docs/rodadas-full.md)
- [Operação dos lotes Ling](docs/lotes-ling-full.md)
- [Operação dos lotes Qwen e Gemma](docs/lotes-qwen-gemma-full.md)
- [Configuração congelada e etapas do full](docs/full-congelado.md)
- [Dev10 e limitações observadas](docs/dev10-openrouter.md)
- [BERTScore precisão/recall e METEOR](docs/dev10-metricas-complementares.md)
- [Leitura dos exemplos do dev10 em HTML](docs/dev10.html)
