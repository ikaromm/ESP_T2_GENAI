# Full OpenRouter congelado

Organização realizada em 26/09/2026, **sem executar o full nem chamadas de calibração**. A geração local de LLM foi removida; embeddings, FAISS e métricas permanecem locais. Alterações não foram commitadas nem publicadas.

## Contrato

- FINDSum Liquidity; manifesto original preservado. São os 1.000 relatórios reservados de avaliação, com ordem aleatória seed 42 e IDs em `configs/full-eval-cohort.csv`. Como todos os 1.000 entram, a aleatoriedade organiza a ordem, não escolhe outro conjunto.
- Seis braços C1/C1t/C2/C3/C4/C5, total de 18.000 gerações para três modelos. Exemplos separados por empresa; quatro exemplos em C3–C5.
- Fonte com prosa e tabelas originais da tarefa, sem filtro de valores financeiros. MiniLM-L6-v2, chunks 220/40 palavras, top-12, contexto RAG de 3.072 tokens.
- Ling Flash Fin gratuito/Novita, Qwen3.7 Flash/Alibaba e Gemma 4 26B A4B/Darkbloom. Sem fallback automático e sem mistura de provedores no bloco de seis braços.
- Temperatura 0, top-p 1, reasoning desativado; seed não enviada a endpoints sem suporte. Até 750 palavras como instrução. Máximo de entrada 49.152 tokens, saída 8.192, margem de contexto 256.
- BERTScore XLNet camada 5 e ROUGE-L F1 primários; precisão/recall BERTScore, METEOR e ROUGE-1/2 descritivos. Cinco contrastes e Holm sobre dez testes por modelo. H3 é sensibilidade; não significância não demonstra equivalência. Sem inferência conjunta entre modelos.
- Métricas sem revisão humana ou avaliação financeira numérica. Respostas encerradas por limite de saída são preservadas, sinalizadas e pontuadas. Nenhuma hipótese depende apenas de respostas selecionadas por sucesso.

## Travas

`configs/full_openrouter.lock.json` registra hashes de código, configurações, manifesto, coorte, dados da tarefa, dependências, snapshots locais de modelos/tokenizers e WordNet. `findsum verify-full` verifica essas identidades offline. Mudanças exigem uma nova decisão de congelamento, sem sobrescrever silenciosamente o lock.

O manifesto preparado deve conter a mesma coorte e configuração. Antes da geração de cada modelo, seus 6.000 prompts devem passar pelo preflight; overflow é rejeitado, não resolvido cortando silenciosamente a fonte de C1. O controle C1t continua sendo o prefixo intencional do desenho. Se algum caso não couber, a preparação para e o problema é registrado; não existe substituição automática por casos mais fáceis.

Retries em falhas transitórias: até seis tentativas, esperas 1/2/4/8/16 segundos e respeito a Retry-After. Limite de 20 chamadas/minuto, incluindo tentativas; documento não é sinônimo de chamada. A cota gratuita é consultada, e novos blocos não começam sem saldo suficiente para suas chamadas pendentes. Retomadas preservam respostas aceitas. Cota diária não garante disponibilidade do pool do provedor.

Tetos pagos: Qwen geração US$ 18, calibração US$ 18, Gemma US$ 9. Ling pago US$ 15 é apenas reserva no cenário orçamentário. Falta de cota ou orçamento interrompe a etapa; análises inferenciais rejeitam dados incompletos. Ver [orçamento](orcamento-full.md).

O lock fixa o que controlamos localmente. Aliases e implementações remotas podem mudar no provedor; identidade retornada, uso, erros e custo ficam registrados, sem promessa de determinismo absoluto.

## Etapas futuras — não executadas nesta organização

Verificação offline:

```bash
uv run --locked findsum verify-full
```

Após decidir iniciar, a preparação local monta os prompts dos 1.000 casos:

```bash
uv run --locked findsum prepare --full --output outputs/full-prepared
```

A calibração Qwen sem `--execute` somente verifica os pré-requisitos. Acrescentar `--execute` autoriza as sondas cobradas:

```bash
uv run --locked findsum calibrate --prepared outputs/full-prepared --output outputs/full-qwen-prepared --budget-usd 18 --allow-eval
```

Com a calibração concluída, o comando abaixo faz a conferência antes da geração; acrescentar `--execute` inicia chamadas dos três modelos:

```bash
uv run --locked findsum run --prepared outputs/full-prepared --qwen-prepared outputs/full-qwen-prepared --output outputs/full-run --qwen-budget 18 --gemma-budget 9 --allow-eval
```

Não há necessidade de escolher novos modelos, prompts ou métricas. Falta executar a preparação e validar a cobertura real dos 18.000 prompts, incluindo contagem remota Qwen, antes de iniciar geração. O congelamento não afirma que essa validação já passou.

Os antigos perfis `experiment.yaml`, `test_50.yaml`, `eval_1000.yaml` e o backend local foram removidos. Resultados antigos e utilitários históricos ficam para rastreabilidade. Backup anterior à migração está no workspace, fora do repositório: `migration-backup-openrouter-20260926/`.

## Verificação realizada

Lock original de 26/09 (arquivado): `17f9a65ffb2284ece04a731504a2dceeff25f2e19d0e4a6a0ae1174a692b9df7`. Coorte: 1.000 IDs; 12 arquivos de dados e 45 arquivos de código/configuração registrados. Suite sem testes slow e sem test_real_data.py passou; Ruff e git diff --check passaram. Nenhuma chamada de geração ou calibração foi feita nesta organização.

## Lotes Ling — 27/09/2026

A pedido do usuário, permitido iniciar Ling de forma independente em dez lotes fixos de 100, sem alterar os 1.000 IDs ou os prompts. A validação global dos três modelos deixa de ser pré-requisito para começar Ling; cada modelo exige seus próprios 6.000 prompts válidos. Qwen/Gemma continuam pendentes. O lock de 26/09 está arquivado em configs/lock-history; consulte o lock atual e [guia dos lotes](lotes-ling-full.md). Nenhuma geração foi iniciada nesta organização.

## Estado vigente — 27/09/2026

Lock ativo: `c0d3eb15c1eb74f516b4d63e22e82eeac614a53d6d63789f61db970c44ce88f1`. Ling preparado em `outputs/full-ling-prepared`: 1.000 documentos, 6.000 prompts válidos, maior entrada 43.743 tokens, teto 49.152 e reserva de saída 8.192. Os 5.934 prompts inicialmente válidos ficaram idênticos após a correção do reconhecimento de tabelas; os 66 restantes foram completados sem substituir casos. 262 testes passaram (sem slow/test_real_data.py), além do preflight real. Gerações OpenRouter nesta preparação: zero. Qwen/Gemma seguem pendentes.


## Atualização operacional em 27/09/2026: três executores de lotes

Disponíveis `findsum ling-batch`, `findsum qwen-batch` e `findsum gemma-batch`, todos com os mesmos dez grupos de 100 documentos. Sem `--execute`, não enviam chamadas. Qwen exige antes a calibração paga dos 6.000 prompts; Gemma usa preparação local a partir dos insumos congelados do Ling. Guia: [lotes Qwen/Gemma](lotes-qwen-gemma-full.md).

Lock operacional atual: `e40bea867c4ca0ff17ed006c198c1710400f1a7761d3ff09f3bc6195bba096c0`. Preserva explicitamente a preparação Ling vinculada a `c0d3eb15c1eb74f516b4d63e22e82eeac614a53d6d63789f61db970c44ce88f1` e sua identidade de retomada. A compatibilidade exige invariância dos insumos científicos; apenas os arquivos operacionais explicitamente relacionados podem diferir. Coorte, fonte, recuperação, exemplos, prompts Ling, métricas e limites permaneceram iguais.

Verificação real paralela do Ling: 6.000 prompts e 213 respostas já aceitas conferidos em 79,2 segundos, com quatro trabalhadores locais, sem API. O usuário iniciou a geração Ling em seu terminal; os números continuam mudando e o ledger é a fonte do progresso atual. Os novos logs não alteram um processo já carregado: aparecem na próxima execução.

Validação de código: 270 testes passaram, excluindo `slow` e `test_real_data.py`. Logs incluem etapas, progresso de prompts, atividade em esperas, requisições, retries, latência, tokens e custo. O paralelismo se restringe às operações locais de validação/preparação; não aumenta a concorrência das chamadas OpenRouter.

Gemma também preparado integralmente: 6.000 prompts, zero erros, maior entrada 43.088 tokens; relatório local `outputs/full-gemma-prepared/report.json`. Qwen ainda exige sondas pagas; a validação sem `--execute` passou e não enviou chamadas.


## Teto dos pagos atualizado em 27/09/2026

Por solicitação do usuário, geração paga e sondas Qwen passam a no máximo 100 requisições/minuto por executor; Ling gratuito mantém 20. Retries contam na janela persistida e as chamadas continuam sequenciais. Sem mudanças nos prompts, coorte, parâmetros de geração, métricas ou tetos monetários. Nenhuma chamada API foi enviada nesta atualização.

Lock operacional vigente: `c010b581a16b9011329df5eeb83bc798c44b9776df8da6237e3e59e7e7f037c3`. O anterior foi arquivado. Preparações Ling (`c0d3...`) e Gemma (`e40bea...`) são aceitas explicitamente por modelo, mantendo a identidade dos ledgers. Verificação real de ambos os diretórios passou; 276 testes passaram, além de Ruff e diff check.
