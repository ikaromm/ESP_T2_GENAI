# Dev com evidência conservadora — 2026-09-26

> Histórico do dev anterior. Em 26/09, após reler o planejamento, o usuário
> fixou ROUGE/BERTScore e retirou avaliação numérica/factual. O filtro de tabelas
> foi removido do código. Os resultados abaixo permanecem históricos, sem
> recalcular nem alterar a fonte usada naquela execução. Ver [protocolo](protocolo.md).


Nota posterior à execução: o usuário fixou os três modelos e avaliação somente
por métricas, sem revisão humana. As fichas vazias abaixo são artefatos históricos,
não pendência. Resultados ruins não exigem aperfeiçoamento contínuo do prompt;
o fechamento trata de entrada, controles, execução e plano de análise. Consulte
o [guia HTML](pipeline.html) para o estado atual e as regras full em preparação.

O usuário autorizou corrigir os problemas observados e repetir dev50 no
OpenRouter. Depois de estabilizar a execução, autorizou Qwen3.7 Flash/Alibaba e
Gemma 4 26B A4B/Darkbloom. A avaliação final continua fechada.

## Mudanças comuns aos seis braços

- Células sem rótulo de coluna/linha, período, unidade explícita ou valor
  atômico, com valores conflitantes ou direção ambígua ficam em quarentena.
  As tuplas originais não são alteradas: `table-audit.json` conserva cada célula
  e os motivos. Os índices de posição não são interpretados como cabeçalhos.
- Neste dev50, 9.448 células foram auditadas e nenhuma foi elegível. Portanto,
  esta rodada gera a partir da prosa. Não demonstra capacidade de sumarização
  tabular e não deve ser descrita como uso de todas as tuplas originais.
  C1 recebe toda a fonte após esse pré-processamento, sem duplicação de chunks.
- Atributos HTML `style` delimitados são removidos, conservando a prosa adjacente.
- Quatro documentos completos dos exemplos entram em C3–C5, com resumos
  completos e o mesmo banco de 1.000 documentos. Isso evita a incompatibilidade
  artificial entre um prefixo de 350 palavras e a referência inteira, mas não
  comprova suporte de todas as referências FINDSum.
- Instrução comum reforça sinais de fluxo de caixa e proíbe criar variações ou
  razões que a fonte não apresenta. Não há correção seletiva das saídas de um braço.

Os parâmetros mudaram simultaneamente, por isso a rodada anterior não é um
controle que isole o efeito de uma única correção. A comparação dos seis braços
continua pareada dentro da nova rodada. Manifesto e conjunto eval preservados.

## Artefatos e execução

- Perfil: `configs/dev_openrouter_ling_evidence.yaml`.
- Preparação: `outputs/dev50-evidence-prepared-20260926/`; 300 prompts Ling e
  300 Gemma validados, máximo 41.232 / 41.028 tokens, respectivamente.
- Ling: `outputs/dev50-ling-evidence-20260926/`; consultar `report.json` para
  o estado real. Novita fixado, reasoning desligado, temperature 0, saída até
  3.072 tokens, custo máximo zero, sem fallback. Retomada exige mesmos prompts.
- Exemplo de preparo: `python scripts/prepare_dev_openrouter.py --config
  configs/dev_openrouter_ling_evidence.yaml --output PASTA_NOVA`.

Estimativa Gemma para 300 chamadas e todas as saídas chegando a 3.072 tokens:
US$ 0,466234422, sem cache/retries. Executor reserva conservadoramente 49.152
entradas + 3.072 saídas por tentativa: US$ 0,8220672 para 300 chamadas. Não é
custo realizado. A geração paga permanece condicionada à conferência do
contrato de cada resposta.

## Calibração de Qwen3.7 pela API

Não foi identificado tokenizer local oficial exato. O catálogo OpenRouter tem
`hugging_face_id: null`. A documentação retorna contagens exatas de entrada em
`usage` e distingue isso de estimativas locais:
[OpenRouter](https://openrouter.ai/docs/api_reference/overview),
[QwenCloud](https://docs.qwencloud.com/developer-guides/run-and-scale/token-counting).

`scripts/prepare_qwen_remote.py` implementa sondas de no máximo um token, que
não contam como resumos. Mede o prompt completo e ajusta prefixos com consultas
à API. Para o contexto, registra o incremento de tokens em um envelope fixo de
mensagem, descontando a contagem do envelope vazio. **Não afirma ter acesso aos
IDs ou à contagem local de tokens nativos do texto isolado.** Exige igualdade
C1t/C2 tanto nessa medida operacional de contexto quanto no prompt completo,
além de contexto idêntico C2–C5. O campo `context_count_method` distingue essa
medida das contagens locais usadas em Ling/Gemma. Documentar essa diferença
antes de comparar modelos; a contagem integral do prompt é sempre conferida
na geração.

Cada sonda preserva request/response, tem cache por conteúdo, reserva de gasto
antes do envio e limite de seis tentativas para 429. Timeout não é repetido sem
auditoria. O comando requer orçamento explícito; falha de calibração bloqueia
a geração. A calibração real validou os 300 prompts: 1.831 tentativas de sonda, sendo
1.810 aceitas e 21 respostas HTTP 429, com custo reportado de US$ 0,50611544.
As 429 foram recuperadas dentro do limite de tentativas. Não substituir silenciosamente o tokenizer por outro Qwen.

O Qwen sobe de faixa quando a entrada alcança 32 mil tokens. O executor limita
49.152 entradas e admite no máximo US$ 0,10 / 0,40 por milhão, com Alibaba
fixado. Fonte: [Alibaba](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-flash).

## Critério de conclusão

Critérios técnicos da rodada: presença dos casos, conclusão das respostas, custo,
provedor, controles e contagens conferidos. A política final de saídas truncadas
ainda está em elaboração. Pela decisão posterior do usuário, revisão humana não
será realizada; métricas automáticas não serão apresentadas como aprovação factual. Registrar custo de calibração
separadamente do custo dos resumos; nenhuma hipótese é confirmada pelo dev.

## Resultado confirmado do Ling corrigido

O lote concluiu as 300 tentativas, sem erros HTTP e com custo reportado zero.
Foram 298 respostas aceitas tecnicamente e duas rejeitadas por `finish_reason=length`
(3.072 tokens): ALAC/C5 e AMPGW/C5. Os arquivos brutos são, respectivamente,
`ling/api/00096.response.json` e `00132.response.json`. Não foram repetidas
seletivamente nem tratadas como respostas completas. Nove das respostas completas
excedem 750 palavras. A rodada permanece incompleta para comparações que exigem
os seis braços de todos os documentos.

O consumo de todas as 300 respostas, incluindo as rejeitadas, foi 6.317.150 tokens
de entrada e 128.397 de saída. `validation.json` separa esses totais dos totais das
298 respostas aceitas. Controles C1t/C2, contexto C2–C5, manifesto, hashes e prompts
preparados foram conferidos. Fichas humanas continuam sem julgamentos.

Inspeção do assistente identificou em ACET/C4 do Ling nomes/valores de exemplos
(JPMorgan, 641,4 e 150,0) ausentes da fonte-alvo e do contexto entregue. A resposta
tem 1.756 palavras. Isso é sinal concreto de contaminação pelos exemplos; não é
uma revisão humana completa nem uma medida de frequência no lote. O Gemma no
mesmo caso não reproduziu esse erro específico, o que não basta para aprová-lo.

Gemma26 foi executado com teto de US$ 1; Qwen37 teve teto de US$ 3 para
calibração e US$ 2 para geração. Reservas não utilizadas são liberadas somente
após validação da resposta e do custo reportado; falhas/pendências mantêm a reserva.
Os limites continuam valendo na retomada, com cache de chamadas concluídas.


Preparação Qwen concluída em `outputs/dev50-qwen37-evidence-prepared-20260926/`:
entrada total prevista de 6.305.418 tokens nos 300 resumos, máximo de 41.124 por
prompt. O dry-run do executor validou hashes, controles e sondas. A geração real
foi concluída em `outputs/dev50-qwen37-evidence-20260926/`. Sondas já aceitas não
foram repetidas ao retomar a calibração.


## Fechamento dos três lotes

Gemma26: 300/300 respostas completas em 300 tentativas, US$ 0,282164602,
6.273.391 tokens de entrada e 84.919 de saída. Nenhuma resposta acima de 750
palavras; 68 abaixo de 50 palavras (indicador descritivo, não taxa de erro).

Qwen37: 300/300 completas em 300 tentativas, US$ 0,463156762 de geração,
6.305.418 tokens de entrada e 114.816 de saída. Três respostas acima de 750 palavras
e uma abaixo de 50. Com a calibração, total Qwen US$ 0,969272202.

Custo reportado total do dev corrigido: US$ 1,251436804. Nenhum erro HTTP nas
900 tentativas de geração; duas respostas Ling truncadas. Os erros 429 ocorreram
nas sondas de calibração Qwen e foram recuperados com retries limitados.

Resultados consolidados e limitações em
[`outputs/dev50-evidence-comparison-20260926/relatorio.md`](../outputs/dev50-evidence-comparison-20260926/relatorio.md).
Os pagos foram exportados para suas pastas `results/`, com métricas e fichas
humanas vazias. Ling permanece incompleto. Nenhuma hipótese confirmada; eval não
executado. Todos os processos desta rodada encerraram. A definição da fonte tabular e as regras do executor/análise full continuam
pendentes. Erros de conteúdo dos modelos são resultados a medir, não requisitos
de aprovação humana ou de respostas perfeitas.
