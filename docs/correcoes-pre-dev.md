# Correções antes do piloto dev — 2026-09-26

Validação posterior em cinco documentos: [validacao-5docs-openrouter.md](validacao-5docs-openrouter.md).
Ela confirmou erros factuais e um corte residual de células no orçamento final;
a divisão de tabelas por linhas não resolveu esse último ponto.
Uma correção posterior protege as células também no orçamento final do prompt;
ver [preparação do dev OpenRouter](preparacao-dev-openrouter.md).

As correções tratam a construção de evidência e instruções. Não constituem
aprovação factual do modelo nem confirmação de hipóteses. O piloto de 50
relatórios e a avaliação final não foram executados nesta etapa.

## Implementação

- Recuperação: consulta fixa pela tarefa (`TASK_INSTRUCTIONS`), sem referência
  humana e sem números específicos dos documentos. O vetor integral do documento
  permanece para C5. FAISS retorna por similaridade; essa ordem é preservada
  antes de aplicar o orçamento, em vez de priorizar o início da fonte.
- Exemplos: fronteiras EXAMPLES/END_EXAMPLES e TARGET_REPORT/END_TARGET_REPORT,
  identificador do relatório demonstrativo e aviso de entrada abreviada. Mesmas
  regras em todos os braços: apenas o alvo fornece fatos; conferir entidade,
  período, direção e valor. Número e estratégias de seleção permanecem iguais.
- Tabelas: seleção pela seção declarada no registro (`mda_liquidity_tables` ou
  `mda_result_tables`), sem relacionar artificialmente índices locais a marcadores.
  Todas as células dessas tabelas entram na fonte C1; demais seções ficam fora.
  Retirados cortes silenciosos de 12 tabelas/120 células. Blocos para recuperação
  respeitam linhas completas, sem sobreposição. Limites de tokens continuam
  validados e podem impedir uma rodada maior; nenhum overflow é aceito.
- Valores: strings com `&` preservadas e marcadas como ambíguas. Nenhuma escala,
  moeda ou sinal é reconstruído sem evidência. O prompt orienta omitir montantes
  de interpretação incerta. Isso não recupera cabeçalhos ausentes do dataset.
- Diagnóstico: `factual_warnings` lista números que aparecem somente nos exemplos
  e montantes com escala ausentes da fonte. São candidatos para revisão, sujeitos
  a falsos positivos/negativos. Presença de um número não comprova entidade,
  período, sinal ou relação; ausência não prova falsidade no 10-K original.
  Não se excluem nem reescrevem gerações em função desses alertas.

O [README oficial do FINDSum](https://github.com/StevenLau6/FINDSum#faq) descreve
as tuplas como linha, coluna, valor, data e coordenadas da célula. Não informa
um campo independente de unidade nem resolve, por si, a numeração entre listas
por seção e marcadores da prosa. A política por seção explicita essa limitação.

A mudança de fonte invalida comparações diretas com os escores anteriores como
se fossem a mesma configuração. Artefatos antigos foram preservados e cada
validação usa diretório novo. Para executar o piloto de 50, usar uma configuração
API própria; `configs/test_50.yaml` continua sendo o perfil local original.

## Verificação

213 testes passaram (excluídos slow e test_real_data.py). Incluem regressões para
consulta de tarefa, prioridade semântica, seleção de tabelas por seção, preservação
de células/valores ambíguos, fronteiras do prompt e alertas de números transferidos.
Ruff passou. Os testes demonstram comportamento do código, não fidelidade da LLM.

Rodada intermediária `outputs/screen-fixes-20260926/`: três documentos dev,
32 exemplos, seis braços. Ling completou 18 saídas; Qwen e Gemma continuaram
bloqueados por 429 dos pools dos provedores após seis tentativas. Essa rodada
precede a divisão das tabelas em blocos e o campo factual_warnings. A inspeção
já encontrou novos erros de escala/direção e transferência de fatos, apesar das
instruções reforçadas. Não aprovar qualidade a partir do sucesso técnico.

Rodada com tabelas divididas: `outputs/screen-fixes-tables-20260926/`.
18 prompts válidos, 17 gerações aceitas; ABT/C1 atingiu `finish_reason=length`
aos 1.536 tokens e foi rejeitado. O limite não foi elevado só para esse caso:
o perfil PoC passou a 3.072 tokens para todos os braços e uma nova rodada completa
foi criada em `outputs/screen-fixes-3072-20260926/`. Não misturar escores entre
rodadas ou aproveitar seletivamente saídas anteriores.

Na rodada de 1.536 tokens, os 17 resultados aceitos tiveram 231 alertas de montante
com escala e 10 de número exclusivo de exemplos (contados por valor e resumo).
Esses números NÃO são contagens de erros factuais. Os alertas foram conferidos
contra os exemplos efetivamente apresentados, sem usar partes não entregues do
relatório demonstrativo. Os metadados preliminares foram preservados em
`audit-before-excerpt-correction/`; texto gerado e respostas da API não mudaram.
Uma inconsistência verificável por leitura é ABEO/C5 dizer que o caixa aumentou
de 130,4 para 13,6 milhões. Esse tipo de erro de direção não é resolvido nem
necessariamente detectado pela coincidência numérica.

A redução da contaminação não está demonstrada estatisticamente. O mapeamento
inventado de tabelas foi removido, mas unidades/cabeçalhos ausentes não foram
recuperados. Para aprovação factual é necessário revisar as afirmações e, quando
a fonte distribuída não basta, conferir o relatório original ou classificar
como evidência insuficiente. Não atribuir unidades por suposição.


A rodada com 3.072 tokens também rejeitou ABT/C1 por `length`; 17 saídas aceitas.
Ela fica preservada como evidência de que elevar o limite sozinho não resolveu.
Foi acrescentado o mesmo pedido de até 750 palavras, sem repetir fatos, a todos
os braços. Com essa instrução, a regressão isolada ABT/C1 terminou em `stop`,
244 palavras e 449 tokens, custo reportado zero. Os 18 prompts receberam nova
pré-validação; os contextos, exemplos e igualdade C1t/C2 foram preservados.
A validação final reenvia esses prompts salvos, reaproveitando apenas a regressão
ABT/C1 idêntica, em `outputs/regression-length-20260926/`. Não reexecuta seleção
nem embeddings, pois não houve nova mudança nesses componentes.

Resultado da validação final: **18/18 gerações aceitas do Ling**, todas `stop`,
entre 199 e 585 palavras, máximo de 1.324 tokens de saída, custo reportado zero.
Os 18 prompts respeitam fonte C1, igualdade de contexto/prompt C1t/C2 e contexto
compartilhado C2–C5. `validation.json` registra essas conferências. O teto de
3.072 tokens é apenas uma margem de execução; a instrução pede até 750 palavras.

Foram registrados 263 alertas de escala e 11 de números exclusivos de exemplos,
sem interpretar esses totais como erros confirmados. A inspeção ainda encontra
montantes e relações que exigem revisão, portanto qualidade factual não aprovada.
As fichas cegas dos 18 casos estão em `ling/human-review.json`, com mapa separado;
nenhum julgamento humano foi preenchido. Não houve BERTScore nem teste de H1–H4.
Quota consultada ao final: 87 usadas, 913 restantes, limite 1.000; variável no tempo.

Conclusão operacional: correções do pipeline verificadas e problema de extensão
resolvido nesta amostra. Pendem a revisão factual das saídas, a validação de
unidades ausentes contra fontes originais e disponibilidade de Qwen/Gemma para
completar a comparação dos três modelos. Não tratar a execução como aprovação
para avaliação final nem afirmar que toda contaminação foi eliminada.
