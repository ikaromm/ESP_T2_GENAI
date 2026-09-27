# Protocolo — alinhamento ao planejamento, 26/09/2026

Base: **Stage 2 - Planejamento - Projeto de Pesquisa - grupo 4.docx**, seção
“Estratégia de avaliação”. O documento nomeia **ROUGE e BERTScore** e prevê
médias, medianas, distribuições e, quando aplicável, testes estatísticos.
O [alinhamento ao DOCX](alinhamento-planejamento.md) distingue essas definições
das decisões operacionais tomadas depois.

**Escopo atual:** comparar a similaridade dos resumos gerados às referências
FINDSum. Por decisão explícita do usuário, não haverá avaliação numérica,
grounding, auditoria factual ou revisão humana. O planejamento previa complementos
numéricos/factuais; essa parte foi retirada do escopo. Não procurar o prompt
perfeito nem exigir resultados favoráveis para prosseguir.

Modelos mantidos: Ling 3.0 Flash Fin gratuito/Novita, Qwen3.7 Flash/Alibaba e
Gemma 4 26B A4B/Darkbloom. O dev50 anterior foi executado; seus artefatos permanecem
inalterados. **Full não executado; executor integrado implementado, validado no dev10.**
O [guia HTML](pipeline.html) mostra separadamente esse histórico e o desenho atual.

## Dataset e entrada

- Tarefa Liquidity do FINDSum, em inglês. O corpus fornece extratos e resumos;
  “fonte inteira” significa o extrato disponível, não o 10-K original completo.
- Preservar o manifesto por empresa: 1.000 exemplos, 500 dev, **1.000 eval**.
  Não misturar empresas entre esses conjuntos nem escolher casos por seus escores.
- Selecionar sem reposição, com semente 42, dentro do split reservado. Salvar
  ordem, `doc_id`, empresa, relatório, split e linha de origem em
  `selected-cases.csv`, além da seleção JSON e hash do manifesto.
  Com todos os 1.000 do eval, sorteia-se a ordem, não uma nova composição.
  O dev50 antigo usou os primeiros 50 IDs do manifesto, não esse novo sorteio.
- Concatenar os três segmentos e suas referências. Remover marcadores de
  formatação sem corrigir números, unidades ou relações contábeis.
- **Preservar a prosa e as tuplas brutas das tabelas da seção da tarefa.**
  Serializar os campos para texto; não excluir células por coluna vazia,
  unidade ausente, `&` ou conflito de valores. Não inventar campos ausentes.
  Tabelas de outras seções não entram. A representação é comum aos seis braços.
- O filtro antigo retirou todas as 9.448 células auditadas no dev50. Foi um
  desvio de pré-processamento e foi removido do código; isso não muda os prompts
  nem as respostas já salvos. A nova fonte exige novo preflight antes de gerar.

## RAG e few-shot

1. Segmentar a fonte em chunks; indexar localmente com embeddings e FAISS.
   Recuperar no próprio relatório usando a instrução fixa de Liquidity como
   consulta. A referência do alvo nunca entra no prompt ou na recuperação.
2. Embeddings de documentos/exemplos cobrem toda a prosa, com janelas que cabem
   no encoder, média ponderada por tokens e normalização. O vetor do documento
   inteiro é usado na escolha dinâmica de exemplos. Esse detalhamento não era
   especificado pelo DOCX.
3. Recuperar até 12 chunks e limitar o contexto a 3.072 tokens no perfil atual.
   C1 não duplica prosa pela sobreposição de chunks. C1t/C2 têm contextos e
   prompts zero-shot com contagens iguais. C2–C5 recebem o mesmo contexto
   recuperado dentro de cada modelo.
4. C3 usa os quatro primeiros IDs ordenados do banco; C4 sorteia quatro com
   semente derivada de `42:doc_id`; C5 seleciona quatro por similaridade entre
   documentos. Os pares usam prosa integral e referência integral do banco
   separado de exemplos; as tabelas desse banco não são carregadas.
5. Manter as mesmas instruções e parâmetros entre braços de cada modelo.
   Delimitar exemplos e alvo. Registrar IDs, prompts e contextos efetivos.
   Não ajustar a saída nem repetir uma resposta porque seu escore foi baixo.

## Configurações e cinco hipóteses operacionais

C1: fonte inteira, zero exemplos. C1t: prefixo com orçamento igual a C2.
C2: RAG, zero exemplos. C3: RAG + quatro fixos. C4: RAG + quatro aleatórios.
C5: RAG + quatro semelhantes. O DOCX descreve cinco configurações C1–C5;
C1t é o controle adicional de orçamento acordado no projeto.

| Hipótese | Contraste | Pergunta |
|---|---|---|
| H1 | C2 − C1 | Qual a diferença com recuperação? |
| H1b | C2 − C1t | Qual a diferença com orçamento de contexto controlado? |
| H2 | C3 − C2 | Qual o efeito de incluir quatro exemplos fixos? |
| H3 | C4 − C3 | Quanto muda ao trocar fixos por aleatórios? |
| H4 | C5 − C4 | Qual o efeito da seleção semântica? |

O DOCX tem uma hipótese geral; esses cinco contrastes a operacionalizam.
**Nenhuma hipótese está confirmada.** Resultados negativos ou ausência de diferença
são resultados válidos. H3 é controle de sensibilidade: não significância não
prova equivalência.

## Métricas e leitura dos resultados

- **BERTScore F1:** similaridade contextual entre geração e referência;
  implementação `xlnet-base-cased`, camada 5, sem reescala por baseline,
  com cobertura integral auditada; sem corte em 512 tokens.
- **ROUGE-1, ROUGE-2 e ROUGE-L (F1):** sobreposição textual com a referência.
- **BERTScore e ROUGE-L** são os dois endpoints dos cinco contrastes.
  ROUGE-1/2 também são reportados descritivamente. O DOCX não escolhe variantes
  nem endpoints: essa é uma definição operacional explícita deste protocolo.
- Salvar escores por documento/braço/modelo, médias, medianas e distribuições.
  Reportar deltas emparelhados e tamanho de efeito, não só p-valores.
  Escores representam similaridade, não porcentagem de correção factual.
- Cinco contrastes × duas métricas = dez testes bilaterais de Wilcoxon, Holm,
  alfa 0,05, **separadamente por modelo**. Não usar essa correção para alegar uma
  conclusão conjunta entre modelos ou escolher somente o modelo significativo.
- O plano versão 3 registra XLNet e rejeição de truncamento; a versão 2
  já havia substituído o antigo endpoint numérico. Artefatos com o plano
  antigo não devem ser reinterpretados como se tivessem usado este protocolo.
  Ausência de braço, documento ou métrica invalida a comparação completa.

Funções numéricas e comandos de revisão antigos permanecem para compatibilidade
histórica, fora do fluxo padrão. Não são exigências nem pendências deste estudo.

## Condições técnicas antes do full

- Objetivo: **1.000 documentos × 6 braços × 3 modelos = 18.000 resumos**.
  Todos os 1.000 precisam passar no preflight de todos os braços/modelos.
  Se houver incompatibilidade de janela, registrar o caso e bloquear a execução;
  não reduzir silenciosamente o experimento nem substituir casos após a geração.
- Contar prompt completo + reserva de saída + margem, com o tokenizer/contagem
  do endpoint. Nunca truncar silenciosamente instruções, exemplos ou C1 para caber.
  C1t e o orçamento RAG são recortes intencionais do desenho, não overflow.
- A preparação local, a contagem remota Qwen e a exportação estão integradas
  por `run_experiment_openrouter.py`. O caminho é validado no dev10; os 1.000
  casos ainda precisam de preparação e preflight próprios.
- Retry limitado para erros transitórios, esperas 1/2/4/8/16 segundos,
  respeitando Retry-After e limite de requisições; salvar estado para retomada.
  Não repetir por baixa qualidade. Falhas persistentes deixam a rodada incompleta.
- Reservar 8.192 tokens de saída, mantendo a orientação de até 750 palavras.
  A janela válida evita overflow de entrada, mas não garante ausência de
  `length`. Preservar, sinalizar e avaliar essas respostas como produzidas;
  não descartá-las nem repetir a geração para melhorar o resultado.
- Validar offline e no caminho de dev a integração de ROUGE, BERTScore e
  comparações. Congelar versões, hashes, seleção, configuração e orçamento.
  Métricas e embeddings locais; geração pelo OpenRouter.

A avaliação usa empresas reservadas **dentro do FINDSum**, sem demonstrar
transferência para outro corpus. Nenhuma chamada de geração ou avaliação final
foi executada durante este realinhamento.

## Demonstração real posterior — um documento dev

`outputs/exemplo-completo-20260926/`: GOOD-0001683168-21-001122 sorteado
com semente 42, seis braços Ling free/Novita, 6/6 stop, custo reportado zero.
Fonte corrigida com tabelas; 43 chunks, sete de tabela. Entrada total130.086
tokens, saída3.436; reserva8.192 por braço. ROUGE e BERTScore executados.

A auditoria mostrou que o BERTScore padrão leu somente512 dos1.059 tokens da
referência e também truncou as previsões de C2/C5. ROUGE usa os textos completos.
Não apresentar esses BERTScores como avaliação integral de textos longos.
A política foi substituída pela cobertura integral com XLNet no dev10, sem confundir com
truncamento de geração, que não ocorreu neste caso. A demonstração não testa
hipóteses inferenciais com n=1. Ver [caso completo](exemplo-completo.html).

## Implementação posterior: dev10 integral

A decisão do usuário mantém até 750 palavras nos prompts. O novo executor
integra a coorte comum e a exportação dos três modelos, com retries e retomada.
O BERTScore mudou para XLNet com auditoria de todos os IDs de tokens antes
de pontuar; o exemplo antigo com RoBERTa permanece histórico. Respostas por
limite de saída são sinalizadas e avaliadas como produzidas. A avaliação final
não foi executada. Ver [dev10](dev10-openrouter.md) e a [página de leitura](dev10.html).


Dev10 concluído: 180 respostas, ROUGE e BERTScore nos 180 pares, sem corte na
métrica. Gemma/TNET/C3 terminou por `length`, preservada e sinalizada; as outras
179 terminaram por `stop`. Custo incluindo calibração US$ 0,205239238.
O caminho integrado passou; preparação e orçamento dos 1.000 continuam antes
do full, que não foi executado.


## Métricas complementares posteriores ao dev10

A pedido do usuário, adicionadas precisão e recall do BERTScore e METEOR
(NLTK/Treebank/Porter/WordNet em inglês). Escopo descritivo; primárias e família
de testes permanecem inalteradas. Os 180 resumos salvos foram reavaliados sem
novas chamadas API, com F1 idêntico. Ver dev10-metricas-complementares.md.


## Congelamento do full — 26/09/2026

Pipeline de geração consolidada exclusivamente no OpenRouter. Backend local e perfis antigos removidos; processamento, embeddings e métricas continuam locais. Coorte, configurações, código e recursos locais congelados; não houve execução do full nem calibração remota nesta etapa. Guia operacional: [full congelado](full-congelado.md); estimativa atual com todo Ling pago: [orçamento](orcamento-full.md). O preflight dos 18.000 prompts continua pendente de preparação e contagem remota Qwen.

## Execução em lotes — 27/09/2026

Ling pode ser executado separadamente, em dez lotes fixos de 100 da mesma coorte de avaliação. Todos os 6.000 prompts Ling são pré-validados antes da primeira chamada; cada invocação executa no máximo um lote. Não se selecionam casos pelo resultado nem se alteram prompts entre lotes. Não são feitas decisões inferenciais a cada 100; a análise final do modelo usa os 1.000 pares por braço. Qwen/Gemma permanecem na mesma coorte, com preflight próprio antes de gerar. Guia: [lotes Ling](lotes-ling-full.md).

A validação real anterior à primeira geração encontrou um defeito no reconhecimento de tabelas: ` | ` em prosa era tratado como delimitador de célula, impedindo C1t/C2 em 11 documentos. A correção usa os marcadores explícitos de serialização; não altera fonte, recuperação, seleção de exemplos, prompts de instrução ou métricas. Não houve exclusão ou substituição de casos. Preflight inicial preservado e novos testes de regressão adicionados.
