# Alinhamento ao planejamento original — 26/09/2026

Documento relido: `Stage 2 - Planejamento - Projeto de Pesquisa - grupo 4.docx`.
SHA-256: `dfd3160f2f687a7f2aecd25541904f7403d95088585a67bcc606dfb15fb973ee`.

Na seção **Estratégia de avaliação**, o documento diz:

> Serão utilizadas métricas automáticas de avaliação de sumarização, como ROUGE e BERTScore, complementadas por métricas ou procedimentos específicos para avaliação de informações numéricas e fidelidade factual, considerando as características do conjunto FINDSum.

| Tema | Planejamento original | Desenho atual |
|---|---|---|
| Métricas textuais | ROUGE e BERTScore | Mantidas; ROUGE-1/2/L e BERTScore F1 |
| Complementos numéricos/factuais | Previstos, sem métrica específica definida | Retirados por decisão expressa do usuário em 26/09 |
| Inferência | Médias, medianas, distribuições; testes quando aplicável | BERTScore/ROUGE-L, cinco contrastes, Wilcoxon e Holm por modelo |
| Hipóteses | Uma hipótese geral de vantagem do RAG + seleção dinâmica | Cinco contrastes H1, H1b, H2, H3, H4; resultados favoráveis não são exigência |
| Configurações | Cinco: zero-shot; RAG; fixos; aleatórios; dinâmicos | Mantidas; C1t acrescentado como controle de orçamento |
| Amostra | Integridade, referência e compatibilidade com infraestrutura | Manifesto existente preservado, 1.000 eval; ordem aleatória rastreável |
| Preparação | Padronização e segmentação quando necessária | Serialização de tabelas brutas, sem validação/correção financeira |
| Exemplos | Pares documento/resumo por similaridade | Banco separado, quatro pares, busca por embeddings do documento |
| Recuperação | Trechos relevantes ao documento de entrada | FAISS no relatório-alvo, consulta fixa da tarefa; escolha operacional |
| Modelo | Aberto, constante entre configurações | Repetir os seis braços separadamente nos três modelos escolhidos |
| Generalização | Documentos não utilizados no desenvolvimento | Empresas reservadas dentro do FINDSum; não transferência externa |

**Desvio corrigido:** o filtro tabular exigia metadados que as células do dev
não traziam e eliminou todas as 9.448 células. Esse filtro não era exigido pelo
planejamento. Foi removido; os dados brutos permanecem sem correção de valores.
O dev anterior é histórico, com fonte diferente; seus resultados não são
recalculados ou apresentados como validação empírica da nova representação.

O documento não especifica F1 numérico, Holm, Wilcoxon, ROUGE-L como endpoint,
quatro exemplos ou 1.000 casos. Esses detalhes vêm da operacionalização do
projeto. A ausência de avaliação numérica/factual é uma redução explícita de
escopo, não uma alegação de que o planejamento nunca a mencionou.

Consulte o [protocolo](protocolo.md) e o [guia interativo](pipeline.html).
