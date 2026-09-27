# Dev10 — métricas complementares

Adição solicitada após inspeção do dev: precisão/recall do BERTScore e METEOR.
São descritivas. BERTScore F1 e ROUGE-L continuam primárias, com os mesmos
cinco contrastes e Holm sobre dez testes por modelo. Não procurar significância
com novas métricas; nenhum teste inferencial novo foi adicionado.

180 pares reavaliados localmente, sem chamadas de geração e sem novo custo API.
Fontes e escores originais preservados em run/; nova exportação em metrics-extended/.
Maior diferença entre F1 original e recalculado: 0.0. Cobertura integral auditada.
Cada média abaixo tem dez documentos; a média geral dá peso igual aos seis braços.

BERTScore: xlnet-base-cased, camada5, sem IDF/reescala, batch1.
METEOR: NLTK3.10.3, TreebankWordTokenizer, lowercase, PorterStemmer e WordNet3.0
em inglês; alpha0.9, beta3.0, gamma0.5. Texto inteiro, uma referência por caso.
Não confundir esta implementação com outras variantes parametrizadas do METEOR.
Precisão/recall são alinhamentos semânticos de tokens, não taxas de acerto factual
nem porcentagens literais de fatos cobertos. F1 é calculado por caso e depois
agregado; não é o harmônico das médias de P/R.

| Modelo | Braço | BERT P | BERT R | BERT F1 | METEOR |
|---|---|---:|---:|---:|---:|
| Ling | C1 | 0.7092 | 0.5847 | 0.6402 | 0.1551 |
| Ling | C1t | 0.6793 | 0.5910 | 0.6313 | 0.1461 |
| Ling | C2 | 0.7301 | 0.5960 | 0.6556 | 0.1652 |
| Ling | C3 | 0.6855 | 0.5748 | 0.6246 | 0.1524 |
| Ling | C4 | 0.7070 | 0.5806 | 0.6366 | 0.1568 |
| Ling | C5 | 0.7246 | 0.5876 | 0.6482 | 0.1556 |
| Qwen | C1 | 0.7485 | 0.6180 | 0.6763 | 0.1478 |
| Qwen | C1t | 0.7142 | 0.6152 | 0.6594 | 0.1488 |
| Qwen | C2 | 0.7215 | 0.5902 | 0.6486 | 0.1542 |
| Qwen | C3 | 0.7377 | 0.5772 | 0.6467 | 0.1088 |
| Qwen | C4 | 0.7506 | 0.5847 | 0.6569 | 0.1239 |
| Qwen | C5 | 0.7743 | 0.6088 | 0.6809 | 0.1478 |
| Gemma | C1 | 0.7270 | 0.5755 | 0.6407 | 0.1123 |
| Gemma | C1t | 0.7045 | 0.5436 | 0.6113 | 0.0890 |
| Gemma | C2 | 0.7079 | 0.5605 | 0.6254 | 0.1062 |
| Gemma | C3 | 0.6871 | 0.5188 | 0.5895 | 0.0782 |
| Gemma | C4 | 0.7208 | 0.5224 | 0.6048 | 0.0863 |
| Gemma | C5 | 0.7267 | 0.5602 | 0.6321 | 0.1035 |

Médias gerais:

| Modelo | BERT P | BERT R | BERT F1 | METEOR |
|---|---:|---:|---:|---:|
| Ling | 0.7060 | 0.5858 | 0.6394 | 0.1552 |
| Qwen | 0.7411 | 0.5990 | 0.6615 | 0.1386 |
| Gemma | 0.7123 | 0.5468 | 0.6173 | 0.0959 |

## Observações descritivas

- Ling lidera METEOR geral (0.1552), Qwen lidera BERTScore P/R/F1.
- Gemma tem precisão média0.7123, acima de Ling0.7060, mas recall0.5468,
  abaixo de Ling0.5858. Sua desvantagem em F1 acompanha menor recall; isso não
  prova a causa nem equivale a uma contagem manual de informações omitidas.
- Qwen C3 versus C2: precisão0.7215→0.7377, recall0.5902→0.5772;
  F1 quase estável0.6486→0.6467; METEOR0.1542→0.1088. A extensão média
  também cai404.2→232.6 palavras. Padrão compatível com menor cobertura,
  sem estabelecer causalidade entre comprimento e escore.
- Qwen C5 versus C4 melhora P/R/F1 e METEOR. Gemma C5 versus C4 também,
  com mudança maior em recall (+0.0377) do que precisão (+0.0059).
- RAG versus fonte inteira no Qwen: BERTScore cai, METEOR sobe;
  a conclusão depende de qual aspecto do alinhamento é medido.
- Gemma/TNET/C3 terminou por length e continua incluído. Não houve exclusão
  por escore, nova geração ou validação factual. Dez documentos não estabelecem
  superioridade entre modelos; hipóteses primárias continuam não confirmadas.

## Reprodução

`uv run --no-sync python -m nltk.downloader wordnet` prepara o recurso local.
`uv run --no-sync python scripts/extend_reference_metrics.py --root outputs/dev10-complete-20260926`
exige que metrics-extended/ ainda não exista, para preservar os resultados.
`uv run --no-sync python scripts/build_dev_review.py --root outputs/dev10-complete-20260926`
reconstrói a página com a extensão automaticamente.
O executor integrado de futuras rodadas também grava as métricas complementares.
Nenhuma rodada futura foi executada nesta alteração.

Fontes: scores.csv, report.json e validation.json em metrics-extended/;
validation registra hashes dos escores e respostas originais, versões e auditoria.
Métodos: [BERTScore](https://github.com/Tiiiger/bert_score) e
[METEOR NLTK](https://www.nltk.org/api/nltk.translate.meteor_score.html).
