# Métricas e painel automáticos

```bash
bash rodar_rodada.sh 100             # gera/retoma, depois atualiza métricas
bash rodar_rodada.sh --metrics-only  # atualiza somente métricas e painel
bash rodar_rodada.sh 100 --dry-run   # valida sem geração nem cálculo de métricas
```

O modo de métricas é local: não consulta a cota, não envia prompts, não abre uma nova rodada e não muda a seleção de documentos. Ele confere o congelamento, os prompts e as respostas salvas antes de calcular. As primeiras conferências podem levar alguns minutos mesmo quando todas as métricas estão em cache.

## Base comparável

A matriz inclui somente o prefixo consecutivo da ordem congelada que possui os seis braços aceitos em todos os três modelos. Se um modelo completou 200 documentos e outro só 100, as médias comparativas continuam nos mesmos 100. Se há uma lacuna no documento 80, a matriz usa os primeiros 79; não procura casos posteriores que deram certo para substituir a lacuna.

Os cartões mostram separadamente respostas aceitas, documentos completos e saídas `length` de cada modelo, em toda a coorte. Portanto, o progresso individual pode ser maior que a quantidade comparável. Saídas `length` são mantidas nas métricas conforme protocolo.

## Método e reaproveitamento

BERTScore precisão/recall/F1 com XLNet-base-cased, camada 5, sem IDF/rescale e sem truncamento; ROUGE-1/2/L F1 e METEOR usam as funções existentes do projeto. Não há métrica numérica nova. Os gráficos destacam BERTScore F1, ROUGE-L e METEOR; a tabela inclui precisão e recall.

O cache fica em `outputs/progress-metrics/cache.json`, ignorado pelo Git. A chave inclui documento, modelo, braço, texto gerado, referência e motivo de término. A identidade do método inclui hashes do código de métricas/limpeza/atualizador e `uv.lock`. Se o método mudar, o cache é invalidado; se um texto mudar, aquele par é recalculado. Cada bloco de 120 pares é salvo para permitir retomada após uma interrupção.

A evolução é cumulativa, com pontos a cada 100 documentos e no prefixo atual, reconstruída dos scores disponíveis. Um ponto de 200 documentos inclui os primeiros 100; os pontos não são amostras independentes. O snapshot histórico de `results/first100-20260927/` é preservado.

## Artefatos para o Git

- `results/progress/README.md`: painel e matriz por braço/modelo.
- `dashboard.png` e `dashboard.svg`: gráficos estáticos renderizados pelo GitHub, sem serviço externo.
- `bertscore-f1`, `rouge-l-f1`, `meteor` e `evolucao` em PNG/SVG: figuras individuais em escala ampliada.
- `scores.csv`: valores individuais sem textos de entrada ou saída.
- `summary.json`: médias, cobertura, IDs, método, hashes dos ledgers e auditoria de tokens.
- `history.json`: pontos cumulativos usados no gráfico.

O Bash calcula os pares ausentes no cache uma única vez. O término da geração de cada modelo não dispara uma segunda pontuação integral dos 6.000 pares. Aos 1.000 documentos completos, ele aplica o plano congelado diretamente ao CSV versionado e grava `comparisons.csv`, `comparisons.json` e `analysis.json`: 30 testes, com Holm em dez testes por modelo, sem repontuar os textos. Matrizes incompletas, duplicadas ou com métricas primárias não finitas são rejeitadas.

Depois, `scripts.evaluation.render_readable_progress` usa as médias e o histórico para criar o painel 2×2, figuras PNG/SVG individuais com eixos ampliados e atualizar os dois READMEs. O link para os testes completos depende dos hashes do CSV e das comparações registrados em `analysis.json`. O código científico de pontuação e o identificador do cache não mudam. Os eixos ampliados são identificados em cada figura e permitem ler diferenças pequenas; a tabela preserva os valores exatos. Para refazer somente a apresentação, sem validar os 6.000 prompts de cada modelo: `uv run --locked python -m scripts.evaluation.render_readable_progress`.

O README principal referencia a imagem atualizada. O Bash prepara os arquivos; **não executa commit/push automaticamente**. Para publicar apenas a nova medição:

```bash
git add results/progress README.md
git commit -m "results: atualiza acompanhamento do experimento"
git push origin main
```

## Falhas e interpretação

Se uma geração falhar, os outros modelos continuam conforme a política existente; ao encerrar a tentativa, o painel usa o prefixo completo disponível. O código de falha da rodada continua diferente de zero. Falhas no cálculo não apagam as gerações; rode `--metrics-only` após corrigir a causa. O painel anterior deve ser considerado desatualizado até essa atualização concluir.

Textos muito longos podem exceder a memória da GPU no BERTScore (ocorreu com 11,6 GiB nas rodadas 161–320 e 321–520). Os blocos já salvos permanecem no cache; retome em CPU, sem geração: `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 bash rodar_rodada.sh --metrics-only`. O modelo, a camada e a configuração são os mesmos; muda só o dispositivo, e podem existir diferenças mínimas de ponto flutuante entre CPU e GPU.

As médias e figuras deste painel são descritivas. Os p-valores ficam na tabela confirmatória separada, gerada pelo Bash somente com os 1.000 documentos completos nos três modelos. Essa análise usa BERTScore F1 e ROUGE-L F1; precisão/recall, ROUGE-1/2 e METEOR continuam descritivos. H3 mede sensibilidade e não demonstra equivalência quando não significativa. O fluxo não modifica prompts ou RAG a partir dos resultados de eval.

## Validação desta implementação

Em 27/09/2026, `bash rodar_rodada.sh --metrics-only` terminou com os 100 documentos atuais e 1.800 pares. As métricas por documento e as médias reproduziram o snapshot publicado (diferença máxima nas médias inferior a `1e-15`). Os hashes dos três ledgers continuaram iguais, confirmando que a atualização não gerou respostas.

Uma segunda atualização do mesmo conjunto reutilizou o cache integral, com zero recálculos e todos os arquivos do painel idênticos byte a byte. PNG inspecionado visualmente. Suíte: 308 testes aprovados, excluindo `slow` e `test_real_data.py`; Ruff, sintaxe Bash e verificação de whitespace aprovados.

Em 29/09/2026 (conclusão em 30/09 UTC), a coorte de 1.000 documentos terminou com 18.000 pares únicos. A atualização incremental preservou os 9.360 scores publicados aos 520 documentos; reaproveitou os 2.880 novos scores do Ling já calculados com o mesmo método e calculou somente os 5.760 pares novos de Qwen/Gemma. O identificador científico do cache permaneceu inalterado. A auditoria confirmou todos os hashes dos requests e respostas, a igualdade C1t/C2 de orçamento, o mesmo contexto em C2–C5 e cobertura integral de tokens nas métricas. Os 30 testes finais estão em `results/progress/comparisons.csv`; custos, proveniência e verificações em [results/audit1000](../results/audit1000/README.md). Foram aprovados 338 testes sem `slow`/`test_real_data.py` e 20 testes direcionados após a correção de exportação CSV; Ruff, whitespace e lock também passaram.
