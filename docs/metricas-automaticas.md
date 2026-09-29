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
- `scores.csv`: valores individuais sem textos de entrada ou saída.
- `summary.json`: médias, cobertura, IDs, método, hashes dos ledgers e auditoria de tokens.
- `history.json`: pontos cumulativos usados no gráfico.

O README principal referencia a imagem atualizada. O Bash prepara os arquivos; **não executa commit/push automaticamente**. Para publicar apenas a nova medição:

```bash
git add results/progress
git commit -m "results: atualiza acompanhamento do experimento"
git push origin main
```

## Falhas e interpretação

Se uma geração falhar, os outros modelos continuam conforme a política existente; ao encerrar a tentativa, o painel usa o prefixo completo disponível. O código de falha da rodada continua diferente de zero. Falhas no cálculo não apagam as gerações; rode `--metrics-only` após corrigir a causa. O painel anterior deve ser considerado desatualizado até essa atualização concluir.

Textos muito longos podem exceder a memória da GPU no BERTScore (ocorreu com 11,6 GiB nas rodadas 161–320 e 321–520). Os blocos já salvos permanecem no cache; retome em CPU, sem geração: `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 bash rodar_rodada.sh --metrics-only`. O modelo, a camada e a configuração são os mesmos; muda só o dispositivo, e podem existir diferenças mínimas de ponto flutuante entre CPU e GPU.

Este painel é descritivo. Não calcula p-valores, não confirma hipóteses e não modifica prompts ou RAG a partir dos resultados parciais. A rotina confirmatória da pipeline continua separada e condicionada aos 1.000 documentos por modelo.

## Validação desta implementação

Em 27/09/2026, `bash rodar_rodada.sh --metrics-only` terminou com os 100 documentos atuais e 1.800 pares. As métricas por documento e as médias reproduziram o snapshot publicado (diferença máxima nas médias inferior a `1e-15`). Os hashes dos três ledgers continuaram iguais, confirmando que a atualização não gerou respostas.

Uma segunda atualização do mesmo conjunto reutilizou o cache integral, com zero recálculos e todos os arquivos do painel idênticos byte a byte. PNG inspecionado visualmente. Suíte: 308 testes aprovados, excluindo `slow` e `test_real_data.py`; Ruff, sintaxe Bash e verificação de whitespace aprovados.
