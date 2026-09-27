# Qwen e Gemma: lotes independentes de 100

Cada modelo usa os mesmos 1.000 documentos e os mesmos dez grupos do Ling, na ordem de `configs/full-eval-cohort.csv`. Um comando executa no máximo 100 documentos × seis braços = 600 gerações, sem contar retries. Não há novo sorteio, substituição de documentos ou chamadas ao Ling nesses comandos.

## Preparação

Execute na raiz do repositório. As pastas de `outputs/` são locais e não acompanham um clone.

Gemma usa o tokenizer oficial local. Para criar sua preparação, se a pasta ainda não existir:

```bash
uv run --locked findsum prepare-gemma
```

Esse comando reutiliza fontes, referências, candidatos recuperados e IDs dos exemplos de `outputs/full-ling-prepared`. Não refaz embeddings nem envia chamadas à API. Recalcula os contextos e os prompts com o tokenizer do Gemma; exige os 6.000 prompts válidos, sem descartar casos.

Qwen não possui tokenizer local validado neste projeto. A contagem usa sondas **pagas**, com saída de até um token. Conferir a preparação comum sem enviar sondas:

```bash
uv run --locked findsum calibrate \
  --prepared outputs/full-ling-prepared \
  --output outputs/full-qwen-prepared \
  --budget-usd 18 --allow-eval
```

Para efetivamente preparar o Qwen, acrescente `--execute` ao comando acima. As sondas não são os resumos do experimento e podem exigir muitas chamadas. O teto de US$ 18 cobre toda a calibração, incluindo retomadas, e não garante sua conclusão. Repita com as mesmas pastas para aproveitar as sondas aceitas. Nenhum lote Qwen pode começar antes de validar os 6.000 prompts. O `.env` do workspace é carregado automaticamente apenas ao executar.

## Executar ou retomar

Conferir prompts e progresso sem chamar a API, após a preparação de cada modelo:

```bash
uv run --locked findsum qwen-batch
uv run --locked findsum gemma-batch
```

Executar **um** lote pago ou retomar o primeiro incompleto:

```bash
uv run --locked findsum qwen-batch --execute
uv run --locked findsum gemma-batch --execute
```

Execute o comando do modelo desejado. Cada um encerra ao terminar seu lote. Repita nas próximas sessões para avançar de 100 em 100; `--batch 1` até `--batch 10` seleciona um grupo específico. Respostas aceitas não são geradas novamente.

| Comando | Modelo / provedor fixado | Preparação | Resultados | Teto cumulativo de geração |
|---|---|---|---|---|
| `qwen-batch` | Qwen3.7 Flash / Alibaba | `outputs/full-qwen-prepared` | `outputs/full-qwen-batches` | US$ 18 |
| `gemma-batch` | Gemma 4 26B A4B / Darkbloom | `outputs/full-gemma-prepared` | `outputs/full-gemma-batches` | US$ 9 |

Os tetos valem para **os dez lotes**, não para cada invocação. `--budget-usd` permite um teto menor; valores acima do congelado são rejeitados. O ledger contabiliza custo reportado das respostas aceitas e reserva conservadora das falhas. Mantenha as pastas para que retomadas preservem o orçamento e a identidade da rodada. Não há troca automática de modelo ou provedor. Não use o executor integrado `findsum run` para continuar estas rodadas independentes.

## Progresso, paralelismo e falhas

A validação de hashes e a recontagem/auditoria dos prompts usam quatro trabalhadores locais. Os logs mostram as etapas desde o início, o avanço a cada 100 prompts e um sinal de atividade a cada 15 segundos durante operações bloqueantes. A ordem da coorte e dos prompts é preservada. Paralelismo local não aumenta a concorrência das chamadas à API.

Durante a geração, o terminal mostra documento/braço, tentativa, espera por frequência, tempo de resposta, tokens, custo e motivo de encerramento. Não imprime credenciais nem o texto dos prompts/resumos. Requests e respostas integrais ficam nos artefatos locais.

O limite operacional dos modelos pagos é **100 requisições/minuto por executor**, incluindo geração, sondas de calibração Qwen e retries. O Ling gratuito mantém 20/minuto. As chamadas continuam sequenciais; aumentar o teto não cria concorrência nem garante 100 chamadas efetivas por minuto. Os timestamps da janela são preservados nas retomadas. Evite rodadas simultâneas na mesma conta se precisar respeitar esse teto de forma agregada. Retries transitórios: até seis tentativas, esperas 1/2/4/8/16 segundos e respeito a `Retry-After`. Após esgotar tentativas, o executor de lotes permite nova rodada depois de uma hora. Falha de transporte com resultado desconhecido exige auditoria, para evitar cobrança/geração duplicada.

Cada diretório de resultados tem `batch-status.json` e uma subpasta `qwen37/` ou `gemma26/` com `ledger.json`, requests e respostas. O status dos lotes é atualizado ao encerrar a invocação; o ledger é atualizado a cada tentativa. Métricas locais e testes das cinco hipóteses são calculados após os 1.000 documentos daquele modelo.

## Compatibilidade com o Ling já iniciado

A atualização operacional preserva os artefatos e a identidade da preparação anterior do Ling. O lock anterior fica arquivado; o novo lock aceita explicitamente essa preparação somente se coorte, dados, configuração, prompts, recuperação, métricas e recursos científicos mantiverem seus hashes. A identidade do ledger continua vinculada ao lock da preparação, evitando invalidar respostas já aceitas. O processo antigo não recebe os novos logs em tempo real; eles aparecem na próxima execução.


## Validação local em 27/09/2026

Preparação Gemma concluída em `outputs/full-gemma-prepared`: 1.000 documentos, 6.000 prompts, zero erros, maior entrada de 43.088 tokens e total de 132.687.868 tokens de entrada. A montagem paralela dos prompts levou 105,7 segundos, além da leitura, conferência e gravação dos artefatos. Nenhuma chamada de geração foi feita pelo assistente.

O cenário extremo de todas as 6.000 respostas Gemma atingirem 8.192 tokens soma US$ 16,3863, sem retries, com essas entradas. O teto cumulativo do executor continua US$ 9, como congelado; ele interrompe antes de ultrapassá-lo, sem garantir que o saldo baste para concluir. Esse cenário extremo não é o custo esperado baseado nas respostas do dev.

A conferência da preparação comum para Qwen (`calibrate` sem `--execute`) passou. As sondas pagas e a geração Qwen não foram iniciadas nesta atualização. Os 270 testes locais passaram, excluindo `slow` e `test_real_data.py`.

O comando real `uv run --locked findsum gemma-batch` também terminou com código 0, sem `--execute`: 6.000 prompts conferidos, `next_batch=1`, 600 chamadas pendentes, nenhuma API. A etapa de integridade e recontagem levou 80,8 segundos.

Em 27/09/2026, o teto pago foi elevado para 100/minuto por solicitação do usuário. É uma política local: o OpenRouter não impõe o teto gratuito de 20/minuto aos modelos pagos, mas provedores e saldo para requisições em andamento podem limitar o uso. Fontes: [suporte OpenRouter](https://openrouter.zendesk.com/hc/en-us/articles/39501163636379-OpenRouter-Rate-Limits-What-You-Need-to-Know) e [limites da API](https://openrouter.ai/docs/api_reference/limits).
