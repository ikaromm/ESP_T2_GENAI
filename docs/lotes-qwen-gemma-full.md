# Qwen e Gemma: lotes independentes de 100

Cada modelo usa os mesmos 1.000 documentos e os mesmos dez grupos do Ling, na ordem de `configs/full-eval-cohort.csv`. Um comando executa no máximo 100 documentos × seis braços = 600 gerações, sem contar retries. Não há novo sorteio, substituição de documentos ou chamadas ao Ling nesses comandos.

## Preparação

Execute na raiz do repositório. As pastas de `outputs/` são locais e não acompanham um clone.

Gemma usa o tokenizer oficial local. Para criar sua preparação, se a pasta ainda não existir:

```bash
uv run --locked findsum prepare-gemma
```

Esse comando reutiliza fontes, referências, candidatos recuperados e IDs dos exemplos de `outputs/full-ling-prepared`. Não refaz embeddings nem envia chamadas à API. Recalcula os contextos e os prompts com o tokenizer do Gemma; exige os 6.000 prompts válidos, sem descartar casos.

Qwen também é preparado localmente, sem chamadas extras:

```bash
uv run --locked python scripts/prepare_gemma_from_common.py \
  --model qwen37 --source outputs/full-ling-prepared \
  --output outputs/full-qwen-prepared
```

O tokenizer `Qwen/Qwen3.8-27B`, com revisão e hashes fixados em `scripts/qwen_local_tokenizer.py`, reproduziu exatamente as 332 contagens de prompts Qwen3.7 Flash já registradas no dev. É compatibilidade empírica, não declaração oficial de identidade dos tokenizers. Os 6.000 prompts são montados e contados na CPU, sem pesos LLM, preservando C1t/C2 e o mesmo RAG em C2–C5. A pasta pronta é reutilizada nos lotes; hashes e contagens são revalidados localmente antes da execução. O `usage.prompt_tokens` de cada geração real ainda precisa coincidir; divergência interrompe para auditoria, sem recalibrar automaticamente.

`findsum calibrate` permanece apenas para auditar/reproduzir preparações remotas históricas. Não faz parte do Bash atual. Não sobrescreva preparações existentes ou misture ledger de geração de uma preparação com outra.

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

O limite dos comandos de lote pagos é **500 requisições/minuto por modelo**, incluindo geração e retries, com redução adaptativa diante de erros. O Ling gratuito mantém 20/minuto. As chamadas pagas são concorrentes, até 100 em voo por modelo; o ritmo e a concorrência caem diante de falhas transitórias. O teto não garante 500 conclusões por minuto. Os timestamps da janela são preservados nas retomadas. Evite rodadas simultâneas na mesma conta se precisar respeitar esse teto de forma agregada. Retries transitórios: até seis tentativas, esperas 1/2/4/8/16 segundos e respeito a `Retry-After`. Após esgotar tentativas, o executor de lotes permite nova rodada depois de uma hora. Falha de transporte com resultado desconhecido exige auditoria, para evitar cobrança/geração duplicada.

Cada diretório de resultados tem `batch-status.json` e uma subpasta `qwen37/` ou `gemma26/` com `ledger.json`, requests e respostas. O status dos lotes é atualizado ao encerrar a invocação; o ledger é atualizado a cada tentativa. Métricas locais e testes das cinco hipóteses são calculados após os 1.000 documentos daquele modelo.

## Compatibilidade com o Ling já iniciado

A atualização operacional preserva os artefatos e a identidade da preparação anterior do Ling. O lock anterior fica arquivado; o novo lock aceita explicitamente essa preparação somente se coorte, dados, configuração, prompts, recuperação, métricas e recursos científicos mantiverem seus hashes. A identidade do ledger continua vinculada ao lock da preparação, evitando invalidar respostas já aceitas. O processo antigo não recebe os novos logs em tempo real; eles aparecem na próxima execução.


## Validação local em 27/09/2026

Preparação Gemma concluída em `outputs/full-gemma-prepared`: 1.000 documentos, 6.000 prompts, zero erros, maior entrada de 43.088 tokens e total de 132.687.868 tokens de entrada. A montagem paralela dos prompts levou 105,7 segundos, além da leitura, conferência e gravação dos artefatos. Nenhuma chamada de geração foi feita pelo assistente.

O cenário extremo de todas as 6.000 respostas Gemma atingirem 8.192 tokens soma US$ 16,3863, sem retries, com essas entradas. O teto cumulativo do executor continua US$ 9, como congelado; ele interrompe antes de ultrapassá-lo, sem garantir que o saldo baste para concluir. Esse cenário extremo não é o custo esperado baseado nas respostas do dev.

A conferência da preparação comum para Qwen (`calibrate` sem `--execute`) passou. As sondas pagas e a geração Qwen não foram iniciadas nesta atualização. Os 270 testes locais passaram, excluindo `slow` e `test_real_data.py`.

O comando real `uv run --locked findsum gemma-batch` também terminou com código 0, sem `--execute`: 6.000 prompts conferidos, `next_batch=1`, 600 chamadas pendentes, nenhuma API. A etapa de integridade e recontagem levou 80,8 segundos.

Em 27/09/2026, o teto pago foi elevado para 100/minuto por solicitação do usuário. É uma política local: o OpenRouter não impõe o teto gratuito de 20/minuto aos modelos pagos, mas provedores e saldo para requisições em andamento podem limitar o uso. Fontes: [suporte OpenRouter](https://openrouter.zendesk.com/hc/en-us/articles/39501163636379-OpenRouter-Rate-Limits-What-You-Need-to-Know) e [limites da API](https://openrouter.ai/docs/api_reference/limits).


## Preparação Qwen local verificada — 27/09/2026

- Artefatos: `outputs/full-qwen-prepared/`, incluindo prompts, tokenizer, contagens, IDs de exemplos e hashes.
- Coorte integral preservada: 1.000 documentos, seis braços, 6.000 prompts válidos, zero erros.
- Maior entrada: 43.338 tokens, abaixo do limite 49.152; saída reservada de 8.192 e margem de contexto 256.
- Soma de entradas dos 6.000 prompts: 133.186.403 tokens.
- Montagem local em quatro trabalhadores: 104,1 segundos, além de carregamento, verificações e gravação.
- Nenhuma chamada de geração ou calibração. Contagens e textos preparados; não são resultados do experimento.
- Lock da preparação local antes do retry adaptativo: `5b14fd8a82480f12ff3765f487135e55ad5ee742ed7d14a3efda48f0a1086f65`; lock anterior arquivado. Preparações Ling e Gemma permanecem compatíveis e respostas Ling não são alteradas.
- 289 testes passaram, excluindo slow e test_real_data.py; Ruff e diff check passaram.

O custo máximo do relatório usa os tetos de preço e todas as saídas no máximo: não é a estimativa média nem uma cobrança. O teto de geração Qwen permanece US$ 18 e continua interrompendo a execução se necessário.

Validação real `bash rodar_rodada.sh 100 --dry-run` passou nos três modelos: Ling 0, Gemma 600 e Qwen 600 gerações pendentes; nenhuma API, rodada ativa não criada. Após ajustes apenas de mensagens de log e compatibilidade do congelamento, os testes direcionados e a verificação dos três locks foram repetidos com sucesso.

Política atual dos lotes e cancelamento: [retry adaptativo](retry-adaptativo-openrouter.md). O teto 100 descrito nas verificações históricas foi substituído por 500 nos lotes pagos, por decisão do usuário após o teste de vazão.
