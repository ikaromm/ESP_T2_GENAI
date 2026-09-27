# PoC OpenRouter — 2026-09-26

Correções posteriores de recuperação, exemplos e tabelas e suas novas rodadas:
[correcoes-pre-dev.md](correcoes-pre-dev.md). Os resultados abaixo preservam o
histórico das configurações anteriores.

Estado atual: backend integrado a `findsum run`. O Qwen gratuito continua bloqueado
por 429. A [triagem posterior de Ling, Qwen e Gemma](triagem-ling-gemma.md) concluiu os
18 casos do Ling, incluindo RAG/few-shot via API; Qwen e Gemma continuaram com
429 após a retomada com seis tentativas, backoff e limite de 20 chamadas/minuto. A execução técnica do Ling funcionou, mas houve contaminação factual
pelos exemplos e problemas de relevância do contexto recuperado.

Resultado da PoC inicial: integração parcialmente validada. Uma geração real funcionou;
requisição com quatro exemplos recebeu HTTP 429 em duas tentativas.
Isso confirma autenticação/geração gratuita, mas não uma rodada completa.

- Modelo solicitado: `qwen/qwen3.8-27b:free`.
- Provedor fixado: `modelrun/fp4`; resposta identificou `ModelRun`.
- Sem fallback, preços máximos de entrada/saída zero, reasoning desativado.
- Temperatura 0; `seed` não enviada porque não consta nos parâmetros suportados.
- Um documento do **dev**: `ABEO-0001493152-21-006705`.
- Quatro exemplos fixos do conjunto `examples` no segundo prompt.
- O conjunto `eval` não foi carregado. Nenhuma mudança no modelo padrão do experimento.

| Caso | Entrada | Saída | Tempo | Resultado |
|---|---:|---:|---:|---|
| Fonte inteira, zero-shot | 9.041 tokens | 579 tokens | 17,066 s | `stop`, custo reportado US$ 0 |
| Mesma fonte + quatro exemplos fixos | não informado pela API | — | — | HTTP 429, inclusive na repetição |

A primeira resposta informou zero tokens de raciocínio. O contador consultado
antes da repetição ainda mostrava 0 usadas / 50 restantes; portanto não foi
possível atribuir o 429 à cota diária nem medir o consumo de cota apenas pelo
contador. Não foi capturado o corpo do erro, então a origem exata do limite
(OpenRouter ou provedor) permanece indeterminada. Não houve troca para modelo pago.

A leitura exploratória encontrou uma inconsistência: o resumo fala em redução
das despesas administrativas e simultaneamente informa aumento de US$ 3,1 milhões;
a fonte informa aumento de 20,7 para 23,8 milhões. Geração bem-sucedida não comprova
fidelidade factual. Não foram calculados resultados de H1–H4 nem BERTScore nesta PoC.

## Artefatos locais

`outputs/poc-openrouter-20260926/` (ignorado pelo git):

- `report.json`: resposta, uso, custo, referência e falhas observadas.
- `full_zero_shot.txt`: resumo gerado.
- `*.request.json`: prompts exatos das duas condições, sem chave de API.

O segundo caso usa fonte inteira com exemplos, **não C3**, que usaria contexto
recuperado. Esta PoC testa o transporte e formatos de prompt; RAG, seleção
dinâmica, equivalência de tokens e toda a cadeia de ablação ainda não foram
validados via API naquela execução. Naquele momento, a integração ainda não
estava conectada ao comando `findsum run`.

## Reproduzir

Na raiz do repositório, com `OPEN_ROUTER_KEY` exportada ou `--env-file` explícito:

```bash
uv run python scripts/poc_openrouter.py --env-file ../.env --output outputs/poc-openrouter-NOVO
```

O comando faz duas chamadas e grava resultados incrementalmente. Não sobrescreve
uma execução existente. Para repetir somente o caso pendente, depois de aguardar
liberação do serviço:

```bash
uv run python scripts/poc_openrouter.py --env-file ../.env \
  --output outputs/poc-openrouter-20260926 --retry-failed
```

A repetição é explícita, não automática; não reenvia o caso já concluído.
Antes de um piloto de seis braços, implementar tratamento de 429 com intervalo e
retomada, conferir os limites do endpoint e registrar as condições efetivas do modelo.

## Diagnóstico posterior do HTTP 429

Nova tentativa do MESMO prompt, agora capturando metadados com credenciais
removidas, confirmou:

```json
{
  "http_status": 429,
  "message": "Provider returned error",
  "provider_name": "ModelRun",
  "limit_source": "upstream_provider_shared_pool"
}
```

A mensagem do provedor informa limitação temporária upstream. Não retornou
`Retry-After` nem `X-RateLimit-*`. A conta mostrou 1 chamada usada e **49 restantes**
antes dessa tentativa. Portanto, o bloqueio capturado é do pool compartilhado
do ModelRun, não esgotamento da cota diária. O contador antes mostrado como zero
passou a registrar a geração bem-sucedida; a observação anterior não era prova
de isenção. O erro não indica tamanho de prompt ou falta de créditos.

Essa captura confirma a causa da nova tentativa. Os dois erros anteriores não
tinham corpo registrado; a mesma causa é plausível, não comprovada retroativamente.
Comprar créditos aumenta a cota diária, mas não garante liberar esse pool.
O cliente agora preserva os campos diagnósticos e headers de limite, com
redação de credenciais. O prompt com exemplos continua pendente; não foram
feitas novas tentativas naquele estágio. A tentativa posterior está descrita abaixo.

## Integração com o pipeline e nova validação no dev

O backend `generation.backend: openrouter` usa o mesmo pipeline de preparação,
embeddings locais, FAISS e seleção de exemplos. O backend padrão continua `local`.
O modelo remoto é restrito a `qwen/qwen3.8-27b:free`, com o provedor e parâmetros
gratuitos acima. A seed continua disponível para seleção local de exemplos,
mas não é enviada à API. Temperatura zero não garante determinismo do serviço.

`OpenRouterSummarizer` carrega somente o tokenizer `Qwen/Qwen3.8-27B`, com
`enable_thinking=False` no chat template. Rejeita overflow antes de enviar,
contagens locais/API divergentes, modelo/provedor divergente, reasoning reportado
ou saída cujo `finish_reason` não seja `stop`. Salva pedidos e respostas em `api/`
antes da validação da resposta; erros HTTP são gravados com credenciais removidas.
As previsões aceitas incluem uso, tempo, provedor e identificador da geração.

Perfil: `configs/poc_openrouter.yaml`, um documento dev e **pool de 32 exemplos**.
Esse pool reduzido serve apenas ao teste de integração; não é o piloto completo.
Os seis braços usam os mesmos parâmetros de recuperação. `preflight.json` salva
todos os prompts, contextos, exemplos e contagens antes da primeira geração.

```bash
uv run --no-sync --env-file ../.env findsum run \
  --config configs/poc_openrouter.yaml --preflight-only
```

Para gerar, omita `--preflight-only`; para esta PoC pode-se usar `--no-bertscore`.
Use um nome de rodada novo no YAML: diretórios não vazios são recusados, inclusive
os criados por preflight. Não há retomada automática no `findsum run`; o
`--retry-failed` do script antigo continua limitado à PoC original de dois prompts.
Nenhum comando de avaliação final foi executado.

Execução real de 2026-09-26: artefatos em
`outputs/poc-openrouter-rag-20260926/` (nome efetivo preservado no `config.yaml`
da rodada; o perfil reutilizável passou a se chamar `poc_openrouter`).

| Braço | Tokens de contexto | Tokens de prompt locais | Exemplos |
|---|---:|---:|---:|
| C1 | 8858 | 9041 | 0 |
| C1t | 2957 | 3140 | 0 |
| C2 | 2957 | 3140 | 0 |
| C3 | 2957 | 10251 | 4 |
| C4 | 2957 | 10362 | 4 |
| C5 | 2957 | 9919 | 4 |

Verificados: igualdade de tokens C1t/C2, contexto idêntico em C2–C5 e quatro
exemplos em cada braço few-shot. Todos cabem no orçamento de 16384 tokens.
A primeira chamada (C1) recebeu HTTP 429, `upstream_provider_shared_pool`, sem
headers de limite; a execução parou sem retry. Não há respostas novas aceitas,
métricas de geração ou hipóteses testadas. A contagem local de C1 coincide com os
9041 tokens reportados na PoC original, mas a correspondência nos outros braços
ainda depende de respostas reais. Não inferir qualidade factual destes testes.

MCP: login OAuth concluído nesta sessão; `ping`, `get_model` e
`list_model_endpoints` funcionaram. A conexão MCP está validada, separadamente da
disponibilidade do pool de geração. Recursos MCP não são expostos pelo servidor
(`resources/list` retorna Method not found); as ferramentas funcionam.
