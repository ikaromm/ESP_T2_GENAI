# Triagem gratuita de Ling, Qwen e Gemma — 2026-09-26

Correções posteriores de recuperação, exemplos e tabelas e suas novas rodadas:
[correcoes-pre-dev.md](correcoes-pre-dev.md). Os resultados abaixo preservam o
histórico das configurações anteriores.

Ling concluiu os 18 casos técnicos. Na retomada com três modelos, Qwen e Gemma
continuaram bloqueados por HTTP 429 dos provedores após seis tentativas cada.
A qualidade factual do Ling não está aprovada: uma inspeção exploratória detectou
transferência de valores de uma demonstração para a empresa alvo. Não há dados
para escolher um vencedor entre os dois modelos ou confirmar H1–H4.

## Escopo e execução

Autorização: testar os dois candidatos em três documentos dev, seis braços por
modelo. Documentos: ABEO-0001493152-21-006705, ABIO-0001564590-21-014148 e
ABT-0001047469-19-000624. Pool reduzido de 32 exemplos, quatro exemplos em C3–C5;
mesmos documentos, exemplos selecionados e parâmetros de recuperação entre
modelos. As bordas dos contextos diferem conforme o tokenizer de cada modelo.
O conjunto eval e o manifesto não foram alterados. Não é avaliação final.

| Modelo | Provedor fixado | Casos concluídos | Resultado |
|---|---|---:|---|
| inclusionai/ling-3.0-flash-fin:free | novita | 18/18 | Todos `stop`, contagens locais/API coincidentes |
| google/gemma-4-31b-it:free | google-ai-studio | 0/18 | Primeiro caso: HTTP 429, `upstream_provider_shared_pool` |

Tokenizers oficiais: `inclusionAI/Ling-3.0-flash-Fin` e
`google/gemma-4-31B-it`, ambos carregados sem pesos da LLM e sem executar código
remoto. Temperatura zero, top_p=1, reasoning desativado, seed não enviada,
max_new_tokens=1536 e max_input_tokens=16384. Preço máximo zero e sem fallback.
Os dois provedores declaram seed, mas ela foi omitida para manter a configuração
de geração da triagem anterior; determinismo remoto não foi demonstrado.

Ambos passaram na pré-validação dos 18 prompts: C1 recebe a fonte disponível
inteira; C1t/C2 têm a mesma contagem de contexto e prompt; C2–C5 compartilham o
contexto; quatro exemplos em cada braço few-shot; nenhum overflow.

Ling: 136.852 tokens de entrada, 8.775 de saída, custo reportado US$ 0,
zero tokens de reasoning, soma de 45,509 segundos nas chamadas da API e mediana
de 2,635 segundos. Esses tempos não incluem preparação local e intervalos.
Na rodada inicial, Gemma não produziu resposta textual; não há medidas de qualidade ou uso de tokens
de geração para ele. Não foram feitas novas tentativas automáticas. Ao final,
a conta indicava 19 chamadas usadas e 31 restantes.

## Achados exploratórios de qualidade

1. **Contaminação pelos exemplos, ABIO/C5/Ling.** O resumo incluiu compras de
   títulos de US$ 268,3 milhões e emissão de ações de US$ 344,3 milhões. Os valores
   e as descrições correspondentes aparecem no resumo do segundo exemplo,
   ADVM-0001628280-21-003623. Não aparecem na fonte nem no contexto de ABIO.
   A menção ao empréstimo BPI France também foi transferida desse exemplo.
   Evidências e trechos estão em `validation.json`. Isso demonstra um problema
   concreto nessa saída, sem constituir revisão humana formal de todos os casos.
2. **Contexto RAG pouco alinhado à tarefa, ABEO.** O contexto recuperado contém
   discussões de reconhecimento de receita/ASC 606 e contratos com Taysha. O Ling
   informa em C2–C5 que não há material para resumir fluxos de caixa e liquidez.
   A seleção usa similaridade com o documento inteiro; a hipótese a investigar
   é que temas dominantes do documento deslocam os trechos da tarefa Liquidity.
   Não se alterou a recuperação durante a rodada para favorecer um resultado.
3. **Unidade monetária a revisar, ABEO/C1.** A saída atribuiu US$ 199 milhões à
   aquisição de tecnologia; a fonte serializada contém `199 & 199,000 (2019)`.
   A magnitude emitida não está justificada por essa representação. É necessário
   revisar unidades e associação das tabelas antes de atribuir toda a falha ao
   modelo ou usar a fonte serializada como verdade inequívoca.

## Métricas descritivas do Ling

Médias de apenas três documentos, sem testes de significância:

| Braço | ROUGE-L | F1 numérico contra referência |
|---|---:|---:|
| C1 | 0,162 | 0,362 |
| C1t | 0,138 | 0,287 |
| C2 | 0,087 | 0,117 |
| C3 | 0,109 | 0,195 |
| C4 | 0,132 | 0,266 |
| C5 | 0,156 | 0,251 |

Essas métricas não medem correção factual. Em particular, presença de números
no contexto não garante unidade, entidade, período ou relação correta.
BERTScore não foi calculado nesta triagem. As fichas `ling/human-review.json`
contêm 18 casos sem rótulos de braço e sem julgamentos preenchidos; revisão
humana e segmentação de afirmações seguem pendentes.

## Artefatos e reprodução

Diretório ignorado pelo git: `outputs/screen-ling-gemma-20260926/`.

- `report.json`: status de cada caso, modelos e cotas observadas.
- `validation.json`: conferência independente de artefatos e evidências da contaminação.
- `ling/` e `gemma/`: configuração, 18 prompts em `preflight.json`, pedidos,
  respostas/erros em `api/`, previsões e métricas dos casos aceitos.
- `ling/human-review.json` e `.map.json`: revisão humana pendente e mapa separado.

```bash
uv run --no-sync --env-file ../.env python scripts/screen_openrouter.py \
  --output outputs/screen-ling-gemma-NOVO
```

Sem `--resume`, o script exige diretório novo. Com `--resume`, confere configuração,
pré-validação e resultados salvos antes de reaproveitar casos concluídos. Os três
modelos gratuitos autorizados estão em `remote_models.py`; `--models` permite
selecionar `ling qwen gemma`. Um erro bloqueia apenas o modelo afetado.

## Retomada dos três modelos e limites

Em 2026-09-26, a pedido do usuário, a triagem passou a incluir Qwen3.8 27B gratuito.
Os 18 prompts do Qwen também passaram na pré-validação, sem gerar novos resumos.
A retomada preservou os 18 resultados do Ling sem chamadas adicionais.
Qwen/ModelRun e Gemma/Google AI Studio receberam `429` com
`limit_source=upstream_provider_shared_pool`, sem headers de limite, em todas
as seis tentativas da rodada com esperas curtas. Antes disso, duas tentativas
adicionais do Qwen foram registradas numa retomada interrompida durante a espera
para aplicar os novos intervalos. Os erros anteriores continuam no histórico.

```bash
uv run --no-sync --env-file ../.env python scripts/screen_openrouter.py \
  --output outputs/screen-ling-gemma-20260926 --resume
```

Política aplicada somente no script de triagem:

- Até seis tentativas por caso: primeira chamada e cinco repetições de HTTP 429,
  com esperas de 1, 2, 4, 8 e 16 segundos.
- `Retry-After` prevalece se maior; acima de 60 segundos, interrompe o modelo
  nesta rodada para evitar repetir antes do prazo. Outros erros não são repetidos.
- Janela móvel persistida em `request-window.json`: no máximo 20 chamadas de
  geração em 60 segundos, contando repetições e todos os modelos desta execução.
  O limitador pode prolongar as esperas. Não coordena outros processos/clientes.
- São requisições, não documentos: seis braços exigem seis chamadas por documento
  e modelo; três modelos exigem 18, antes de qualquer repetição.

Consulta real após a rodada, às 13:59 UTC: limite diário 1.000, usadas 19,
restantes 981. Esse contador não permite concluir que erros nunca consomem cota.
Ter cota diária disponível não libera capacidade no provedor. Referência:
[limites do OpenRouter](https://openrouter.ai/docs/api_reference/limits).

`continuation-validation.json` registra a consulta e a verificação SHA-256 de
26 arquivos preservados, incluindo previsões/respostas do Ling, ficha de revisão
e manifesto. Snapshot anterior em `history/20260926T135354/`. A validação inicial
em `validation.json` permanece como evidência daquela rodada. Nenhuma chamada
paga, avaliação final ou aprovação factual foi realizada nesta retomada.

Próximo trabalho recomendado: revisar relevância do contexto e separação factual
entre demonstrações e relatório alvo no dev, validar unidades das tabelas e
realizar revisão humana. Gemma precisa completar a geração para haver comparação
entre modelos. Nenhum modelo foi fechado e nenhuma hipótese foi confirmada.
