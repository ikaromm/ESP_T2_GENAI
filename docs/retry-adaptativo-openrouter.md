# Rodadas paralelas e retry adaptativo — 27/09/2026

Por decisão do usuário após o teste de vazão, Qwen e Gemma usam teto de **500 requisições/minuto por modelo** nos comandos de lote e no Bash. O benchmark observou 429 em 200 e 500 RPM; portanto, 500 é o teto desejado, não uma capacidade sustentada demonstrada. O Ling gratuito mantém 20 RPM e a consulta de cota diária.

```bash
bash rodar_rodada.sh 100 --dry-run  # conferência local
bash rodar_rodada.sh 100            # executa/retoma o mesmo grupo nos três modelos
```

Os três modelos rodam simultaneamente. Dentro dos pagos, até 100 requisições podem ficar em voo por modelo; todas passam pelo espaçamento de envios e pela janela móvel de 60 segundos. Nas pendências atuais, o Ling será pulado, e Qwen/Gemma completarão as respostas faltantes dos primeiros 100 documentos.

## Como reage a falhas

- HTTP 408, 429, 500, 502, 503 e 504 voltam para a fila. HTTP 402 só é transitório quando os metadados identificam especificamente esgotamento do orçamento temporário de chamadas em voo; falta de créditos/cota da chave não é repetida automaticamente.
- Backoff por caso: 1, 2, 4, 8 e 16 segundos, com jitter de até 0,5 segundo e respeito ao maior `Retry-After`. A pausa vale também para novos envios daquele modelo, evitando manter a pressão durante a recuperação.
- Cada falha transitória reduz o ritmo pela metade, até o piso de 50 RPM. A concorrência cai proporcionalmente: por exemplo, 500 RPM permite até 100 em voo; 250 permite 50; 50 permite 10. Chamadas já em voo são recolhidas, não canceladas.
- Após 25 sucessos e pelo menos 30 segundos sem erro ou aumento anterior, o ritmo pode subir 50 RPM, até o teto de 500.
- Retries vencidos têm prioridade na fila. Há no máximo seis tentativas por caso em cada ciclo; depois disso, o modelo pausa e uma nova execução só abre outro ciclo após uma hora. Um `Retry-After` maior que 60 segundos pausa a execução e deixa o prazo salvo para retomada.
- Timeout/transporte com resultado desconhecido, resposta inválida, divergência de tokens/modelo/provedor e erros permanentes exigem auditoria. Não se presume que a chamada deixou de gerar ou cobrar.

## Persistência, orçamento e interrupção

Um único coordenador por modelo escreve o ledger; as threads fazem somente HTTP. O custo máximo de cada tentativa é reservado antes do envio, incluindo chamadas simultâneas. Custos reportados de respostas aceitas liberam a parte não utilizada; falhas continuam com reserva conservadora. Tetos cumulativos permanecem Qwen US$18 e Gemma US$9, cobrindo todos os lotes e retries.

`request-window.json` preserva os horários de envio. `adaptive-rate.json` preserva o ritmo reduzido e as pausas; o `ledger.json` registra cada tentativa e o prazo do retry. Aceitas não são repetidas, e duas tentativas do mesmo caso nunca ficam em voo simultaneamente.

Ctrl+C interrompe novos envios, aguarda as chamadas em voo e salva as respostas recebidas. Isso pode levar até o timeout da chamada. O plano da rodada fica preservado; a retomada usa os mesmos documentos, sem substituir casos ou alterar prompts. Falha em um modelo não impede a conclusão dos outros.

Logs mostram progresso, chamadas em voo, tamanho da fila, ritmo atual e teto. Reutilize o Bash acima; não execute clientes paralelos externos para o mesmo modelo se precisar que a janela local seja o limite agregado da conta.

## Validação

Implementação testada localmente, incluindo retry e `Retry-After`, limite de seis tentativas, orçamento em voo, resultado incerto, cancelamento, aumento gradual do ritmo, seleção do executor pago e execução simultânea dos modelos. Nenhuma nova chamada de geração foi enviada para validar esta integração. O teste de carga anterior e seus custos permanecem em [relatório de vazão](teste-vazao-openrouter-20260927.md).

## Estimativas de tempo, não medições da nova política

Com base nas latências do benchmark e no processamento paralelo: terminar as pendências atuais (415 Qwen e 300 Gemma, Ling concluído) pode levar aproximadamente 8–15 minutos; um novo grupo de 100 documentos só nos dois pagos, 10–20 minutos. Incluindo os 100 documentos do Ling gratuito, planeje cerca de 35–45 minutos, pois 600 chamadas a 20 RPM já exigem aproximadamente 30 minutos de envio. As faixas incluem validação local e margem de retries; indisponibilidade prolongada pode excedê-las.

O teto 500 não significa 500 conclusões por minuto nem ganho linear de cinco vezes. A política adaptativa ainda não foi medida em uma nova execução real; as medições existentes são as do benchmark, com 429 em 200/500 RPM.

Ao finalizar os 1.000 documentos, as métricas dos modelos são calculadas uma por vez dentro do Bash para evitar sobrepor os avaliadores locais; a concorrência acima é da geração pela API.

Validação final: 303 testes passaram, Ruff e diff check passaram, e a conferência integral do Bash nos três modelos passou sem API. O benchmark avulso não tinha criado plano ativo; foi registrado `outputs/full-rounds/active-round.json` com exatamente os 100 IDs de seu plano original, e a retomada foi verificada: Ling 0, Gemma 300 e Qwen 415 pendentes. Com isso, completar documentos no benchmark não desloca esta rodada para novos IDs. Evidência: `outputs/full-rounds/resume-first-100-validation.json`.

Lock ativo: `9339ce0e144f31ffa596482584fe5c96aca99a2fb2c742dcc86a10a191470644`, com compatibilidade explícita das preparações anteriores. Respostas preservadas: Ling 600, Qwen 185, Gemma 300. Alterações locais, ainda sem commit/push.
