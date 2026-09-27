# Ling gratuito: dez lotes da coorte final

Os lotes são partes do mesmo experimento: 1.000 documentos do manifesto reservado, na ordem congelada, seis braços por documento. Lote 1 contém posições 1–100; lote 2, 101–200; até lote 10, 901–1.000. Não há novo sorteio, troca de casos ou adaptação por resultado.

## Comandos

Execute na raiz do repositório:

```bash
cd /home/ikaromm/Projetos/pos_graduacao/ESP_T2_GENAI
uv run --locked findsum ling-batch
```

Sem `--execute`, valida os 6.000 prompts Ling e mostra o progresso; não consulta a API nem gera resumos. A preparação fica em `outputs/full-ling-prepared` e os resultados em `outputs/full-ling-batches`.

Para executar **um** lote de 100 ou retomar o primeiro lote incompleto:

```bash
uv run --locked findsum ling-batch --execute
```

Repita esse mesmo comando nas próximas sessões/dias. Ele para depois de um lote, sem avançar automaticamente para o seguinte. O comando com `--batch 1 --execute` fixa o primeiro lote; repeti-lo depois da conclusão não gera novamente. `--batch` aceita somente 1 a 10.

São 600 gerações por lote, sem retries. Se houver 1.000 chamadas disponíveis, sobram 400 para tentativas adicionais. A cota é consultada ao executar; o comando respeita o restante e 20 chamadas/minuto, mas não reserva saldo contra outras aplicações usando a conta. Não execute outros clientes simultaneamente. O pool gratuito ainda pode limitar chamadas.

## Retomada e evidências

Um único ledger mantém os índices globais dos 6.000 casos, modelo, provedor, prompt e resposta. Respostas aceitas são validadas e puladas na retomada. Arquivos:

- `batch-status.json`: progresso dos dez lotes e IDs de cada um.
- `ling-free/ledger.json`: tentativas, status, consumo e duração.
- `ling-free/*.request.json` e `*.response.json`: entrada e saída integrais.
- `quota-before.json`: cota consultada antes da execução.

Somente Ling gratuito/Novita, preço máximo zero, sem fallback pago. Falhas HTTP transitórias recebem até seis tentativas com espera 1/2/4/8/16 segundos e respeito a Retry-After. Após esgotar tentativas, o circuito permite uma nova rodada na retomada após uma hora. Uma falha de transporte com resultado desconhecido exige auditoria, pois repetir automaticamente poderia duplicar chamada já processada.

Ao completar os 1.000 documentos, o executor calcula métricas locais e comparações no diretório `ling-free/results/`. Se as métricas falharem, repetir o comando recalcula a análise sem repetir gerações aceitas. Não são produzidos testes de hipótese finais a cada lote de 100. Qwen/Gemma continuam pendentes e usarão a mesma coorte; começar Ling não confirma que os prompts desses outros modelos já passaram pela contagem remota.

## Preparação e congelamento

A preparação local do Ling deve cobrir os 1.000 documentos antes da primeira geração. Comando de reconstrução, apenas se não existir preparação:

```bash
uv run --locked python scripts/prepare_full_openrouter.py --full --config configs/full_openrouter.yaml --models ling-free --output outputs/full-ling-prepared
```

O executor rejeita preparação incompleta ou alterada. Não apague resultados para repetir um lote. O novo congelamento preserva o anterior em `configs/lock-history/` e mantém coorte, prompts e métricas; a mudança permite preparar e executar um modelo por vez, em lotes. As outras etapas continuam disponíveis no executor integrado de três modelos.

## Correção encontrada na validação real

O primeiro preflight produziu 5.934 prompts: 11 documentos falharam no pareamento C1t/C2. O recorte confundia ` | ` dentro de prosa com linha de tabela. Corrigido para reconhecer apenas os blocos marcados pelo serializador, mantendo proteção de células e igualdade de tokens. Os 11 controles passaram na rechecagem; todos os prompts foram revalidados antes de liberar geração. Nenhum documento foi removido. Insumos e relatório inicial preservados em `outputs/full-ling-prepared-initial`; a preparação vigente usa a mesma fonte, recuperação e IDs de exemplos.

Validação concluída: 6.000/6.000 prompts, 1.000/1.000 documentos, zero erros. Maior entrada: 43.743 tokens; entrada máxima permitida: 49.152; reserva de saída: 8.192; janela Ling: 262.144. Nenhum dos 5.934 prompts inicialmente válidos foi alterado. 262 testes passaram, Ruff e diff check passaram. Relatório: `outputs/full-ling-prepared/report.json`. Lock ativo: `c0d3eb15c1eb74f516b4d63e22e82eeac614a53d6d63789f61db970c44ce88f1`. Nenhuma chamada de geração foi executada pelo assistente.

O comando real `uv run --locked findsum ling-batch` foi executado sem `--execute` e terminou com código 0: `accepted_generations=0`, `next_batch=1`, 600 chamadas pendentes no lote 1. Nenhuma consulta de cota ou geração foi realizada.
