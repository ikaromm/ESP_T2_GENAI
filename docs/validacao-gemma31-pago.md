# Validacao limitada Gemma 4 31B pago — 2026-09-26

Autorizada a continuidade com Gemma 4 31B, mantendo os cinco documentos dev e seis bracos. Provedor fixado Reka (`reka`), modelo `google/gemma-4-31b-it`, sem fallback, temperature 0, top_p 1, reasoning desativado e sem seed. Teto local conservador US$ 0,10; limites de preco entrada 0,08/saida 0,30 USD por milhao. Reservas persistidas antes de cada tentativa; erros desconhecidos nao liberam reserva.

Reproducao: `uv run --no-sync --env-file ../.env python scripts/replay_gemma_paid.py --execute`. Sem --execute apenas pre-valida. O script recusa retomar uma tentativa que exige auditoria.

Os 30 prompts salvos do teste gratuito passaram novamente na pre-validacao local, sem alterar contexto ou exemplos. A primeira chamada retornou texto (ABEO/C1), 11.181 tokens de entrada e 429 de saida, sem reasoning; custo reportado US$ 0,00102318. O tokenizer local contou 11.180 tokens. O lote foi interrompido por divergencia de um token, antes de aceitar qualquer caso ou fazer a segunda chamada. Nenhuma avaliacao final executada.

Artefatos: `outputs/screen-gemma31-paid-5docs-20260926/`, incluindo requisicao, resposta integral, ledger com reserva e status.json com custo observado. A reserva e conservadora e nao representa a cobranca efetiva. A resposta inicial repete representacoes ambiguas com ampersand; qualidade factual ainda nao aprovada. A divergencia de contagem precisa ser explicada antes de continuar, sem relaxar silenciosamente a igualdade C1t/C2.

Validacao do script: preflight de 30 prompts, checks locais de payload, teto e rejeicao de contagem/custo invalido, Ruff aprovado. Cliente gratuito e seu bloqueio a custos pagos permaneceram intactos.
