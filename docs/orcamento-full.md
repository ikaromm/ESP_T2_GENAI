# Orçamento do full — 26/09/2026

Estimativa para 1.000 documentos × seis braços × três modelos: **18.000 gerações**. Todo o Ling é precificado como pago neste orçamento. Não foram feitas novas gerações ou sondas de calibração para calculá-lo.

## Estimativa principal

| Etapa | Chamadas projetadas | Custo US$ | Teto reservado US$ |
|---|---:|---:|---:|
| Ling Flash Fin / DeepInfra pago | 6.000 | 8,62 | 15,00 |
| Qwen3.7 Flash / Alibaba | 6.000 | 12,66 | 18,00 |
| Gemma 4 26B A4B / Darkbloom | 6.000 | 6,20 | 9,00 |
| Calibração de tokens Qwen | aproximadamente 33.200 | 14,56 | 18,00 |
| **Total** | **aproximadamente 51.200** | **42,03** | **60,00** |

Total sem arredondamento: US$ 42,0294106. Com margem de 30%: **US$ 54,64**. Reserva sugerida: **US$ 60**. Os tetos interrompem o trabalho quando atingidos; não garantem conclusão.

A calibração faz sondas remuneradas para contar os prompts e ajustar o controle C1t/C2 com o tokenizer remoto do Qwen. Não são resumos adicionais. A projeção escala as 332 sondas do dev10; cache compartilhado, tamanho dos documentos e retries podem mudar a quantidade.

## Base de cálculo

Uso real das 60 respostas por modelo do dev10 corrigido, escalado por 100. Cada prompt foi precificado na faixa correspondente, sem desconto de cache. Fórmula: `(tokens de entrada × tarifa de entrada + tokens de saída × tarifa de saída) / 1.000.000`.

Tarifas por milhão, consultadas no catálogo público em 27/09/2026 UTC (26/09 no Brasil):

| Modelo/provedor | Entrada US$ | Saída US$ |
|---|---:|---:|
| Ling / DeepInfra FP4 | 0,06 | 0,18 |
| Qwen / Alibaba, abaixo de 32.000 tokens | 0,03 | 0,13 |
| Qwen / Alibaba, a partir de 32.000 tokens | 0,10 | 0,40 |
| Gemma / Darkbloom | 0,042 | 0,22 |

O teto de entrada de 49.152 tokens evita a faixa Qwen de 256 mil, mas **não** evita a faixa de 32 mil. Por isso, multiplicar todos os prompts por US$ 0,03/0,13 subestima o orçamento.

Fontes: [Ling](https://openrouter.ai/api/v1/models/inclusionai/ling-3.0-flash-fin/endpoints), [Qwen](https://openrouter.ai/api/v1/models/qwen/qwen3.7-flash/endpoints), [Gemma](https://openrouter.ai/api/v1/models/google/gemma-4-26b-a4b-it/endpoints). Snapshots e cálculo reproduzível: `outputs/full-planning-20260926/`; script `scripts/budget_full_openrouter.py`.

## Incerteza e execução

Mantendo os mesmos tamanhos de entrada, mas usando 8.192 tokens em **todas** as saídas, a projeção sobe para **US$ 72,45**, ou **US$ 94,19** com 30% de margem. Cerca de US$ 100 cobre esse cenário específico; não é um pior caso absoluto. A estimativa principal depende de comprimentos semelhantes ao dev10. Não inclui taxas de compra de créditos, câmbio, impostos nem computação local.

O Ling **continua configurado no gratuito / Novita**. A linha paga é um cenário financeiro solicitado, não uma migração automática. Migrar a execução Ling para DeepInfra exige um perfil separado e validação do contrato desse provedor, preservando a identidade experimental. Os tetos ativos pagos são US$ 18 + US$ 18 + US$ 9; US$ 15 ficam reservados ao cenário Ling pago.

Ainda não foram preparados os prompts dos 1.000 documentos nem executada sua calibração. O orçamento será refinável com os artefatos dessa etapa, após autorização para as chamadas.
