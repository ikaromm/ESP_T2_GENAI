# Resultados parciais: primeiros 100 documentos

[Interpretação e matriz completa](../../docs/metricas-primeiros100-20260927.md).

São 1.800 respostas: os mesmos 100 documentos FINDSum Liquidity, seis braços e três modelos. Estes resultados pertencem à coorte de avaliação de 1.000 documentos; não são o dev nem uma avaliação final completa.

- `scores.csv`: métricas por documento, modelo e braço; inclui o motivo de término.
- `summary.json`: médias dos 18 grupos e diferenças descritivas H1/H1b/H2/H3/H4.
- `cohort.json`: IDs e ordem dos 100 documentos, com identificação do plano.
- `validation.json`: hashes dos artefatos locais de origem e auditoria de tokens, modelo/camada BERTScore e tempo de cálculo.

As saídas completas e referências permanecem nos artefatos locais ignorados pelo Git. Os caminhos em `validation.json` identificam essas fontes locais; os hashes não substituem os arquivos para recalcular as métricas.

Não foram calculados p-valores nem testes confirmatórios. Nenhuma hipótese está confirmada por este recorte; H3 é controle de sensibilidade. As duas saídas Gemma com `finish_reason=length` estão incluídas conforme protocolo. BERTScore foi calculado sobre os textos integrais, sem truncamento.

Nenhuma nova chamada de geração foi necessária para calcular estas métricas. Os resultados parciais não alteram a configuração congelada do experimento.
