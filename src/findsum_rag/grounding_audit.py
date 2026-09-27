"""Alertas para revisao, sem corrigir/descartar saidas ou aprovar factualidade."""

from __future__ import annotations

from .metrics import extract_numbers


def audit_amounts(prediction: str, source: str, context: str, examples) -> list[dict]:
    """Sinaliza escalas sem suporte numerico e valores exclusivos de exemplos.

    Coincidencia numerica nao prova entidade, periodo, direcao ou unidade.
    Ausencia pode ser efeito da representacao do dado: sao candidatos a revisao,
    nunca rotulos humanos nem criterio para excluir seletivamente geracoes.
    """
    source_values = {n.value for n in extract_numbers(source)}
    context_values = {n.value for n in extract_numbers(context)}
    example_values = {
        e.doc_id: {n.value for n in extract_numbers(e.document + " " + e.summary)}
        for e in examples
    }
    warnings, seen = [], set()
    for n in extract_numbers(prediction):
        if n.value in seen:
            continue
        seen.add(n.value)
        matching = [key for key, values in example_values.items() if n.value in values]
        if n.value not in source_values and (matching or n.scaled):
            warnings.append({
                "kind": "example_only_number" if matching else "unsupported_scaled_amount",
                "raw_number": n.raw, "normalized_value": n.value,
                "example_ids": matching, "in_source": False,
                "in_context": n.value in context_values,
                "requires_human_review": True,
            })
    return warnings
