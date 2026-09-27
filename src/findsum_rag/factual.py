"""Revisao humana de afirmacoes atomicas; sem inferir fatos de numeros isolados."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

LABELS = {"supported", "contradicted", "insufficient_evidence"}


def evidence_digest(case: dict) -> str:
    evidence = {key: case[key] for key in ("prediction", "source", "context")}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()


def export_review(run_dir: Path, destination: Path) -> None:
    """Exporta casos sem rotulo de braco; o mapa separado permite reuniao posterior.

    O revisor segmenta o resumo em afirmacoes atomicas. Nenhuma afirmacao ou
    classificacao factual e preenchida automaticamente.
    """
    cases, mapping = [], {}
    for path in sorted(run_dir.glob("*/predictions.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            case_id = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            mapping[case_id] = {
                "arm": row["arm"],
                "doc_id": row["doc_id"],
                "evidence_sha256": evidence_digest(row),
            }
            cases.append(
                {
                    "case_id": case_id,
                    "prediction": row["prediction"],
                    "source": row["source"],
                    "context": row["context"],
                    "reviewer": "",
                    "segmentation_complete": False,
                    "claims": [],
                }
            )
    if not cases:
        raise ValueError("nenhuma previsao com evidencias para revisar")
    if destination.exists() or destination.with_suffix(".map.json").exists():
        raise ValueError("arquivo de revisao ou mapa ja existe")
    # Ordem pseudoaleatoria deterministica, sem agrupar os bracos.
    cases.sort(key=lambda c: c["case_id"])
    destination.write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    destination.with_suffix(".map.json").write_text(json.dumps(mapping, indent=2))


def score_review(cases: list[dict]) -> list[dict]:
    """Rejeita revisoes incompletas e exige evidencia localizavel e justificativa.

    Cada claim requer: text, entity, period, relation, value ("not_applicable"
    quando cabivel), source_label, context_label, source_evidence,
    context_evidence e rationale. A validade semantica e responsabilidade humana.
    """
    results, seen = [], set()
    for case in cases:
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError("caso duplicado")
        seen.add(case_id)
        if not case["reviewer"].strip() or case["segmentation_complete"] is not True:
            raise ValueError(f"{case_id}: revisao incompleta")
        claims = case["claims"]
        if not claims:
            raise ValueError(f"{case_id}: nenhuma afirmacao revisada; nao atribuir escore perfeito")
        for claim in claims:
            for field in ("text", "entity", "period", "relation", "value", "rationale"):
                if not str(claim.get(field, "")).strip():
                    raise ValueError(f"{case_id}: campo obrigatorio {field}")
            if claim["text"] not in case["prediction"]:
                raise ValueError(f"{case_id}: afirmacao nao localizavel no resumo")
            for scope in ("source", "context"):
                label = claim.get(f"{scope}_label")
                if label not in LABELS:
                    raise ValueError(f"{case_id}: rotulo invalido em {scope}")
                quote = claim.get(f"{scope}_evidence", "")
                if label != "insufficient_evidence" and (not quote or quote not in case[scope]):
                    raise ValueError(f"{case_id}: evidencia ausente ou nao localizavel em {scope}")
            if claim["context_label"] == "supported" and claim["source_label"] != "supported":
                raise ValueError(f"{case_id}: suporte no contexto exige suporte na fonte")
        result = {"case_id": case_id, "n_claims": len(claims)}
        for scope in ("source", "context"):
            for label in sorted(LABELS):
                result[f"{scope}_{label}_rate"] = sum(
                    claim[f"{scope}_label"] == label for claim in claims
                ) / len(claims)
        results.append(result)
    if not results:
        raise ValueError("nenhuma revisao")
    return results
