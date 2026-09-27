"""Contratos de preparacao e elegibilidade, sem rede ou geracao."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

ARMS = ("C1", "C1t", "C2", "C3", "C4", "C5")
FULL_MODELS = ("ling-free", "qwen37", "gemma26")


def random_records(records, count, *, seed, split):
    """Sorteio sem reposicao dentro do split reservado, independente de respostas."""
    if split not in {"dev", "eval"}:
        raise ValueError("split de selecao invalido")
    pool = sorted(records, key=lambda r: r["doc_id"])
    if len({r["doc_id"] for r in pool}) != len(pool):
        raise ValueError("documentos duplicados no conjunto")
    count = len(pool) if count is None else count
    if not 0 < count <= len(pool):
        raise ValueError("tamanho da amostra fora do conjunto reservado")
    return random.Random(f"cohort:{seed}:{split}").sample(pool, count)


@dataclass(frozen=True)
class Limits:
    max_input: int = 49152
    max_output: int = 3072
    context_length: int = 131072
    margin: int = 256
    allow_length: bool = False

    def check(self, prompt_tokens):
        if not isinstance(prompt_tokens, int) or not 0 < prompt_tokens <= self.max_input:
            raise ValueError("entrada acima do teto ou contagem invalida")
        if self.max_output <= 0 or self.margin < 0:
            raise ValueError("reserva de saida/margem invalida")
        if prompt_tokens + self.max_output + self.margin > self.context_length:
            raise ValueError("entrada + saida reservada + margem excedem a janela")

    def to_dict(self):
        return asdict(self)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hashes(folder):
    folder = Path(folder)
    for relative, digest in json.loads((folder / "sha256.json").read_text()).items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != digest:
            raise ValueError(f"artefato preparado alterado: {relative}")


def selected_records(folder):
    selection = json.loads((Path(folder) / "selection.json").read_text())
    split = selection.get("split", "dev")
    if split not in {"dev", "eval"}:
        raise ValueError("split de geracao invalido")
    records = selection.get("documents", selection.get("dev"))
    manifest_path = Path(selection.get("manifest_path", "data/interim/splits-liquidity.json"))
    if sha(manifest_path) != selection["manifest_sha256"]:
        raise ValueError("manifesto alterado")
    manifest = json.loads(manifest_path.read_text())
    if selection.get("sampling", {}).get("method") == "random_without_replacement":
        expected = random_records(
            manifest["sets"][split], len(records), seed=selection["sampling"]["seed"], split=split
        )
    else:
        expected = manifest["sets"][split][: len(records)]
    if not records or records != expected:
        raise ValueError("candidatos diferem do manifesto reservado")
    return split, records


def eligibility(
    candidate_ids, rows_by_model, limits_by_model, errors_by_model=None, *, expected_count=1000
):
    """Exige a coorte completa ANTES da geracao; nunca reduz os 1000 silenciosamente."""
    if len(candidate_ids) != expected_count:
        raise ValueError(f"esperados {expected_count} documentos; recebidos {len(candidate_ids)}")
    errors_by_model = errors_by_model or {}
    eligible, excluded = [], []
    if set(rows_by_model) != set(FULL_MODELS):
        raise ValueError("os tres modelos sao obrigatorios")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidatos duplicados")
    indices = {}
    for model, rows in rows_by_model.items():
        index = {}
        for row in rows:
            key = (row["doc_id"], row["arm"])
            if key in index or row["doc_id"] not in candidate_ids or row["arm"] not in ARMS:
                raise ValueError("preflight duplicado ou fora da selecao")
            index[key] = row
        indices[model] = index
    for doc_id in candidate_ids:
        reasons = []
        for model, index in indices.items():
            group = {arm: index.get((doc_id, arm)) for arm in ARMS}
            if not all(group.values()):
                known = [e for e in errors_by_model.get(model, []) if e["doc_id"] == doc_id]
                if not known:
                    raise ValueError(f"preflight incompleto sem motivo: {model}/{doc_id}")
                reasons.append({"model": model, "reason": "preflight_failed", "details": known})
                continue
            try:
                for row in group.values():
                    limits_by_model[model].check(row["prompt_tokens"])
            except ValueError as exc:
                reasons.append({"model": model, "reason": str(exc)})
            if any(
                group["C1t"][key] != group["C2"][key] for key in ("context_tokens", "prompt_tokens")
            ):
                raise ValueError(f"controle C1t/C2 invalido: {model}/{doc_id}")
            if len({group[a]["context"] for a in ("C2", "C3", "C4", "C5")}) != 1:
                raise ValueError(f"contextos RAG diferentes: {model}/{doc_id}")
        if reasons:
            excluded.append({"doc_id": doc_id, "reasons": reasons})
        else:
            eligible.append(doc_id)
    if not eligible:
        raise ValueError("nenhum documento elegivel nos tres modelos")
    return {
        "ready": not excluded and len(eligible) == expected_count,
        "expected_count": expected_count,
        "eligible_doc_ids": eligible,
        "excluded": excluded,
        "candidate_count": len(candidate_ids),
        "eligible_count": len(eligible),
    }
