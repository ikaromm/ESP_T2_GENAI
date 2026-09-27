"""Contrato congelado do full: verificacao offline antes de qualquer chamada."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

DEFAULT_LOCK = Path("configs/full_openrouter.lock.json")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lock_digest(data):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in data.items() if k != "lock_id"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def verify_full_lock(path=DEFAULT_LOCK, *, prepared=None, check_assets=True):
    from huggingface_hub.constants import HF_HUB_CACHE

    path = Path(path)
    if not path.is_file():
        raise ValueError("full exige configuracao congelada antes de qualquer chamada")
    data = json.loads(path.read_text())
    if data.get("lock_id") != lock_digest(data):
        raise ValueError("identificador do lock invalido")
    for section in ["files", "dataset_files"]:
        for name, expected in data[section].items():
            if digest(name) != expected:
                raise ValueError(f"arquivo difere do full congelado: {name}")
    for package, version in data["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"versao diferente da congelada: {package}")
    if check_assets:
        for model, spec in data["hf_assets"].items():
            base = Path(HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))
            if (base / "refs/main").read_text().strip() != spec["revision"]:
                raise ValueError(f"revisao local mudou: {model}")
            for name, expected in spec["files"].items():
                if digest(base / "snapshots" / spec["revision"] / name) != expected:
                    raise ValueError(f"asset local mudou: {model}/{name}")
    if data.get("wordnet_sha256"):
        import nltk

        resource = nltk.data.find("corpora/wordnet.zip")
        if digest(resource.zipfile.filename) != data["wordnet_sha256"]:
            raise ValueError("recurso WordNet difere do congelado")
    if prepared is not None:
        prepared = Path(prepared)
        selection = json.loads((prepared / "selection.json").read_text())
        if selection["split"] != "eval" or selection["documents"] != data["cohort"]:
            raise ValueError("coorte difere dos 1000 casos congelados")
        from .config import ExperimentConfig

        expected = ExperimentConfig.from_yaml("configs/full_openrouter.yaml")
        actual = ExperimentConfig.from_yaml(prepared / "config.yaml")
        if actual != expected:
            raise ValueError("configuracao preparada difere do full congelado")
        prepared_lock = json.loads((prepared / "full-lock.json").read_text())
        if prepared_lock["lock_id"] != data["lock_id"]:
            raise ValueError("preparacao pertence a outro lock")
    # Evita atualizacao automatica de main nos modelos avaliadores/tokenizers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_OFFLINE = True
    return data
