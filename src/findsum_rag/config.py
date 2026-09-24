"""Configuracao dos experimentos.

As cinco configuracoes do projeto (C1 a C5) sao declaradas aqui como dados, nao
como codigo espalhado pelo pipeline. A comparacao so e valida se tudo o que nao
esta sob investigacao permanecer constante entre elas, portanto modelo, limites
de contexto e parametros de geracao vivem no nivel do experimento, e cada
configuracao varia apenas o uso de RAG e a estrategia de exemplos.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from .data import Task


class RetrievalConfig(BaseModel):
    """Parametros da recuperacao de contexto (RAG)."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=220, gt=0, description="palavras por trecho")
    chunk_overlap: int = Field(default=40, ge=0)
    top_k: int = Field(default=12, gt=0, description="trechos recuperados por documento")
    query_words: int = Field(default=400, gt=0, description="palavras usadas na consulta")
    include_tables: bool = True

    @model_validator(mode="after")
    def _check_overlap(self) -> RetrievalConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap deve ser menor que chunk_size")
        return self


class GenerationConfig(BaseModel):
    """Parametros da LLM. Mantidos identicos entre C1 e C5."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = Field(
        default=True, description="quantizacao 4-bit, necessaria em GPU de 12 GB"
    )
    # Os resumos de referencia do FINDSum-Liquidity tem ~1000 palavras em media
    # (medido em val: min 384, max 1181), ou seja ~1300 tokens. Um limite menor
    # truncaria a geracao e penalizaria a cobertura de TODAS as configuracoes.
    max_new_tokens: int = Field(default=1536, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, description="0 = geracao greedy")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 42
    # Orcamento de prompt: ~3.5k tokens de contexto recuperado (12 x 220 palavras)
    # + ~3.6k de dois exemplos com resumo integral + instrucoes.
    max_input_tokens: int = Field(
        default=12288, gt=0, description="orcamento de contexto do prompt"
    )
    dtype: str = "bfloat16"

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


class ExperimentArm(BaseModel):
    """Uma das configuracoes comparadas."""

    id: str
    label: str
    use_rag: bool
    example_strategy: str = Field(pattern="^(none|fixed|random|dynamic)$")
    n_examples: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_examples(self) -> ExperimentArm:
        if self.example_strategy == "none" and self.n_examples:
            raise ValueError(f"{self.id}: estrategia 'none' nao aceita n_examples > 0")
        if self.example_strategy != "none" and not self.n_examples:
            raise ValueError(
                f"{self.id}: estrategia {self.example_strategy!r} exige n_examples > 0"
            )
        return self


DEFAULT_ARMS: list[ExperimentArm] = [
    ExperimentArm(
        id="C1", label="Baseline (sem RAG, sem few-shot)", use_rag=False, example_strategy="none"
    ),
    ExperimentArm(id="C2", label="RAG", use_rag=True, example_strategy="none"),
    ExperimentArm(
        id="C3", label="RAG + few-shot fixo", use_rag=True, example_strategy="fixed", n_examples=2
    ),
    ExperimentArm(
        id="C4",
        label="RAG + few-shot aleatorio",
        use_rag=True,
        example_strategy="random",
        n_examples=2,
    ),
    ExperimentArm(
        id="C5",
        label="RAG + few-shot dinamico",
        use_rag=True,
        example_strategy="dynamic",
        n_examples=2,
    ),
]


class DataConfig(BaseModel):
    """Escopo dos dados usados no experimento."""

    root: Path = Path("data/raw/findsum")
    task: Task = Task.LIQUIDITY
    eval_split: str = "val"
    example_split: str = "train"
    n_eval_docs: int = Field(default=100, gt=0, description="documentos avaliados")
    n_example_docs: int = Field(
        default=500, gt=0, description="documentos que formam a base de exemplos"
    )
    example_max_words: int = Field(
        default=350, gt=0, description="palavras por documento de exemplo no prompt"
    )

    @model_validator(mode="after")
    def _check_splits(self) -> DataConfig:
        if self.eval_split == self.example_split:
            raise ValueError(
                "eval_split e example_split iguais causariam vazamento entre "
                "avaliacao e base de exemplos"
            )
        return self


class ExperimentConfig(BaseModel):
    """Configuracao completa de uma rodada experimental."""

    name: str = "baseline-run"
    output_dir: Path = Path("outputs")
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    arms: list[ExperimentArm] = Field(default_factory=lambda: list(DEFAULT_ARMS))

    @model_validator(mode="after")
    def _check_arms(self) -> ExperimentConfig:
        ids = [a.id for a in self.arms]
        if len(ids) != len(set(ids)):
            raise ValueError(f"ids de configuracao repetidos: {ids}")
        if not self.arms:
            raise ValueError("e preciso pelo menos uma configuracao em arms")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Carrega a configuracao de um arquivo YAML."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Grava a configuracao efetiva, para registro junto dos resultados."""
        payload = self.model_dump(mode="json")
        Path(path).write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def arm(self, arm_id: str) -> ExperimentArm:
        """Recupera uma configuracao pelo id.

        Raises:
            KeyError: se o id nao existir.
        """
        for arm in self.arms:
            if arm.id == arm_id:
                return arm
        raise KeyError(f"configuracao {arm_id!r} nao encontrada; disponiveis: "
                       f"{[a.id for a in self.arms]}")
