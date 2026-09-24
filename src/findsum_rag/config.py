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
    """Parametros da LLM. Mantidos identicos entre as configuracoes."""

    # Qwen3.5 nao tem tamanho 7B; os densos sao 0.8B, 2B, 4B, 9B e 27B. O 9B e o
    # maior que cabe nos 12 GB da GPU em NF4 (~5,3 GB); o 27B pediria ~15 GB.
    model_name: str = "Qwen/Qwen3.5-9B"
    load_in_4bit: bool = Field(
        default=True, description="quantizacao 4-bit, necessaria em GPU de 12 GB"
    )
    # Os resumos de referencia do FINDSum-Liquidity tem ~1000 palavras de mediana
    # (~1374 tokens). Um limite menor truncaria a geracao e penalizaria a
    # cobertura de TODAS as configuracoes.
    max_new_tokens: int = Field(default=1536, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, description="0 = geracao greedy")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 42
    # Medido: 4 exemplos com resumo integral + top_k=12 dao ~10,3 mil tokens;
    # com top_k=27 (documento todo recuperado) sobem a ~14,3 mil. 16384 cobre os
    # dois casos, e a VRAM permite (pico medido de 7,9 GB em 10,3 mil tokens).
    max_input_tokens: int = Field(
        default=16384, gt=0, description="orcamento de contexto do prompt"
    )
    dtype: str = "bfloat16"

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


class ExperimentArm(BaseModel):
    """Uma das configuracoes comparadas."""

    id: str
    label: str
    context_mode: str = Field(
        pattern="^(full|retrieved|truncated)$",
        description=(
            "full = documento inteiro no prompt; retrieved = top_k trechos por "
            "similaridade (RAG); truncated = primeiros top_k trechos na ordem "
            "original, controle para isolar o efeito de recuperar vs cortar"
        ),
    )
    example_strategy: str = Field(pattern="^(none|fixed|random|dynamic)$")
    n_examples: int = Field(default=0, ge=0)

    @property
    def use_rag(self) -> bool:
        """Se o contexto foi escolhido por recuperacao."""
        return self.context_mode == "retrieved"

    @property
    def uses_full_document(self) -> bool:
        return self.context_mode == "full"

    @model_validator(mode="after")
    def _check_examples(self) -> ExperimentArm:
        if self.example_strategy == "none" and self.n_examples:
            raise ValueError(f"{self.id}: estrategia 'none' nao aceita n_examples > 0")
        if self.example_strategy != "none" and not self.n_examples:
            raise ValueError(
                f"{self.id}: estrategia {self.example_strategy!r} exige n_examples > 0"
            )
        return self


# C1 recebe o documento INTEIRO: ele cabe na janela do modelo (7,3 mil tokens de
# mediana contra 262 mil), logo truncar seria construir um baseline artificial.
# Isso muda o que H1 afirma -- deixa de ser "o RAG da mais informacao" e passa a
# ser "um recorte curado supera o documento inteiro", testavel pela degradacao
# conhecida de atencao em contexto longo.
#
# C1t existe para separar os dois efeitos: com o MESMO orcamento de C2, mas
# cortando em vez de recuperar, isola quanto do resultado vem de recuperar e
# quanto vem apenas de reduzir o contexto.
N_EXAMPLES = 4

DEFAULT_ARMS: list[ExperimentArm] = [
    ExperimentArm(
        id="C1",
        label="Baseline (documento inteiro, sem few-shot)",
        context_mode="full",
        example_strategy="none",
    ),
    ExperimentArm(
        id="C1t",
        label="Controle (truncado no orcamento do RAG, sem few-shot)",
        context_mode="truncated",
        example_strategy="none",
    ),
    ExperimentArm(
        id="C2", label="RAG", context_mode="retrieved", example_strategy="none"
    ),
    ExperimentArm(
        id="C3",
        label="RAG + few-shot fixo",
        context_mode="retrieved",
        example_strategy="fixed",
        n_examples=N_EXAMPLES,
    ),
    ExperimentArm(
        id="C4",
        label="RAG + few-shot aleatorio",
        context_mode="retrieved",
        example_strategy="random",
        n_examples=N_EXAMPLES,
    ),
    ExperimentArm(
        id="C5",
        label="RAG + few-shot dinamico",
        context_mode="retrieved",
        example_strategy="dynamic",
        n_examples=N_EXAMPLES,
    ),
]


class DataConfig(BaseModel):
    """Escopo dos dados usados no experimento.

    Os conjuntos vem de um manifesto congelado (`scripts/build_splits.py`), nao
    dos splits originais do FINDSum: aqui nada e treinado, e o que se precisa e
    separar base de exemplos, desenvolvimento e avaliacao com empresa unica.
    """

    root: Path = Path("data/raw/findsum")
    task: Task = Task.LIQUIDITY
    manifest: Path = Path("data/interim/splits-liquidity.json")
    eval_set: str = Field(default="dev", description="conjunto do manifesto a avaliar")
    example_set: str = Field(default="examples", description="conjunto das demonstracoes")
    n_eval_docs: int | None = Field(
        default=50,
        gt=0,
        description="limita o conjunto avaliado; None usa ele inteiro",
    )
    n_example_docs: int | None = Field(default=None, gt=0)
    example_max_words: int = Field(
        default=350, gt=0, description="palavras do DOCUMENTO de cada exemplo"
    )
    example_max_summary_words: int | None = Field(
        default=None,
        gt=0,
        description=(
            "palavras do RESUMO de cada exemplo; None mantem integral, que e o "
            "padrao porque a extensao-alvo e parte do que o few-shot ensina"
        ),
    )

    @model_validator(mode="after")
    def _check_sets(self) -> DataConfig:
        if self.eval_set == self.example_set:
            raise ValueError(
                "eval_set e example_set iguais causariam vazamento entre "
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
