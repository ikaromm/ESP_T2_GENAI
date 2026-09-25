"""Testes dos perfis de execucao usados pelo deploy.sh.

Um perfil quebrado so apareceria depois de subir a imagem no servidor e esperar
o download do modelo. Estes testes pegam o erro em 1 segundo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from findsum_rag.config import ExperimentConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PROFILES = sorted(p.stem for p in CONFIGS.glob("*.yaml"))


def test_expected_profiles_exist():
    assert "test_50" in PROFILES
    assert "eval_1000" in PROFILES


@pytest.mark.parametrize("profile", PROFILES)
def test_profile_is_valid(profile: str):
    """Todo YAML em configs/ tem de validar contra o schema."""
    config = ExperimentConfig.from_yaml(CONFIGS / f"{profile}.yaml")
    assert config.name
    assert config.arms


@pytest.mark.parametrize("profile", PROFILES)
def test_profile_name_matches_filename(profile: str):
    """O `name` define o diretorio de saida; divergir do arquivo confunde."""
    if profile == "experiment":  # gerado por `findsum init-config`
        return
    config = ExperimentConfig.from_yaml(CONFIGS / f"{profile}.yaml")
    assert config.name == profile, (
        f"o perfil {profile}.yaml declara name={config.name!r}; a saida iria "
        f"para outputs/{config.name} em vez de outputs/{profile}"
    )


def test_test_50_does_not_touch_the_evaluation_set():
    """O conjunto `eval` e aberto uma unica vez, pelo perfil eval_1000."""
    dev = ExperimentConfig.from_yaml(CONFIGS / "test_50.yaml")
    assert dev.data.eval_set == "dev"
    assert dev.data.n_eval_docs == 50


def test_eval_1000_uses_the_full_evaluation_set():
    final = ExperimentConfig.from_yaml(CONFIGS / "eval_1000.yaml")
    assert final.data.eval_set == "eval"
    # None = usa o conjunto inteiro, sem limite.
    assert final.data.n_eval_docs is None


def test_profiles_never_reuse_the_example_set_for_evaluation():
    for profile in PROFILES:
        config = ExperimentConfig.from_yaml(CONFIGS / f"{profile}.yaml")
        assert config.data.eval_set != config.data.example_set


@pytest.mark.parametrize("profile", ["test_50", "eval_1000"])
def test_profiles_cover_the_full_ablation_chain(profile: str):
    config = ExperimentConfig.from_yaml(CONFIGS / f"{profile}.yaml")
    assert [a.id for a in config.arms] == ["C1", "C1t", "C2", "C3", "C4", "C5"]
    by_id = {a.id: a for a in config.arms}
    assert by_id["C1"].context_mode == "full"
    assert by_id["C1t"].context_mode == "truncated"
    assert by_id["C2"].context_mode == "retrieved"


@pytest.mark.parametrize("profile", ["test_50", "eval_1000"])
def test_retrieval_budget_is_smaller_than_the_document(profile: str):
    """`top_k` alto demais faz o RAG trazer o documento inteiro e C2 = C1.

    Um documento do FINDSum-Liquidity gera ~39 trechos com os parametros padrao
    (medido). Com `top_k` nessa ordem, C2 deixa de ser uma condicao distinta.
    """
    config = ExperimentConfig.from_yaml(CONFIGS / f"{profile}.yaml")
    assert config.retrieval.top_k < 30, (
        f"{profile}: top_k={config.retrieval.top_k} se aproxima do total de "
        "trechos do documento; C2 viraria uma copia de C1"
    )


def test_profiles_keep_generation_identical_across_arms():
    """O que varia entre configuracoes e contexto e exemplos, nada mais.

    Os parametros de geracao vivem no nivel do experimento justamente para que
    nao possam divergir por configuracao.
    """
    for profile in ("test_50", "eval_1000"):
        raw = yaml.safe_load((CONFIGS / f"{profile}.yaml").read_text(encoding="utf-8"))
        for arm in raw["arms"]:
            assert set(arm) <= {"id", "label", "context_mode", "example_strategy", "n_examples"}, (
                f"{profile}: a configuracao {arm.get('id')} declara campos alem "
                "de contexto/exemplos, o que quebraria o controle experimental"
            )


def test_max_input_tokens_fits_the_worst_case_prompt():
    """4 exemplos com resumo integral + top_k trechos precisam caber.

    Medido: 4 exemplos integrais com top_k=12 dao ~10,3 mil tokens; com top_k=27,
    ~14,3 mil. Um orcamento menor truncaria silenciosamente as configuracoes com
    few-shot e so elas, invalidando a comparacao.
    """
    for profile in ("test_50", "eval_1000"):
        config = ExperimentConfig.from_yaml(CONFIGS / f"{profile}.yaml")
        assert config.generation.max_input_tokens >= 14500, (
            f"{profile}: max_input_tokens={config.generation.max_input_tokens} "
            "corre risco de truncar as configuracoes com 4 exemplos"
        )
