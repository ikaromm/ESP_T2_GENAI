"""Endpoints gratuitos autorizados para triagem no dev (2026-09-26)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FreeEndpoint:
    provider: str
    response_provider: str
    tokenizer: str
    context_length: int = 262144
    max_completion_tokens: int = 32768


FREE_ENDPOINTS = {
    "qwen/qwen3.8-27b:free": FreeEndpoint(
        "modelrun/fp4", "ModelRun", "Qwen/Qwen3.8-27B", max_completion_tokens=235929
    ),
    "inclusionai/ling-3.0-flash-fin:free": FreeEndpoint(
        "novita", "Novita", "inclusionAI/Ling-3.0-flash-Fin"
    ),
    "google/gemma-4-31b-it:free": FreeEndpoint(
        "google-ai-studio", "Google AI Studio", "google/gemma-4-31B-it"
    ),
}


# Contratos publicos consultados em 2026-09-26; sem geracao de validacao nesta migracao.
PAID_ENDPOINTS = {
    "inclusionai/ling-3.0-flash-fin": FreeEndpoint(
        "deepinfra/fp4",
        "DeepInfra",
        "inclusionAI/Ling-3.0-flash-Fin",
        max_completion_tokens=235929,
    ),
    "qwen/qwen3.7-flash": FreeEndpoint(
        "alibaba",
        "Alibaba",
        "",
        context_length=1000000,
        max_completion_tokens=65536,
    ),
    "google/gemma-4-26b-a4b-it": FreeEndpoint(
        "darkbloom",
        "Darkbloom",
        "google/gemma-4-26B-A4B-it",
        context_length=131072,
    ),
}
