"""Cliente para PoCs nos endpoints gratuitos autorizados, sem fallback pago."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

from .config import GenerationConfig
from .generate import Generation, PromptTokenizer, clean_generation
from .prompts import Prompt
from .remote_models import FREE_ENDPOINTS

MODEL = "qwen/qwen3.8-27b:free"
PROVIDER = "modelrun/fp4"
BASE_URL = "https://openrouter.ai/api/v1"
TOKENIZER = "Qwen/Qwen3.8-27B"


class OpenRouterHTTPError(RuntimeError):
    """Erro sanitizado com status e Retry-After para retries limitados."""

    def __init__(self, details: dict):
        self.details = details
        self.status = details["http_status"]
        super().__init__(
            f"OpenRouter HTTP {self.status}: " + json.dumps(details, ensure_ascii=False)
        )

    def retry_delay(self, default: float) -> float:
        value = self.details.get("headers", {}).get("Retry-After")
        if value is None:
            return default
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError):
                return default
        return max(default, seconds)


class OpenRouterFreeClient:
    def __init__(self, key: str, *, timeout: float = 180, model: str = MODEL):
        if not key:
            raise ValueError("OPEN_ROUTER_KEY ausente")
        if model not in FREE_ENDPOINTS:
            raise ValueError("modelo fora da lista gratuita autorizada")
        self._key = key
        self.timeout = timeout
        self.model = model

    def _request(self, route: str, payload: dict | None = None) -> dict:
        request = urllib.request.Request(
            BASE_URL + route,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Somente resposta diagnostica e headers de limite; nunca headers de autenticacao.
            raw = exc.read(32768).decode("utf-8", errors="replace")
            raw = raw.replace(self._key, "[REDACTED]")
            raw = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", raw)
            raw = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", raw)
            try:
                error = json.loads(raw).get("error", {})
            except (ValueError, AttributeError):
                error = {"message": "resposta de erro nao JSON"}
            if not isinstance(error, dict):
                error = {"message": "formato de erro inesperado"}
            metadata = error.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            details = {
                "http_status": exc.code,
                "message": error.get("message"),
                "metadata": {
                    k: metadata[k]
                    for k in (
                        "provider_name",
                        "provider_code",
                        "error_type",
                        "limit_source",
                        "reason",
                        "raw",
                    )
                    if k in metadata
                },
                "headers": {
                    k: exc.headers[k]
                    for k in (
                        "Retry-After",
                        "X-RateLimit-Limit",
                        "X-RateLimit-Remaining",
                        "X-RateLimit-Reset",
                    )
                    if exc.headers and k in exc.headers
                },
            }
            raise OpenRouterHTTPError(details) from None

    def quota(self) -> dict:
        data = self._request("/key")["data"]
        return {
            "is_free_tier": data.get("is_free_tier"),
            "free_model_daily_requests": data.get("free_model_daily_requests"),
        }

    @staticmethod
    def payload(messages: list[dict], max_tokens: int = 1536, *, model: str = MODEL) -> dict:
        if model not in FREE_ENDPOINTS:
            raise ValueError("modelo fora da lista gratuita autorizada")
        endpoint = FREE_ENDPOINTS[model]
        if not 0 < max_tokens <= endpoint.max_completion_tokens:
            raise ValueError("max_tokens fora do limite do endpoint")
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
            "reasoning": {"enabled": False},
            "provider": {
                "only": [endpoint.provider],
                "allow_fallbacks": False,
                "require_parameters": True,
                "max_price": {"prompt": 0, "completion": 0},
            },
        }

    def complete(self, messages: list[dict], *, max_tokens: int = 1536) -> dict:
        start = time.monotonic()
        data = self._request(
            "/chat/completions", self.payload(messages, max_tokens, model=self.model)
        )
        if data.get("error"):
            raise RuntimeError("OpenRouter retornou erro no corpo; sem fallback")
        choices = data.get("choices", [])
        if not choices or not isinstance(choices[0].get("message", {}).get("content"), str):
            raise RuntimeError("resposta sem conteudo textual")
        choice = choices[0]
        text = choice["message"]["content"]
        if not text.strip():
            raise RuntimeError("resposta vazia; conferir reasoning e limite de saida")
        usage = data.get("usage", {})
        if usage.get("cost") not in (None, 0):
            raise RuntimeError("custo inesperado em endpoint gratuito; interrompido")
        return {
            "id": data.get("id"),
            "model": data.get("model"),
            "provider": data.get("provider"),
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }


class OpenRouterSummarizer(PromptTokenizer):
    """Mesmo contrato local, geracao remota gratuita e auditoria por chamada.

    Nao carrega pesos da LLM nem faz retries. O tokenizer e o chat template
    precisam concordar com a contagem efetivamente reportada pelo provedor.
    """

    def __init__(self, config: GenerationConfig, *, audit_dir: Path):
        config = GenerationConfig.model_validate(config.model_dump())
        if config.backend != "openrouter":
            raise ValueError("backend deve ser openrouter")
        super().__init__(config)
        self.endpoint = FREE_ENDPOINTS[config.model_name]
        self.client = OpenRouterFreeClient(
            os.environ.get("OPEN_ROUTER_KEY", ""), model=config.model_name
        )
        self.audit_dir = audit_dir
        self._calls = 0

    def load_tokenizer(self) -> None:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.endpoint.tokenizer, trust_remote_code=False
            )
            if not self._tokenizer.is_fast or not self._tokenizer.chat_template:
                raise ValueError("OpenRouter exige tokenizer fast e chat template do modelo")

    def load(self) -> None:
        self.load_tokenizer()

    def render(self, prompt: Prompt) -> str:
        return self.tokenizer.apply_chat_template(
            prompt.as_messages(),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate(self, prompt: Prompt) -> Generation:
        self.load_tokenizer()
        count = self.count_tokens(prompt)
        if count > self.config.max_input_tokens:
            raise ValueError("prompt excede max_input_tokens; nenhuma chamada enviada")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._calls += 1
        stem = self.audit_dir / f"{self._calls:05d}"
        request = self.client.payload(
            prompt.as_messages(), self.config.max_new_tokens, model=self.config.model_name
        )
        with stem.with_suffix(".request.json").open("x") as handle:
            json.dump(request, handle, ensure_ascii=False, indent=2)
        try:
            result = self.client.complete(
                prompt.as_messages(), max_tokens=self.config.max_new_tokens
            )
        except Exception as exc:
            stem.with_suffix(".error.json").write_text(json.dumps({"error": str(exc)}))
            raise
        stem.with_suffix(".response.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        if (
            result.get("model") != self.config.model_name
            or result.get("provider") != self.endpoint.response_provider
        ):
            raise ValueError("modelo/provedor da resposta diverge da PoC fixada")
        if result["finish_reason"] != "stop":
            raise ValueError("geracao incompleta; resposta preservada para auditoria")
        usage = result["usage"]
        if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
            raise ValueError("API reportou reasoning apesar de desativado")
        if usage.get("prompt_tokens") != count:
            raise ValueError(
                f"contagem local/API diverge: {count}/{usage.get('prompt_tokens')}; "
                "rodada interrompida para validar tokenizer/template"
            )
        completion = usage.get("completion_tokens")
        if not isinstance(completion, int) or completion <= 0:
            raise ValueError("API nao informou completion_tokens validos")
        return Generation(
            text=clean_generation(result["text"]),
            prompt_tokens=count,
            completion_tokens=completion,
            truncated_prompt=False,
            metadata={
                key: result[key]
                for key in ("id", "model", "provider", "finish_reason", "usage", "elapsed_seconds")
            }
            | {
                "tokenizer": self.endpoint.tokenizer,
                "seed_sent": False,
                "reasoning_enabled": False,
            },
        )
