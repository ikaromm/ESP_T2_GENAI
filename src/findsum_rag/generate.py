"""Contratos de saida e tokenizacao; geracao exclusivamente via OpenRouter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import GenerationConfig
from .prompts import Prompt

# Rotulos que modelos instruidos costumam prefixar a resposta apesar da
# instrucao "output only the summary". Deixa-los no texto contaminaria as
# metricas: viram n-gramas sem ancoragem na fonte e deslocam o ROUGE.
PREAMBLE_RE = re.compile(
    r"^\s*(?:here\s+(?:is|are)\s+)?(?:a|the)?\s*summary\s*(?:of[^:\n]{0,80})?\s*:\s*",
    re.IGNORECASE,
)


def clean_generation(text: str) -> str:
    """Remove rotulos de preambulo e normaliza espacos da saida da LLM."""
    text = text.strip()
    # Um modelo pode emitir "Summary:" mais de uma vez (rotulo + eco do prompt).
    while True:
        stripped = PREAMBLE_RE.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped.strip()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass
class Generation:
    """Saida da LLM para um prompt, com contagens para auditoria de contexto."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    truncated_prompt: bool
    metadata: dict = field(default_factory=dict)


class PromptTokenizer:
    """Renderizacao e contagem local; nao carrega pesos nem gera texto."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("tokenizer nao carregado; chame load_tokenizer()")
        return self._tokenizer

    def render(self, prompt: Prompt) -> str:
        return self.tokenizer.apply_chat_template(
            prompt.as_messages(),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def count_tokens(self, prompt: Prompt) -> int:
        return len(self.tokenizer(self.render(prompt), add_special_tokens=False)["input_ids"])
