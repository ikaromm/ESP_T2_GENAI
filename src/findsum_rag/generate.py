"""Execucao da LLM de pesos abertos.

A geracao e mantida deterministica por padrao (`temperature=0`, greedy) porque o
estudo compara configuracoes entre si: amostragem introduziria variancia que se
confundiria com o efeito das tecnicas investigadas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


class Summarizer:
    """Gera resumos com um modelo causal do Hugging Face.

    A quantizacao em 4 bits e o padrao: um modelo de 7B em bf16 ocupa ~15 GB e
    nao cabe nos 12 GB da GPU disponivel, enquanto em NF4 ocupa ~5 GB e deixa
    folga para o cache de atencao dos prompts longos deste estudo.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Carrega tokenizer e modelo. Idempotente.

        Alguns checkpoints recentes (Qwen3.5, por exemplo) sao multimodais e
        declaram `*ForConditionalGeneration`. Aqui o uso e so de texto, por isso
        tenta-se primeiro a classe causal -- que carrega apenas a torre de texto
        e economiza VRAM -- com recuo para a classe declarada no checkpoint.
        """
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = getattr(torch, cfg.dtype)
        kwargs: dict = {"device_map": "auto", "dtype": dtype}

        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        try:
            self._model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
        except (ValueError, KeyError, OSError) as exc:
            from transformers import AutoModel

            print(
                f"AutoModelForCausalLM falhou para {cfg.model_name} ({exc}); "
                "recorrendo a AutoModel",
                flush=True,
            )
            self._model = AutoModel.from_pretrained(cfg.model_name, **kwargs)

        self._model.eval()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("modelo nao carregado; chame load()")
        return self._tokenizer

    def render(self, prompt: Prompt) -> str:
        """Aplica o chat template do modelo ao prompt."""
        return self.tokenizer.apply_chat_template(
            prompt.as_messages(), tokenize=False, add_generation_prompt=True
        )

    def count_tokens(self, prompt: Prompt) -> int:
        """Numero de tokens do prompt renderizado."""
        return len(self.tokenizer(self.render(prompt), add_special_tokens=False)["input_ids"])

    def generate(self, prompt: Prompt) -> Generation:
        """Gera o resumo para um prompt.

        Prompts que excedem `max_input_tokens` sao truncados pela ESQUERDA, o que
        preserva a instrucao final ("write the summary now") e o fim do contexto.
        O corte e sinalizado em `Generation.truncated_prompt` para que os casos
        afetados possam ser reportados em vez de passarem silenciosamente.
        """
        import torch

        self.load()
        cfg = self.config
        text = self.render(prompt)

        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        truncated = input_ids.shape[1] > cfg.max_input_tokens
        if truncated:
            input_ids = input_ids[:, -cfg.max_input_tokens :]
            encoded = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        encoded = {k: v.to(self._model.device) for k, v in dict(encoded).items()}
        torch.manual_seed(cfg.seed)

        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature if cfg.do_sample else None,
                top_p=cfg.top_p if cfg.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = encoded["input_ids"].shape[1]
        completion = output[0][prompt_len:]
        raw = self.tokenizer.decode(completion, skip_special_tokens=True)
        return Generation(
            text=clean_generation(raw),
            prompt_tokens=int(prompt_len),
            completion_tokens=int(completion.shape[0]),
            truncated_prompt=truncated,
        )
