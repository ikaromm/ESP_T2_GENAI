"""Tokenizer compativel empiricamente com 332 contagens Alibaba Qwen3.7 no dev.

Nao afirma identidade oficial entre modelos. O executor continua exigindo que
usage.prompt_tokens coincida com o preflight em cada geracao real.
"""

import hashlib

QWEN_TOKENIZER = "Qwen/Qwen3.8-27B"
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER_SHA256 = "ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0"
TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"


def load_qwen_tokenizer(path=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path or QWEN_TOKENIZER,
        revision=None if path else QWEN_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    if (
        not tokenizer.is_fast
        or not tokenizer.chat_template
        or hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        != TOKENIZER_SHA256
        or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != TEMPLATE_SHA256
    ):
        raise ValueError("tokenizer/template Qwen difere do validado no dev")
    return tokenizer
