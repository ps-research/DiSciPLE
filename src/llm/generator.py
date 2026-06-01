"""LLM code generator backed by Unsloth.

Loads ``unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`` (4-bit) and generates
programs from prompts. Unsloth is imported lazily inside ``load()`` so this
module can be imported in CPU-only contexts without touching the GPU.
"""
from __future__ import annotations

import re

from src.config import Config

# Matches a fenced code block: ```python ... ``` or ``` ... ``` (first one).
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


class LLMGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load the model + tokenizer (one-time). Imports Unsloth here, lazily."""
        if self.model is not None:
            return

        from unsloth import FastLanguageModel  # noqa: E402  (lazy, GPU-touching)

        dtype = self.config.llm.dtype
        if isinstance(dtype, str):
            import torch
            dtype = getattr(torch, dtype)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.llm.model,
            max_seq_length=self.config.llm.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.config.llm.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLMGenerator.load() must be called before generate().")

    def _gen_kwargs(self, overrides: dict) -> dict:
        llm = self.config.llm
        return {
            "do_sample": True,
            "temperature": overrides.get("temperature", llm.temperature),
            "top_p": overrides.get("top_p", llm.top_p),
            "max_new_tokens": overrides.get("max_new_tokens", llm.max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def generate(self, prompt: str, **overrides) -> str:
        """Generate a single completion; returns only the newly generated text."""
        self._require_loaded()
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        import torch
        with torch.no_grad():
            out = self.model.generate(input_ids=inputs, **self._gen_kwargs(overrides))

        new_tokens = out[0][inputs.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: list[str], **overrides) -> list[str]:
        """Left-padded batched generation; returns texts aligned to input order."""
        self._require_loaded()
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in prompts
        ]

        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)

            import torch
            with torch.no_grad():
                out = self.model.generate(**enc, **self._gen_kwargs(overrides))
        finally:
            self.tokenizer.padding_side = prev_side

        input_len = enc["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        return [
            self.tokenizer.decode(seq, skip_special_tokens=True).strip()
            for seq in new_tokens
        ]

    @staticmethod
    def extract_code(text: str) -> str:
        """Return code inside the first fenced block, else the stripped text."""
        m = _CODE_FENCE.search(text)
        if m:
            return m.group(1).strip()
        return text.strip()
