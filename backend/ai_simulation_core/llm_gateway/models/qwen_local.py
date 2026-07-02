# Qwen 로컬 추론 어댑터

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import BaseLLM


class QwenLocalLLM(BaseLLM):
    """
    Qwen 로컬 LLM 추론 어댑터.

    Hugging Face Qwen 모델을 로컬 환경에서 로딩,
    chat() 인터페이스로 텍스트 응답을 생성
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
        self.model_name = model_name
        self.device = self._get_device()

        print(f"사용 장치: {self.device}")

        self.tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)

        dtype = torch.float16 if self.device in {"mps", "cuda", "cpu"} else torch.float32

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
        )

        self.model.to(self.device)
        self.model.eval()

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def chat(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.device)

        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_token_id,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return response.strip()