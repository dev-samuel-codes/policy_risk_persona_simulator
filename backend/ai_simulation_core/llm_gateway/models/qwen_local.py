# 로컬 LLM 추론 어댑터

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLocalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
        self.model_name = model_name
        self.device = self._get_device()

        print(f"사용 장치: {self.device}")

        self.tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)

        dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32

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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
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

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
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


def main() -> None:
    llm = QwenLocalLLM()

    while True:
        prompt = input("\n사용자> ").strip()

        if prompt.lower() in {"q", "quit", "exit"}:
            print("종료합니다.")
            break

        if not prompt:
            continue

        response = llm.chat(prompt)
        print(f"\nQwen> {response}")


if __name__ == "__main__":
    main()