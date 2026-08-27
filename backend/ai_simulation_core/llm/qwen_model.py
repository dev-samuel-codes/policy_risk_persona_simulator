"""
로컬 Qwen 모델을 로드하고 응답을 생성한다.

사용 예:

    from backend.ai_simulation_core.llm.qwen_model import LLM

    prompt = "정책에 대한 시민 반응을 생성해 주세요."
    llm = LLM()
    try:
        answer = llm.generate(prompt)
        print(answer)
    finally:
        llm.close()
"""

import gc
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CUDA_MAX_MEMORY = "9GiB"
CPU_MAX_MEMORY = "22GiB"


class LLM:
    def get_device(self) -> torch.device:
        """사용 가능한 가속기를 MPS, CUDA, CPU 순으로 선택한다."""
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def __init__(self) -> None:
        self.model_name = QWEN_MODEL_NAME
        self.device = self.get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # CUDA 경로에서는 BF16과 메모리 상한을 사용해 선택된 Qwen 모델의
        # 레이어와 버퍼를 GPU와 CPU에 자동 배치한다.
        model_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if self.device.type == "cuda":
            model_kwargs.update(
                {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                    "max_memory": {0: CUDA_MAX_MEMORY, "cpu": CPU_MAX_MEMORY},
                    "offload_buffers": True,
                }
            )

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self._closed = False

    def generate(self, prompt: str) -> str:
        """프롬프트에 대한 모델 응답을 문자열로 생성한다."""
        if self._closed:
            raise RuntimeError("종료된 LLM은 다시 사용할 수 없습니다.")

        messages = [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 아직 숫자 토큰으로 변환하지 않고 문자열 상태로 반환
            add_generation_prompt=True,  # 모델에게 모델이 답변할 차례라는 신호를 붙임
            enable_thinking=False,  # 추론 과정 출력을 비활성화
        )

        # 채팅 템플릿 문자열을 모델 입력 텐서로 변환한다.
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,  # 새로 생성할 수 있는 최대 토큰 수
                # 사실 검증이 필요한 구조화 JSON이므로 확률 샘플링을 사용하지 않는다.
                do_sample=False,
            )

        # 생성 결과에 포함된 입력 토큰을 잘라 새 응답만 남긴다.
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                model_inputs.input_ids,
                generated_ids,
            )
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,  # 특수 토큰 제거
        )[0]

        return response

    def close(self) -> None:
        """모델 참조와 가속기 캐시를 해제한다."""
        if self._closed:
            return

        self._closed = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        self.model = None
        self.tokenizer = None

        # PyTorch 텐서의 마지막 참조를 정리한 뒤 캐시된 VRAM도 반환한다.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
