# 로컬 LLM 로드 및 응답 생성 모듈

"""
다른 파일에서

llm = QwenLocalLLM()
anwser = llm.generate(prompt)
print(answer)

를 작성하면 사용가능
"""

import gc
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN_MODEL_NAME = "Qwen/Qwen3-8B"
CUDA_MAX_MEMORY = "9GiB"
CPU_MAX_MEMORY = "22GiB"


# LLM을 로컬에서 실행하기 위한 클래스
class LLM:
    # 윈도우, 맥에서 mps, cpu, gpu 구분
    def get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    # 클래스 초기화: LLM 객체를 만들 때 자동으로 실행
    def __init__(self) -> None:

        self.model_name = QWEN_MODEL_NAME
        self.device = self.get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Qwen3-8B의 원본 BF16 정밀도를 유지한다. RTX 5070 12GB에는 전체
        # 가중치가 들어가지 않으므로 일부 레이어와 버퍼만 CPU에 분산한다.
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

    # 사용자 질문을 받아서 모델 답변을 문자열로 반환
    def generate(self, prompt: str) -> str:
        if self._closed:
            raise RuntimeError("종료된 LLM은 다시 사용할 수 없습니다.")

        # 메세지 구성
        messages = [
            {
                "role": "system",  # 모델의 역할과 행동기준 설정 : content
                "content": "",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # 채팅 템플릿 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 아직 숫자 토큰으로 변환하지 않고 문자열 상태로 반환
            add_generation_prompt=True,  # 모델에게 모델이 답변할 차례라는 신호를 붙임
            enable_thinking=False,  # Thinking 내용 생략
        )

        # 토큰화: 문자열을 모델 입력용 텐서로 변환
        model_inputs = self.tokenizer(
            [text],  # 문자열을 하나를 리스트로 랩핑
            return_tensors="pt",
        ).to(self.device)

        # 답변 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,  # 새로 생성할 수 있는 최대 토큰 수
                # 사실 검증이 필요한 구조화 JSON이므로 확률 샘플링을 사용하지 않는다.
                do_sample=False,
            )

        # 입력 프롬프트 부분 제거
        generated_ids = [
            output_ids[len(input_ids) :]  # 입력 길이 만큼 앞부분 제거
            for input_ids, output_ids in zip(
                model_inputs.input_ids,
                generated_ids,
            )
        ]

        # 토큰을 문자열로 변환
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
